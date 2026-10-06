"""One head and two workers on one machine (PRD 18.1): found over mDNS, joined,
benchmarked, planned across both, loaded, READY. Then one worker dies and a new
one takes its place.

Sockets and multicast need the sandbox off. A `PermissionError` on bind is the
sandbox, not a bug.
"""

import asyncio
import copy

import pytest
import torch

from baton.head import serve
from baton.worker import probe
from baton.worker.daemon import Rejected, WorkerConfig, WorkerDaemon

GB = 1024**3


async def until(done, timeout=60.0) -> None:
    async with asyncio.timeout(timeout):
        while not done():
            await asyncio.sleep(0.05)


@pytest.fixture
def cluster(tiny_checkpoint, tmp_path, monkeypatch):
    """A head over the tiny checkpoint, and a factory for workers that can each
    hold three of its five units (four layers plus the head), so one worker is
    never enough and the plan must split."""
    directory, _, whole = tiny_checkpoint
    monkeypatch.setattr(serve, "JOIN_QUIET_S", 0.3)
    monkeypatch.setattr(probe, "memory_total_free", lambda backend: (64 * GB, 60 * GB))
    dtype = probe.probe_capabilities("cpu", tmp_path).compute_dtype
    layer_bytes = serve.read_metadata(str(directory)).planner_model("bf16").weight_bytes_per_layer
    budget = int(3.5 * layer_bytes / 0.8) * serve.DTYPE_BYTES[dtype] // 2

    lines: list[str] = []
    options = serve.ServeOptions(
        model=str(directory),
        quant="bf16",
        ctx=64,
        control_port=0,
        no_local_worker=True,
        min_workers=2,
    )
    head = serve.Head(options, echo=lines.append)

    def worker(name: str, where: str = "auto", **extra) -> WorkerDaemon:
        config = WorkerConfig(
            head=where, name=name, device="cpu", max_mem_bytes=budget, cache_dir=tmp_path, **extra
        )
        return WorkerDaemon(config)

    return head, worker, whole, lines


async def test_cluster_forms_over_mdns_and_survives_a_lost_worker(cluster) -> None:
    head, worker, whole, lines = cluster
    workers = {name: worker(name) for name in ("a", "b")}
    tasks = {"head": asyncio.create_task(head.run())}
    tasks |= {name: asyncio.create_task(w.run()) for name, w in workers.items()}
    try:
        await until(lambda: head.state == "ready")

        # Both workers hold a contiguous part, and together they hold the model.
        ring = head.registry.ring()
        assert sorted(w.name for w in ring) == ["a", "b"], "\n".join(lines)
        assert (ring[0].first_layer, ring[-1].last_layer) == (0, 3)
        assert ring[0].last_layer + 1 == ring[1].first_layer
        assert [w.holds_lm_head for w in ring] == [False, True]
        n1, nk = (workers[w.name] for w in ring)
        assert n1.loaded_rev == nk.loaded_rev == head.plan_rev == 1

        # The two resident shards compute exactly what the whole model computes.
        ids = torch.tensor([1, 5, 9, 2])
        reference = copy.deepcopy(whole).to(dtype=next(n1.stack.parameters()).dtype)
        with torch.no_grad():
            got = nk.stack(n1.stack(n1.stack.embed(ids)))
            assert torch.equal(got, reference(reference.embed(ids)))

        # The ring link is open and authenticated: N1 dialled Nk's data port.
        await until(lambda: len(nk._inbound) == 1, timeout=5)
        assert n1._next is not None and nk._next is None

        # Nk dies. The head notices, leaves READY, and waits for a second worker.
        tasks[nk.name].cancel()
        await until(lambda: head.registry.get(nk.name).state == "lost", timeout=15)
        assert head.state != "ready"

        # A replacement joins by address (no mDNS). The head plans again and reloads.
        workers["c"] = worker("c", f"127.0.0.1:{head.control_port}")
        tasks["c"] = asyncio.create_task(workers["c"].run())
        await until(lambda: head.state == "ready" and head.plan_rev == 2)
        assert sorted(w.name for w in head.registry.ring()) == sorted([n1.name, "c"])
        assert workers["c"].stack is not None and n1.loaded_rev == 2
    finally:
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)


async def test_a_wrong_token_is_refused_and_not_retried(cluster) -> None:
    head, worker, _, _ = cluster
    head.options.token = "s3cret"
    await head.start_servers()
    try:
        intruder = worker("x", f"127.0.0.1:{head.control_port}", token="guess")
        with pytest.raises(Rejected, match="auth"):
            await asyncio.wait_for(intruder.run(), timeout=10)
        assert head.registry.get("x") is None
    finally:
        await head.close()
