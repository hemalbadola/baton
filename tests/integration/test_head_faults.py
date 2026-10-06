"""The head against workers that misbehave. Each test is one reviewer finding
(BAT-14 to BAT-21) turned into a check.

The workers here are fakes: a socket and hand-written frames, so a test can
answer late, answer wrong, or not answer.
"""

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from baton.common.net import read_frame, send_frame
from baton.head import serve
from baton.head.registry import Capabilities, Health, Registry

GB = 1024**3


class Fake:
    """A worker that says only what the test tells it to say."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def join(self, head, usable: int, benched: bool = True, data_addr: str | None = None):
        self.reader, self.writer = await asyncio.open_connection("127.0.0.1", head.control_port)
        port = self.writer.get_extra_info("sockname")[1]
        caps = {"backend": "cpu", "usable_bytes": usable, "compute_dtype": "bf16"}
        if benched:
            caps["bench"] = {"t_dec_ms": 1.0, "t_pre_ms": 5.0}
        hello = {"t": "hello", "name": self.name, "token": "", "version": "0", "caps": caps}
        await send_frame(self.writer, hello | {"data_addr": data_addr or f"127.0.0.1:{port}"})
        reply, _ = await read_frame(self.reader)
        self.beat = asyncio.create_task(self._beat())
        return reply

    async def _beat(self) -> None:
        with contextlib.suppress(OSError):
            while True:
                await self.send(t="health", mem_free=0, kv_used=0, queue_depth=0, active_reqs=0)
                await asyncio.sleep(1)

    async def send(self, **meta) -> None:
        await send_frame(self.writer, meta)

    async def recv(self, timeout: float = 5.0) -> dict:
        async with asyncio.timeout(timeout):
            return (await read_frame(self.reader))[0]


async def until(done, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not done():
            await asyncio.sleep(0.02)


@pytest.fixture
def running(tiny_checkpoint, monkeypatch):
    """`start(**options)` runs a head over the tiny checkpoint, with no mDNS."""
    directory = str(tiny_checkpoint[0])
    monkeypatch.setattr(serve, "BENCH_TIMEOUT_S", 0.3)
    layer = serve.read_metadata(directory).planner_model("bf16").weight_bytes_per_layer
    tasks = []

    async def start(**extra):
        options = serve.ServeOptions(
            model=directory, quant="bf16", ctx=64, control_port=0, no_local_worker=True, wait_s=0.2
        )
        for key, value in extra.items():
            setattr(options, key, value)
        head = serve.Head(options, echo=lambda line: None)

        async def no_mdns() -> None:
            return None

        head.advertise = no_mdns
        tasks.append(asyncio.create_task(head.run()))
        await until(lambda: head.state == "idle")
        return head

    # `half` holds 3 of the model's 5 units, so one such worker is never enough.
    yield SimpleNamespace(start=start, half=int(3.5 * layer / 0.8), tasks=tasks)
    for task in tasks:
        task.cancel()


def control(peer: str, local: str) -> SimpleNamespace:
    ends = {"peername": (peer, 5000), "sockname": (local, 7711)}
    return SimpleNamespace(get_extra_info=ends.get)


def test_a_remote_worker_is_never_told_to_dial_loopback() -> None:
    """BAT-14. The local worker reports 127.0.0.1. A remote peer must get the
    head's LAN address for it, the one its own control socket already uses."""
    registry = Registry()
    caps = Capabilities("cpu", GB, 1.0, 5.0)
    local = registry.hello("local", ("127.0.0.1", 9001), caps, control("127.0.0.1", "127.0.0.1"))
    remote = registry.hello(
        "remote", ("192.168.1.7", 9002), caps, control("192.168.1.7", "192.168.1.5")
    )
    assert serve.Head._ring_addr(remote, local) == "192.168.1.5:9001"
    assert serve.Head._ring_addr(local, remote) == "192.168.1.7:9002"
    assert serve.Head._ring_addr(local, local) == "127.0.0.1:9001"  # two workers, one machine


async def test_a_worker_that_joins_during_the_bench_is_not_missed(running) -> None:
    """BAT-15. The join lands while `a` is still benchmarking."""
    head = await running.start()
    a, b = Fake("a"), Fake("b")
    await a.join(head, running.half, benched=False)
    assert (await a.recv())["t"] == "bench"
    await b.join(head, running.half, benched=False)
    await a.send(t="bench_result", t_dec_ms=1.0, t_pre_ms=5.0, quant="bf16", dtype="bf16")
    # `a` alone is infeasible. The head must come back for `b`.
    assert (await b.recv())["t"] == "bench"


async def test_an_error_from_outside_the_ring_does_not_fail_the_load(running) -> None:
    """BAT-16. `c` never answers the bench, then reports an error mid-load."""
    head = await running.start()
    a, c = Fake("a"), Fake("c")
    await a.join(head, GB)
    await c.join(head, GB, benched=False)
    load = await a.recv()
    assert load["t"] == "load" and load["range"] == [0, 3]
    await c.send(t="error", code="bench_failed", message="late")
    await a.send(
        t="error", code="load_failed", message="from an old plan", rev=load["plan_rev"] - 1
    )
    await a.send(t="loaded", rev=load["plan_rev"], resident_bytes=1, seconds=0.1, from_cache=False)
    await until(lambda: head.state == "ready")


async def test_a_standby_worker_with_a_stale_range_cannot_take_ready_down(running) -> None:
    """BAT-17. `b` held layers under an old plan, came back, and left again."""
    head = await running.start()
    caps = Capabilities("cpu", GB, 1.0, 5.0)
    head.registry.hello("b", ("10.0.0.2", 9000), caps, None)
    head.registry.assign("b", 0, 1, False)
    head.registry.hello("b", ("10.0.0.2", 9000), caps, None)  # reconnect: standby, range kept
    head.state = "ready"
    head._lose("b", "gone again")
    assert head.state == "ready"


async def test_a_second_machine_with_the_same_name_is_refused(running) -> None:
    """BAT-18. Two laptops with one hostname must not share a registry entry."""
    head = await running.start(min_workers=3)
    first, second = Fake("laptop"), Fake("laptop")
    assert (await first.join(head, GB, data_addr="10.0.0.1:5001"))["t"] == "welcome"
    reply = await second.join(head, GB, data_addr="10.0.0.2:5002")
    assert (reply["t"], reply["code"]) == ("error", "name_in_use")
    assert head.registry.get("laptop").data_addr == "10.0.0.1:5001"


async def test_a_lost_ring_member_ends_the_load_at_once(running) -> None:
    """BAT-19. `b` dies mid-load. The head must not wait for `a` to finish."""
    head = await running.start(min_workers=2)
    a, b = Fake("a"), Fake("b")
    await a.join(head, running.half)
    await b.join(head, running.half)
    assert (await a.recv())["t"] == (await b.recv())["t"] == "load"
    b.beat.cancel()
    b.writer.close()
    assert (await a.recv())["t"] == "unload"
    assert head.state == "idle"


def test_non_finite_numbers_from_the_wire_are_dropped() -> None:
    """BAT-21. The frame decoder accepts JSON `NaN` and `Infinity`."""
    registry = Registry()
    registry.hello("a", ("10.0.0.1", 9000), Capabilities("cpu", GB, 0.0, 0.0), None)
    registry.on_bench("a", float("nan"), 5.0)
    registry.on_bench("a", 1.0, float("inf"))
    assert registry.devices() == []
    assert Health.from_frame({"mem_free": float("inf")}, at=0.0).free_bytes == 0


async def test_the_plan_leaves_room_for_the_embedding_table_on_n1(running) -> None:
    """BAT-32. Three workers with room for 4.2, 1.6 and 1.3 layers. The planner
    alone makes the smallest one N1 with one layer, and the embedding table
    then does not fit beside it. The head must move N1 to a worker with room."""
    head = await running.start(kv_fraction=0.0)
    model = head.metadata.planner_model("bf16")
    layer, embed = model.weight_bytes_per_layer, head.metadata.embed_bytes
    room = {"a": int(4.2 * layer), "b": int(1.6 * layer), "c": int(1.3 * layer)}
    speed = {"a": 1.0, "b": 9.0, "c": 4.0}
    fleet = serve.Fleet(
        tuple(serve.Device(n, room[n], speed[n], 5 * speed[n]) for n in ("a", "b", "c"))
    )

    def over(plan_) -> list[str]:
        return [
            a.name
            for a in plan_.assignments
            if a.layers * layer
            + (embed if a.position == 0 else 0)
            + (model.lm_head_bytes if a.holds_lm_head else 0)
            > room[a.name]
        ]

    alone = serve.plan(fleet, model, serve.Workload(kv_fraction=0.0, ctx=64))
    assert over(alone) == ["c"]  # the fault this ticket is about
    made = head.make_plan(fleet)
    assert made.feasible and over(made) == []
    assert made.assignments[0].name == "b"
