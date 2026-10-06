"""The worker against a head that misbehaves. Each test is one reviewer finding
(BAT-24 to BAT-27) turned into a check. The head here is a fake: a socket and
hand-written frames."""

import asyncio

from baton.common.net import read_frame, send_frame
from baton.head import serve
from baton.worker import daemon, probe
from baton.worker.daemon import WorkerConfig, WorkerDaemon

GB = 1024**3


async def until(done, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not done():
            await asyncio.sleep(0.02)


def worker(port: int, tmp_path, monkeypatch) -> WorkerDaemon:
    monkeypatch.setattr(probe, "memory_total_free", lambda backend: (64 * GB, 60 * GB))
    monkeypatch.setattr(daemon, "CONTROL_RETRY_S", 0.05)
    config = WorkerConfig(head=f"127.0.0.1:{port}", name="w", device="cpu", cache_dir=tmp_path)
    return WorkerDaemon(config)


async def welcome(reader, writer) -> None:
    hello, _ = await read_frame(reader)
    assert hello["t"] == "hello"
    await send_frame(writer, {"t": "welcome", "cluster_id": "c1", "your_name": hello["name"]})


async def test_a_bad_handshake_is_retried_not_fatal(tmp_path, monkeypatch) -> None:
    """BAT-24. The first two connections fail in the two ways a real network
    does: a reset, then a port that is not Baton. The third is a head."""
    seen = 0

    async def flaky(reader, writer) -> None:
        nonlocal seen
        seen += 1
        if seen == 1:
            writer.close()
        elif seen == 2:
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await writer.drain()
            writer.close()
        else:
            await welcome(reader, writer)
            await reader.read()

    server = await asyncio.start_server(flaky, "127.0.0.1", 0)
    w = worker(server.sockets[0].getsockname()[1], tmp_path, monkeypatch)
    task = asyncio.create_task(w.run())
    try:
        await until(lambda: w.cluster_id == "c1")
        assert seen == 3 and not task.done()
    finally:
        task.cancel()
        server.close()
        await asyncio.gather(task, return_exceptions=True)


async def test_unload_after_load_leaves_nothing_resident(tiny_checkpoint, tmp_path, monkeypatch):
    """BAT-26, BAT-27. `load` then `unload` arrive in one read. The worker must
    end empty, and must not report `loaded` for a plan the head gave up."""
    directory = str(tiny_checkpoint[0])
    meta = serve.read_metadata(directory)
    load = {
        "t": "load",
        "plan_rev": 1,
        "model": directory,
        "spec": meta.spec.to_dict(),
        "range": [0, 3],
        "quant": "bf16",
        "roles": {"embed": True, "head": True},
        "ctx_max": 64,
        "kv_budget_bytes": 0,
        "next_node": "",
        "head_data_addr": "",
        "index": meta.index,
        "headers": {},
    }
    replies: list[str] = []

    async def head(reader, writer) -> None:
        await welcome(reader, writer)
        await send_frame(writer, load)
        await send_frame(writer, {"t": "unload"})
        await send_frame(
            writer, {"t": "bench", "spec": load["spec"], "quant": "bf16", "dtype": "fp32"}
        )
        while True:
            frame, _ = await read_frame(reader)
            replies.append(frame["t"])

    server = await asyncio.start_server(head, "127.0.0.1", 0)
    w = worker(server.sockets[0].getsockname()[1], tmp_path, monkeypatch)
    task = asyncio.create_task(w.run())
    try:
        # The bench is queued behind both, so its reply marks the end of the pair.
        await until(lambda: "bench_result" in replies)
        assert w.stack is None and w.loaded_rev is None and w.resident_bytes == 0
        assert "loaded" not in replies and "error" not in replies
    finally:
        task.cancel()
        server.close()
        await asyncio.gather(task, return_exceptions=True)
