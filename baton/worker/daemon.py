"""Worker daemon: startup, control loop, reconnect (PRD 6.1, 11).

The daemon is the only part of the worker that talks to the head. It runs an
asyncio loop on the main thread and never touches the model: every frame goes
to `engine.ForwardEngine.handle` through `asyncio.to_thread`, one at a time.
That split is what keeps the 2 s heartbeat alive while a prefill chunk runs.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import gc
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Coroutine
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from baton import __version__
from baton.common.framing import FrameError
from baton.common.net import (
    CONTROL_CONNECT_TIMEOUT,
    CONTROL_PORT,
    AuthError,
    LinkClosed,
    accept_data_link,
    close_writer,
    connect_with_retry,
    format_addr,
    open_data_link,
    parse_addr,
    read_frame,
    send_frame,
)
from baton.worker.probe import (
    Backend,
    BenchResult,
    Capabilities,
    MemoryBudget,
    memory_budget,
    pick_device,
    probe_capabilities,
    run_bench,
    torch_dtype,
)

if TYPE_CHECKING:  # imported for types only
    from baton.model.layers import DecoderStack
    from baton.worker.engine import ForwardEngine

log = logging.getLogger("baton.worker")

#: Heartbeat period, `health` every 2 s (PRD 6.1 step 6).
HEALTH_INTERVAL_S = 2.0

#: Control reconnect period after a drop (PRD 6.1 reconnect rule).
CONTROL_RETRY_S = 2.0

#: How long a worker holds its shard while the head is gone (PRD 6.1, 11.3).
SHARD_HOLD_S = 30.0

#: mDNS browse budget before falling back to the retry loop (PRD 6.1 step 4).
MDNS_BROWSE_S = 10.0

#: Head discovery retry period once mDNS and `--head` have both failed.
DISCOVERY_RETRY_S = 5.0

#: Data connection budget after `load` (PRD 6.4 step 4, 8.6).
DATA_CONNECT_S = 30.0

#: mDNS service type the head advertises.
SERVICE_TYPE = "_baton._tcp.local."

#: `load_progress` period while a shard loads (PRD 6.4 step 2).
PROGRESS_INTERVAL_S = 0.5

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _cache_name(model: object) -> str:
    """A repo id as it is. A checkpoint directory by its name, never by its path:
    a path would escape the cache root."""
    text = str(model)
    return (
        text
        if "/" in text and not Path(text).is_absolute() and ".." not in text
        else Path(text).name
    )


class Rejected(Exception):
    """The head refused `hello`. `code` says why: `auth`, `name_in_use`, `bad_hello`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(slots=True)
class WorkerConfig:
    """Parsed `baton worker` arguments (PRD 6.1)."""

    head: str = "auto"
    """`auto` for mDNS, else `HOST` or `HOST:PORT`."""

    max_mem_bytes: int | None = None
    device: str = "auto"
    name: str | None = None
    """Defaults to the hostname. The reconnect path reuses it, so the head can
    match a returning worker to its loaded shard."""

    token: str | None = None
    data_port: int = 0
    """0 means an OS-assigned ephemeral port."""

    cache_dir: Path = Path.home() / ".cache" / "baton"
    wire_dtype: str = "bf16"


class SleepInhibitor:
    """Keeps the laptop awake while the worker holds a shard (PRD 6.1 step 7).

    One implementation per platform: `caffeinate -i -w <pid>` on macOS,
    `SetThreadExecutionState` through `ctypes` on Windows, `systemd-inhibit
    --what=idle --who=baton` on Linux when it is present. An unsupported
    platform is not an error: the worker warns once and runs on.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> Self:
        try:
            if sys.platform == "darwin":
                self._proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            elif sys.platform == "win32":
                self._windows(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
            elif shutil.which("systemd-inhibit"):
                self._proc = subprocess.Popen(
                    # `tail --pid` ends when this process does, however it dies.
                    ["systemd-inhibit", "--what=idle", "--who=baton"]
                    + ["tail", f"--pid={os.getpid()}", "-f", "/dev/null"]
                )
            else:
                log.warning("no way to prevent sleep on this platform; keep the machine awake")
        except OSError as exc:
            log.warning("cannot prevent sleep: %s", exc)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        elif sys.platform == "win32":
            self._windows(_ES_CONTINUOUS)

    @staticmethod
    def _windows(flags: int) -> None:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]


async def _reachable(host: str, port: int, timeout_s: float = 1.0) -> bool:
    """True when a TCP connection to `host:port` opens. The probe closes it at once."""
    try:
        async with asyncio.timeout(timeout_s):
            _, writer = await asyncio.open_connection(host, port)
    except (OSError, TimeoutError):
        return False
    await close_writer(writer)
    return True


async def discover_head(head: str, timeout_s: float = MDNS_BROWSE_S) -> tuple[str, int] | None:
    """Resolve the head address (PRD 6.1 step 4).

    An explicit `HOST[:PORT]` is parsed and returned without a browse. `auto`
    browses mDNS for `_baton._tcp.local.` for `timeout_s`. Returns None when
    nothing answers: the caller retries every 5 s forever and prints one line
    per minute, because a worker left running before the head starts is the
    normal case, not a failure.

    A head on a machine with Wi-Fi, Ethernet and a VPN advertises every one of
    its addresses. Only one of them may be routable from here, so each is
    dialled and the first that answers wins.
    """
    if head != "auto":
        return parse_addr(head) if ":" in head else (head, CONTROL_PORT)

    from zeroconf import IPVersion, ServiceStateChange
    from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

    names: asyncio.Queue[str] = asyncio.Queue()

    def on_change(zeroconf: object, service_type: str, name: str, state_change: object) -> None:
        if state_change is not ServiceStateChange.Removed:
            names.put_nowait(name)

    mdns = AsyncZeroconf()
    browser = AsyncServiceBrowser(mdns.zeroconf, SERVICE_TYPE, handlers=[on_change])
    try:
        async with asyncio.timeout(timeout_s):
            while True:
                info = AsyncServiceInfo(SERVICE_TYPE, await names.get())
                if not await info.async_request(mdns.zeroconf, 3000) or info.port is None:
                    continue
                for host in info.parsed_addresses(IPVersion.V4Only):
                    if await _reachable(host, info.port):
                        return host, info.port
    except TimeoutError:
        return None
    finally:
        await browser.async_cancel()
        await mdns.async_close()


class WorkerDaemon:
    """The control loop.

    Owns the control connection, the heartbeat, the data server, and the
    resident shard. Every method here is a coroutine on the main thread and
    must stay sub-millisecond: anything longer runs on another thread.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self.name = config.name or socket.gethostname().split(".")[0]
        self.backend: Backend | None = None
        self.caps: Capabilities | None = None
        self.budget: MemoryBudget | None = None
        self.engine: ForwardEngine | None = None
        self.stack: DecoderStack | None = None
        self.resident_bytes = 0
        self.cluster_id = ""
        self.data_port = 0
        self.loaded_rev: int | None = None
        """Plan revision of the resident shard, echoed in `health` so a
        returning worker can skip reloading (PRD 6.1, 11.3). None means nothing
        is resident, which is also how `handle_load` knows to skip the free
        step."""

        self.reconnect_deadline: float | None = None
        """`monotonic()` at which the 30 s shard hold expires. None while the
        control connection is up. These two fields are the whole worker state:
        every branch in PRD 6.1 and 11.3 turns on one of them."""

        self._writer: asyncio.StreamWriter | None = None
        self._send_lock = asyncio.Lock()
        self._device_lock = asyncio.Lock()
        """Bench, load and unload hold the device one at a time."""

        self._epoch = 0
        """Bumped by every `load`, `unload`, lost control link and `stop`. A load
        carries the value it started with, and gives up without a reply the
        moment the two differ: a newer instruction, or a new session, owns the
        device from then on."""

        self._data_server: asyncio.Server | None = None
        self._next: asyncio.StreamWriter | None = None
        """The outbound ring link to Ni+1. It belongs to the resident plan."""

        self._inbound: set[asyncio.StreamWriter] = set()
        self._tasks: set[asyncio.Task[None]] = set()

        self._inbox: asyncio.Queue[tuple[dict[str, Any], bytes]] = asyncio.Queue()
        """Data-plane frames for the engine, from the ring links and from the head."""

        self._pump: asyncio.Task[None] | None = None
        self._next_addr = ""
        self._compute_ms: collections.deque[float] = collections.deque(maxlen=64)
        """Milliseconds of the last decode frames on the compute thread, for the dashboard."""

    async def run(self) -> None:
        """Run the startup sequence, then serve until stopped.

        Steps in PRD 6.1 order: pick device, probe, start the data server,
        find the head, open the control connection, send `hello`, start the
        heartbeat, inhibit sleep, wait for `load`.
        """
        cfg = self.config
        self.backend = pick_device(cfg.device)
        self.budget = memory_budget(self.backend, cfg.max_mem_bytes)
        self.caps = probe_capabilities(self.backend, cfg.cache_dir)
        log.info(
            "%s: %s %s, %.1f GB usable, link %s",
            self.name,
            self.backend,
            self.caps.compute_dtype,
            self.budget.usable_bytes / 1024**3,
            self.caps.link,
        )
        await self.start_data_server()
        addr: tuple[str, int] | None = None
        try:
            with SleepInhibitor():
                while True:
                    addr = addr or await self._find_head()
                    try:
                        reader = await self.connect_control(addr)
                    except Rejected as exc:
                        if exc.code != "name_in_use":
                            raise
                        # The head may still hold this name for a socket it has
                        # not timed out yet. That clears by itself in 6 s.
                        log.warning("the head refused the name for now: %s", exc)
                        await asyncio.sleep(DISCOVERY_RETRY_S)
                        continue
                    except (OSError, FrameError, ValueError) as exc:  # TimeoutError is an OSError
                        # A reset during `hello`, a late `welcome`, a port that is
                        # not Baton. None of these may end the worker (PRD 8.6).
                        log.warning("handshake with the head failed: %r", exc)
                        deadline = self.reconnect_deadline
                        if deadline is None or time.monotonic() >= deadline:
                            await self.handle_unload()  # past the shard hold, or nothing held
                            addr = self.reconnect_deadline = None
                        await asyncio.sleep(CONTROL_RETRY_S)
                        continue
                    with contextlib.suppress(LinkClosed, FrameError, OSError, ValueError):
                        await self._serve(reader)
                    await self.on_control_lost()
                    if self.loaded_rev is None:
                        addr = None  # nothing to hold, so look for a head again
        finally:
            await self.stop()

    async def _find_head(self) -> tuple[str, int]:
        """Step 4, forever: every 5 s, one line a minute."""
        last_log = 0.0
        while True:
            addr = await discover_head(self.config.head)
            if addr is not None:
                return addr
            if time.monotonic() - last_log >= 60.0:
                last_log = time.monotonic()
                log.warning("no head found on the network yet; still looking")
            await asyncio.sleep(DISCOVERY_RETRY_S)

    async def start_data_server(self) -> tuple[str, int]:
        """Bind the data server on all LAN interfaces. Returns `(ip, port)`.

        Started before the head is found, because `hello` must carry the real
        `data_addr` and an ephemeral port is only known after the bind. The ip
        that peers dial is filled in by `connect_control`: it is the local end
        of the control socket, the one interface known to reach this LAN.
        """
        self._data_server = await asyncio.start_server(
            self._on_data, "0.0.0.0", self.config.data_port
        )
        self.data_port = self._data_server.sockets[0].getsockname()[1]
        return "0.0.0.0", self.data_port

    async def _on_data(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Accept one ring link (PRD 8.5) and queue every frame for the engine."""
        try:
            await accept_data_link(reader, writer, self.cluster_id, self.config.token or "")
            self._inbound.add(writer)
            while True:
                meta, payload = await read_frame(reader)
                self._inbox.put_nowait((meta, payload))
        except (AuthError, LinkClosed, TimeoutError, OSError) as exc:
            log.debug("data link refused or closed: %s", exc)
        finally:
            self._inbound.discard(writer)
            await close_writer(writer)

    async def connect_control(self, addr: tuple[str, int]) -> asyncio.StreamReader:
        """Open the control connection and send `hello` (PRD 6.1 step 5, 8.5).

        Retries forever on refusal: every second at first contact, every 2 s
        with a 30 s budget while a shard is held (the reconnect rule). Raises
        `Rejected` on a wrong token or a name in use, which is a configuration
        mistake and will not fix itself, and `TimeoutError` when the shard hold
        runs out.
        """
        budget = None
        if self.reconnect_deadline is not None:
            budget = max(0.0, self.reconnect_deadline - time.monotonic())
        reader, writer = await connect_with_retry(
            *addr,
            total_timeout=budget,
            retry_interval=CONTROL_RETRY_S if budget is not None else 1.0,
            label="head",
        )
        assert self.caps is not None and self.budget is not None
        caps = asdict(self.caps) | {"usable_bytes": self.budget.usable_bytes}
        try:
            local_ip = writer.get_extra_info("sockname")[0]
            await send_frame(
                writer,
                {
                    "t": "hello",
                    "name": self.name,
                    "token": self.config.token or "",
                    "version": __version__,
                    "caps": caps,
                    "data_addr": format_addr(local_ip, self.data_port),
                },
            )
            async with asyncio.timeout(CONTROL_CONNECT_TIMEOUT):
                reply, _ = await read_frame(reader)
        except BaseException:
            await close_writer(writer)
            raise
        if reply.get("t") != "welcome":
            await close_writer(writer)
            raise Rejected(str(reply.get("code", "refused")), str(reply.get("message", reply)))

        self.cluster_id = str(reply.get("cluster_id", ""))
        self.name = str(reply.get("your_name") or self.name)
        self.reconnect_deadline = None
        self._writer = writer
        if self.loaded_rev is not None and reply.get("plan_rev") != self.loaded_rev:
            await self.handle_unload()  # the head moved on while this worker was away
        log.info("joined cluster %s at %s as %s", self.cluster_id, format_addr(*addr), self.name)
        return reader

    async def _serve(self, reader: asyncio.StreamReader) -> None:
        """Heartbeat plus the frame loop, until the control connection drops."""
        beat = asyncio.create_task(self.heartbeat())
        try:
            while True:
                meta, _ = await read_frame(reader)
                await self.handle_control(meta)
        finally:
            beat.cancel()

    async def _send(self, meta: dict[str, Any]) -> None:
        """Write one control frame. A lock keeps frames from several tasks whole."""
        if self._writer is None:
            return
        async with self._send_lock:
            await send_frame(self._writer, meta)

    def _spawn(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._reap)

    def _reap(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.warning("worker task failed: %r", task.exception())

    async def heartbeat(self) -> None:
        """Send `health` every 2 s until the connection drops (PRD 8.3).

        Fields come from daemon state, never from a device query. Six seconds
        of silence makes the head mark this worker lost (PRD 8.6), so this
        coroutine must never await anything that can block on the device.
        """
        assert self.budget is not None
        try:
            while True:
                await self._send(
                    {
                        "t": "health",
                        "mem_free": max(0, self.budget.usable_bytes - self.resident_bytes),
                        "kv_used": self.engine.kv.used_bytes if self.engine else 0,
                        "queue_depth": self._inbox.qsize(),
                        "active_reqs": self.engine.active_reqs if self.engine else 0,
                        "loaded_rev": self.loaded_rev,
                        "compute_ms": list(self._compute_ms)[-12:],
                    }
                )
                await asyncio.sleep(HEALTH_INTERVAL_S)
        except OSError:
            # The head is gone. Closing the socket wakes the frame loop, which
            # owns the reconnect.
            if self._writer is not None:
                self._writer.close()

    async def handle_control(self, frame: dict[str, Any]) -> None:
        """Dispatch one head-to-worker message (PRD 8.3).

        `bench` and `load` run as tasks so the frame loop keeps reading. A type
        this build does not handle is answered with `error`, never ignored: a
        head that waits for a reply must not wait forever.
        """
        kind = frame.get("t")
        if kind == "bench":
            self._spawn(self._bench_and_reply(frame))
        elif kind == "load":
            self._epoch += 1
            self._spawn(self.handle_load(frame, self._epoch))
        elif kind in ("prompt", "next", "abort", "release"):
            self._inbox.put_nowait((frame, b""))
        elif kind == "unload":
            # A task, like `load`: tasks start in creation order and the device
            # lock is first-in first-out, so `load` then `unload` cannot swap.
            self._epoch += 1
            self._spawn(self.handle_unload())
        else:
            await self._send(
                {"t": "error", "code": "unsupported", "message": f"no handler for {kind!r}"}
            )

    async def _bench_and_reply(self, frame: dict[str, Any]) -> None:
        try:
            result = await self.handle_bench(frame["spec"], frame["quant"], frame["dtype"])
        except Exception as exc:  # noqa: BLE001 - reported to the head, which owns the decision
            await self._send({"t": "error", "code": "bench_failed", "message": repr(exc)})
            return
        await self._send({"t": "bench_result", **asdict(result)})

    async def handle_bench(self, spec: object, quant: str, compute_dtype: str) -> BenchResult:
        """Run the benchmark off the control loop and return the result.

        The benchmark holds the device for up to 10 s. Running it here would
        stop the heartbeat and get this worker marked lost at 6 s.

        ponytail: `asyncio.to_thread` behind the device lock. Once the forward
        engine runs requests, the bench must queue on its compute thread.
        """
        assert self.backend is not None and self.caps is not None
        async with self._device_lock:
            result = await asyncio.to_thread(run_bench, spec, quant, compute_dtype, self.backend)
        self.caps = replace(self.caps, bench=result)
        return result

    async def handle_load(self, msg: dict[str, Any], epoch: int | None = None) -> None:
        """Load a shard (PRD 6.4).

        Frees any resident plan first, fetches the layer range one tensor at a
        time, reports `load_progress` every 500 ms, opens the data connection
        to `next_node` within 30 s, then replies `loaded`. Any failure is
        reported as `error` and leaves nothing resident.

        A load that was superseded (see `_epoch`) stops after the tensor in
        hand, frees what it built, and says nothing: its reply would land on a
        plan, or a session, that never asked for it.
        """
        if epoch is None:
            epoch = self._epoch

        def superseded() -> bool:
            return epoch != self._epoch

        def on_progress(*step: int) -> None:  # runs on the loader thread
            if superseded():
                raise RuntimeError("load superseded")
            progress[:] = step

        from baton.model import safetensors_io as sio
        from baton.model.cache import BlobCache
        from baton.model.spec import ModelSpec
        from baton.worker.engine import ForwardEngine, KVPool, ShardTooLarge, load_shard

        assert self.backend is not None and self.caps is not None and self.budget is not None
        progress = [0, 0, 0]
        async with self._device_lock:
            if superseded():
                return
            self._free()
            # Measured again: the laptop is not as empty as it was at startup.
            self.budget = memory_budget(self.backend, self.config.max_mem_bytes)
            started = time.perf_counter()

            async def report() -> None:
                while True:
                    await asyncio.sleep(PROGRESS_INTERVAL_S)
                    done, total, fetched = progress
                    await self._send(
                        {"t": "load_progress", "done": done, "total": total, "bytes": fetched}
                    )

            reporter = asyncio.create_task(report())
            source = None
            try:
                # `load` arrives over the wire: every field is checked by use,
                # and anything malformed lands in the handler below.
                if msg["quant"] != "bf16":
                    raise ValueError(f"quant {msg['quant']!r} is not loadable yet (BAT-10)")
                first, last = (int(layer) for layer in msg["range"])
                spec = ModelSpec(**msg["spec"])
                dtype = torch_dtype(self.caps.compute_dtype)
                url = str(msg.get("source_url") or "")
                if url:  # the head holds the files (BAT-43)
                    source = sio.RangeSource(url)
                else:
                    source = sio.open_source(str(msg["model"]), msg.get("hf_token"))
                cache = None
                # No copy for a local checkpoint, or beside a head on this machine.
                if isinstance(source, sio.RangeSource) and "//127.0.0.1" not in url:
                    cache = BlobCache(
                        self.config.cache_dir, _cache_name(msg["model"]), "main", source
                    )
                stack, resident = await asyncio.to_thread(
                    load_shard,
                    spec,
                    first,
                    last,
                    embed=bool(msg["roles"]["embed"]),
                    head=bool(msg["roles"]["head"]),
                    source=source,
                    index=msg["index"],
                    ctx_max=int(msg["ctx_max"]),
                    device=self.backend,
                    dtype=dtype,
                    budget_bytes=self.budget.usable_bytes,
                    on_progress=on_progress,
                    cache=cache,
                )
                self.stack, self.resident_bytes = stack, resident
                self._next_addr = str(msg["next_node"])
                if msg["next_node"]:
                    _, self._next = await open_data_link(
                        str(msg["next_node"]),
                        self.cluster_id,
                        self.config.token or "",
                        total_timeout=DATA_CONNECT_S,
                    )
                per_token = 2 * spec.n_kv_heads * spec.head_dim * dtype.itemsize
                engine = ForwardEngine(
                    stack,
                    KVPool(int(msg["kv_budget_bytes"]), stack.n_local_layers, per_token),
                    device=self.backend,
                    dtype=dtype,
                    wire=self.config.wire_dtype,
                    ctx_max=int(msg["ctx_max"]),
                )
                rev = int(msg["plan_rev"])
                if superseded():
                    raise RuntimeError("load superseded")
            except Exception as exc:  # noqa: BLE001 - reported to the head, which re-plans
                self._free()
                if superseded():
                    return
                code = "oom" if isinstance(exc, ShardTooLarge | MemoryError) else "load_failed"
                # `rev` lets the head drop an error that belongs to an old plan.
                await self._send(
                    {"t": "error", "rev": msg.get("plan_rev"), "code": code, "message": repr(exc)}
                )
                return
            finally:
                reporter.cancel()
                if hasattr(source, "close"):
                    source.close()

            self.loaded_rev = rev
            self.engine = engine
            self._pump = asyncio.create_task(self._pump_frames(engine))
            await self._send(
                {
                    "t": "loaded",
                    "rev": rev,
                    "resident_bytes": resident,
                    "seconds": time.perf_counter() - started,
                    "from_cache": cache is not None and cache.fetched_bytes == 0,
                }
            )

    def _free(self) -> None:
        """Drop the resident shard and its outbound ring link. Safe when nothing is loaded.

        Inbound links are left alone: they belong to the peer's plan, and the
        peer closes its own end when it frees. Closing them here would cut a
        link that a faster neighbour opened for the plan now being loaded.
        """
        if self._pump is not None:
            self._pump.cancel()
            self._pump = None
        self.engine = None
        self._inbox = asyncio.Queue()
        if self._next is not None:
            self._next.close()
            self._next = None
        self.stack = None
        self.resident_bytes = 0
        self.loaded_rev = None
        gc.collect()
        if self.backend in ("cuda", "mps"):
            import torch

            getattr(torch, self.backend).empty_cache()

    async def handle_unload(self) -> None:
        """Free the shard. Stay connected and idle."""
        async with self._device_lock:
            self._free()

    async def on_control_lost(self) -> None:
        """Reconnect without unloading (PRD 6.1 reconnect rule, 11.3).

        Starts the 30 s shard hold. `run` redials the same head inside it, and
        `connect_control` keeps the shard only when the head still names the
        same plan revision. Past the hold the shard is freed and the worker
        returns to discovery.
        """
        self._epoch += 1  # a load in flight must not answer on the next session
        if self._writer is not None:
            await close_writer(self._writer)
            self._writer = None
        if self.loaded_rev is not None:
            self.reconnect_deadline = time.monotonic() + SHARD_HOLD_S
        log.warning("control connection lost; reconnecting")

    async def on_peer_lost(self, peer: str) -> None:
        """Report `link_down{peer}` when a ring data socket fails (PRD 11.1).

        The worker does not re-plan and does not drop its shard. The head owns
        that decision (PRD 11.2).
        """
        await self._send({"t": "link_down", "peer": peer})

    async def _pump_frames(self, engine: ForwardEngine) -> None:
        """Feed the engine one frame at a time and send what it returns (PRD 6.6).

        A failed frame ends its own request, not the worker: the error carries
        `req`, and the head aborts the ring for it.
        """
        from baton.worker.engine import OutOfMemoryOnKV

        while True:
            meta, payload = await self._inbox.get()
            try:
                started = time.perf_counter()
                outs = await asyncio.to_thread(engine.handle, meta, payload)
                if meta.get("n", 1) == 1 and meta["t"] in ("next", "act"):  # a decode step
                    self._compute_ms.append((time.perf_counter() - started) * 1000)
                for out in outs:
                    if out.to == "head":
                        await self._send(out.meta)
                    elif self._next is None:  # one node: the ring is this process
                        self._inbox.put_nowait((out.meta, out.payload))
                    else:
                        await send_frame(self._next, out.meta, out.payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported to the head, which owns the decision
                req = meta.get("req")
                if req:
                    engine.drop(req)
                if isinstance(exc, OSError):
                    await self.on_peer_lost(self._next_addr)
                code = "oom" if isinstance(exc, OutOfMemoryOnKV) else "forward_failed"
                frame = {"t": "error", "code": code, "message": repr(exc)}
                await self._send(frame | ({"req": req} if req else {}))

    async def stop(self) -> None:
        """Cancel work in flight, close sockets, free the shard."""
        self._epoch += 1  # stops the loader thread, which a cancel cannot reach
        for task in self._tasks:
            task.cancel()
        if self._data_server is not None:
            self._data_server.close()
        for link in self._inbound:
            link.close()
        if self._writer is not None:
            await close_writer(self._writer)
            self._writer = None
        self._free()


def run_worker(config: WorkerConfig) -> int:
    """Blocking entry point called by `baton worker` (PRD 15). Returns exit code."""

    async def main() -> None:
        task = asyncio.current_task()
        assert task is not None
        # SIGTERM must run `stop`, as Ctrl-C does. The head stops its local
        # worker with SIGTERM.
        with contextlib.suppress(NotImplementedError):  # no signal handlers on Windows
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        with contextlib.suppress(asyncio.CancelledError):
            await WorkerDaemon(config).run()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        return 130
    except Rejected as exc:
        log.error("the head refused this worker: %s", exc)
        return 1
    return 0
