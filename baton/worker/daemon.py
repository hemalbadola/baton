"""Worker daemon: startup, control loop, reconnect (PRD 6.1, 11).

The daemon is the only part of the worker that talks to the head. It runs an
asyncio loop on the main thread and never touches the model: every unit of real
work goes to `engine.ForwardEngine`, which owns its own compute thread. That
split is what keeps the 2 s heartbeat alive while a prefill chunk runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from baton.worker.probe import Backend, BenchResult, Capabilities, MemoryBudget

if TYPE_CHECKING:  # transport lane (PRD 8); imported for types only
    from baton.common.messages import Frame

    from baton.worker.engine import ForwardEngine

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

    def __enter__(self) -> SleepInhibitor:
        raise NotImplementedError

    def __exit__(self, *exc: object) -> None:
        raise NotImplementedError


async def discover_head(head: str, timeout_s: float = MDNS_BROWSE_S) -> tuple[str, int] | None:
    """Resolve the head address (PRD 6.1 step 4).

    An explicit `HOST[:PORT]` is parsed and returned without a browse. `auto`
    browses mDNS for `_baton._tcp.local.` for `timeout_s`. Returns None when
    nothing answers: the caller retries every 5 s forever and prints one line
    per minute, because a worker left running before the head starts is the
    normal case, not a failure.
    """
    raise NotImplementedError


class WorkerDaemon:
    """The control loop.

    Owns the control connection, the heartbeat, the data server, and the
    engine's lifetime. Every method here is a coroutine on the main thread and
    must stay sub-millisecond: anything longer belongs on the compute thread.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self.backend: Backend | None = None
        self.caps: Capabilities | None = None
        self.budget: MemoryBudget | None = None
        self.engine: ForwardEngine | None = None
        self.loaded_rev: int | None = None
        """Plan revision of the resident shard, echoed in `hello` and `health`
        so a returning worker can skip reloading (PRD 6.1, 11.3). None means
        nothing is resident, which is also how `handle_load` knows to skip the
        free step."""

        self.reconnect_deadline: float | None = None
        """`perf_counter()` at which the 30 s shard hold expires. None while the
        control connection is up. These two fields are the whole worker state:
        every branch in PRD 6.1 and 11.3 turns on one of them."""

    async def run(self) -> None:
        """Run the startup sequence, then serve until stopped.

        Steps in PRD 6.1 order: pick device, probe, start the data server,
        find the head, open the control connection, send `hello`, start the
        heartbeat, inhibit sleep, wait for `load`.
        """
        raise NotImplementedError

    async def start_data_server(self) -> tuple[str, int]:
        """Bind the data server on all LAN interfaces. Returns `(ip, port)`.

        Started before the head is found, because `hello` must carry the real
        `data_addr` and an ephemeral port is only known after the bind.
        """
        raise NotImplementedError

    async def connect_control(self, addr: tuple[str, int]) -> None:
        """Open the control connection and send `hello` (PRD 6.1 step 5, 8.5).

        Retries every 5 s forever on refusal. Raises only on a token rejection,
        which is a configuration mistake and will not fix itself.
        """
        raise NotImplementedError

    async def heartbeat(self) -> None:
        """Send `health` every 2 s until the connection drops (PRD 8.3).

        Fields come from the engine and the pool, never from a device query:
        `mem_free`, `kv_used`, `queue_depth`, `active_reqs`, `loaded_rev`. Six
        seconds of silence makes the head mark this worker lost (PRD 8.6), so
        this coroutine must never await anything that can block on the device.
        """
        raise NotImplementedError

    async def handle_control(self, frame: Frame) -> None:
        """Dispatch one head-to-worker message (PRD 8.3).

        `welcome`, `bench`, `load`, `unload`, `ping_peer`, `abort`. Each
        handler either answers from daemon state or submits a job to the
        engine. None of them run model code inline.
        """
        raise NotImplementedError

    async def handle_bench(self, spec: object, quant: str, compute_dtype: str) -> BenchResult:
        """Run the benchmark on the compute thread, reply `bench_result`.

        The benchmark holds the device for up to 10 s. Running it here would
        stop the heartbeat and get this worker marked lost at 6 s, so it is
        submitted to the engine and awaited through the outbox.
        """
        raise NotImplementedError

    async def handle_load(self, msg: dict[str, object]) -> None:
        """Load a shard (PRD 6.4).

        Frees any differing resident plan first, then delegates the tensor work
        to the shards lane (PRD 5.3) and reports `load_progress` every 500 ms.
        Preallocates RoPE tables for `ctx_max`, opens the data connection to
        `next_node` within 30 s, then replies `loaded`.
        """
        raise NotImplementedError

    async def handle_unload(self) -> None:
        """Free the shard and every KV entry. Stay connected and idle."""
        raise NotImplementedError

    async def on_control_lost(self) -> None:
        """Reconnect without unloading (PRD 6.1 reconnect rule, 11.3).

        Retry every 2 s with the same name for 30 s. If the head returns inside
        that window with the same plan revision, resume with no reload.
        Otherwise unload and return to discovery.
        """
        raise NotImplementedError

    async def on_peer_lost(self, peer: str) -> None:
        """Report `link_down{peer}` when a ring data socket closes (PRD 11.1).

        The worker does not re-plan and does not drop its shard. The head owns
        that decision (PRD 11.2).
        """
        raise NotImplementedError

    async def stop(self) -> None:
        """Stop the engine, close sockets, release the sleep inhibitor."""
        raise NotImplementedError


def run_worker(config: WorkerConfig) -> int:
    """Blocking entry point called by `baton worker` (PRD 15). Returns exit code."""
    raise NotImplementedError
