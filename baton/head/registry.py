"""Worker registry on the head. PRD 7.2 and 7.6.

An in-memory dict keyed by worker name. It is persisted nowhere: a restarted
head rebuilds it from the workers that reconnect.

One worker moves through four states:

    hello           -> standby   (known, benched or not, holds no layers)
    assign(plan)    -> loading   (the plan gave it a layer range)
    loaded frame    -> loaded    (the shard is resident, ring links are open)
    6 s of silence  -> lost      (or the control socket closed)

`standby` is also where a reconnecting worker lands, because a returning socket
proves the process is alive and proves nothing about the shard.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

HEALTH_INTERVAL_S = 2.0
HEALTH_TIMEOUT_S = 6.0

# PRD 6.3: usable_bytes = min(mem_free_bytes, --max-mem) - os_reserve. The head
# applies this only when a worker's caps block does not carry the result.
OS_RESERVE_BYTES: dict[str, int] = {
    "cuda": 1 * 1024**3,
    "mps": 2 * 1024**3,
    "cpu": 3 * 1024**3 // 2,
}

WorkerState = Literal["standby", "loading", "loaded", "lost"]

# The two states that make a name un-reusable from a new address (PRD 7.2).
LIVE_STATES: frozenset[str] = frozenset({"loading", "loaded"})


@dataclass
class Capabilities:
    """The `hello` payload of one worker (PRD 6.2, 9.1).

    `usable_bytes` and the two timings feed the planner directly.
    """

    backend: str
    usable_bytes: int
    t_dec_ms: float
    t_pre_ms: float
    int4_fast_path: bool = False
    link: str = "unknown"
    compute_dtype: str = "bf16"

    @classmethod
    def from_caps(cls, caps: dict[str, Any]) -> Capabilities:
        """Parse the `caps` block of one `hello` frame (PRD 6.2).

        This is a trust boundary: `caps` arrives over the wire, so every field
        is optional here and every value is coerced. A worker that sends
        `usable_bytes` wins, because only the worker knows its own `--max-mem`.

        The timings stay 0.0 until `bench` returns. The planner rejects a fleet
        with a zero cost, so `on_bench` must run before `Registry.devices`.
        """
        backend = str(caps.get("backend", "cpu"))
        usable = caps.get("usable_bytes")
        if usable is None:
            free = _as_int(caps.get("mem_free_bytes"))
            usable = free - OS_RESERVE_BYTES.get(backend, OS_RESERVE_BYTES["cpu"])
        bench = caps.get("bench") or {}
        if not isinstance(bench, dict):
            bench = {}
        return cls(
            backend=backend,
            usable_bytes=max(0, _as_int(usable)),
            t_dec_ms=_as_float(bench.get("t_dec_ms")),
            t_pre_ms=_as_float(bench.get("t_pre_ms")),
            int4_fast_path=bool(caps.get("int4_fast_path", False)),
            link=str(caps.get("link", "unknown")),
            compute_dtype=str(caps.get("compute_dtype", "bf16")),
        )


@dataclass
class Health:
    """The last `health` frame of one worker (PRD 7.6)."""

    at: float
    free_bytes: int
    kv_used_bytes: int
    queue_depth: int
    loaded_rev: int | None = None
    compute_ms: list[float] = field(default_factory=list)

    @classmethod
    def from_frame(cls, meta: dict[str, Any], at: float) -> Health:
        """Read one `health` frame. `at` is stamped by the head, not the wire.

        The head never trusts a peer clock for liveness (PRD 17.2), so the
        arrival time decides a timeout, not a timestamp inside the frame.
        """
        rev = meta.get("loaded_rev")
        times = meta.get("compute_ms")
        return cls(
            at=at,
            free_bytes=_as_int(meta.get("mem_free")),
            kv_used_bytes=_as_int(meta.get("kv_used")),
            queue_depth=_as_int(meta.get("queue_depth")),
            loaded_rev=None if rev is None else _as_int(rev),
            compute_ms=[_as_float(v) for v in times][-12:] if isinstance(times, list) else [],
        )


@dataclass
class Worker:
    """One entry of the registry (PRD 7.2)."""

    name: str
    data_address: tuple[str, int]
    capabilities: Capabilities
    control: Any
    state: WorkerState = "standby"
    health: Health | None = None
    first_layer: int | None = None
    last_layer: int | None = None
    holds_lm_head: bool = False
    last_seen: float = 0.0

    @property
    def layers(self) -> int:
        """Layer count of the assigned range, or 0 when the worker is standby."""
        if self.first_layer is None or self.last_layer is None:
            return 0
        return self.last_layer - self.first_layer + 1

    @property
    def data_addr(self) -> str:
        """The `ip:port` string that a ring peer dials (PRD 8.5)."""
        return f"{self.data_address[0]}:{self.data_address[1]}"


class NameInUse(Exception):
    """A second `hello` used a live name from a different data address (PRD 7.2)."""


@dataclass
class Registry:
    """Every worker the head knows about (PRD 7.2).

    `control` is the write side of a worker's control socket. The annotation is
    Any because the protocol lane owns the frame layer (PRD 8).

    `clock` exists so a test can drive the 6 s health timeout without sleeping.
    """

    workers: dict[str, Worker] = field(default_factory=dict)
    clock: Callable[[], float] = time.monotonic

    def hello(
        self,
        name: str,
        data_address: tuple[str, int],
        capabilities: Capabilities,
        control: Any,
    ) -> Worker:
        """Add a worker, or replace the entry of a reconnecting one.

        A reconnect from the same data address replaces the old entry and keeps
        its layer range, so PRD 11.3 can skip planning when the worker reports
        the same `loaded_rev`. The state still drops to `standby`: a live socket
        proves the process is up, never that the shard is still resident. The
        caller confirms the revision and calls `mark_loaded`.

        A second `hello` with a name already `loading` or `loaded` and a
        different data address raises NameInUse, and the caller answers
        `error{"name in use"}`.
        """
        if not name:
            raise ValueError("hello carries an empty name")
        old = self.workers.get(name)
        if old is not None and old.state in LIVE_STATES and old.data_address != data_address:
            raise NameInUse(f"name in use: {name} is live at {old.data_addr}")
        worker = Worker(
            name=name,
            data_address=data_address,
            capabilities=capabilities,
            control=control,
            last_seen=self.clock(),
        )
        if old is not None and old.data_address == data_address:
            worker.first_layer = old.first_layer
            worker.last_layer = old.last_layer
            worker.holds_lm_head = old.holds_lm_head
        # A plain assignment keeps the original insertion slot, so a reconnect
        # does not jump to the end of the join order.
        self.workers[name] = worker
        return worker

    def get(self, name: str) -> Worker | None:
        """The worker of that name, or None."""
        return self.workers.get(name)

    def __iter__(self) -> Iterator[Worker]:
        """Every worker, in join order."""
        return iter(list(self.workers.values()))

    def __len__(self) -> int:
        return len(self.workers)

    def on_health(self, name: str, health: Health) -> None:
        """Record a `health` frame. Workers send one every 2 s (PRD 7.6).

        A frame from an unknown worker is dropped. A frame from a `lost` worker
        is recorded and does NOT revive it: only a fresh `hello` does that, and
        the head must re-check the plan revision first (PRD 11.2 step 4).
        """
        worker = self.workers.get(name)
        if worker is None:
            return
        worker.health = health
        worker.last_seen = health.at

    def on_bench(self, name: str, t_dec_ms: float, t_pre_ms: float) -> None:
        """Record a `bench_result` reply. These two numbers drive the planner.

        A timing that is not a finite positive number is dropped: NaN passes
        every comparison the planner makes and then breaks its arithmetic.
        """
        worker = self.workers.get(name)
        dec, pre = _as_float(t_dec_ms), _as_float(t_pre_ms)
        if worker is None or not (0 < dec < math.inf and 0 < pre < math.inf):
            return
        worker.capabilities.t_dec_ms = dec
        worker.capabilities.t_pre_ms = pre

    def timed_out(self, now: float) -> list[Worker]:
        """Workers with no `health` frame for `HEALTH_TIMEOUT_S` (PRD 7.6).

        The clock starts at `hello`, not at the first `health`, so a worker that
        joins and never heartbeats still times out.
        """
        return [
            w
            for w in self.workers.values()
            if w.state != "lost" and now - w.last_seen > HEALTH_TIMEOUT_S
        ]

    def mark_lost(self, name: str) -> Worker | None:
        """Move a worker to `lost` after a timeout or a control socket close.

        A `link_down{peer}` frame from a ring node is not enough on its own: the
        head waits one health timeout of `peer` first, so that a transient
        socket reset does not force a re-plan (PRD 7.6). `serve` holds that
        timer; this method is the end of it.
        """
        worker = self.workers.get(name)
        if worker is None:
            return None
        worker.state = "lost"
        return worker

    def assign(self, name: str, first_layer: int, last_layer: int, holds_lm_head: bool) -> None:
        """Record the range that the plan gave this worker. Sets state `loading`."""
        worker = self.workers.get(name)
        if worker is None:
            raise KeyError(name)
        if first_layer < 0 or last_layer < first_layer:
            raise ValueError(f"bad range for {name}: [{first_layer}, {last_layer}]")
        worker.first_layer = first_layer
        worker.last_layer = last_layer
        worker.holds_lm_head = holds_lm_head
        worker.state = "loading"

    def mark_loaded(self, name: str) -> None:
        """Record a `loaded` reply. The head moves to READY when all are loaded."""
        worker = self.workers.get(name)
        if worker is None:
            raise KeyError(name)
        if worker.first_layer is None:
            raise ValueError(f"{name} reported loaded with no assigned range")
        worker.state = "loaded"

    def release(self, name: str) -> None:
        """Drop a worker's layer range and return it to `standby` (PRD 11.2).

        Used on `unload` and before a re-plan, so a stale range never reaches
        the ring.
        """
        worker = self.workers.get(name)
        if worker is None:
            return
        worker.first_layer = None
        worker.last_layer = None
        worker.holds_lm_head = False
        if worker.state != "lost":
            worker.state = "standby"

    def ring(self) -> list[Worker]:
        """The loaded workers in ring order, N1 first.

        Ring order is layer order: N1 holds layer 0 and Nk holds the lm_head
        (PRD 4.3). A worker with no range cannot be in the ring.
        """
        loaded = [w for w in self.workers.values() if w.state == "loaded" and w.layers > 0]
        loaded.sort(key=lambda w: w.first_layer or 0)
        return loaded

    def roster(self) -> list[Worker]:
        """Every worker in join order, the structured form.

        This backs `api.HeadSeam.snapshot` and `.subscribe`, which build one
        `NodeSnapshot` per entry. It returns objects rather than the printed
        table of PRD 7.1 step 5, because a string cannot back a
        `ClusterSnapshot`. `roster_table` renders the CLI view from the same
        list.
        """
        return list(self.workers.values())

    def devices(self) -> list[tuple[str, int, float, float, bool]]:
        """Planner input rows for every worker that can hold layers (PRD 9.1).

        One tuple per worker: `(name, usable_bytes, t_dec_ms, t_pre_ms,
        int4_fast_path)`. `serve` turns these into `planner.Device`; the
        registry does not import the planner.

        A `lost` worker is left out. A worker that has not benched is left out
        too, because a zero cost makes the planner reject the whole fleet.
        """
        rows = []
        for w in self.workers.values():
            caps = w.capabilities
            if w.state == "lost" or caps.t_dec_ms <= 0.0 or caps.t_pre_ms <= 0.0:
                continue
            rows.append(
                (w.name, caps.usable_bytes, caps.t_dec_ms, caps.t_pre_ms, caps.int4_fast_path)
            )
        return rows

    def roster_table(self) -> str:
        """The joined-worker table that `baton serve` prints (PRD 7.1 step 5)."""
        wide = max([16, *(len(name) for name in self.workers)])
        header = (
            f"{'NAME':<{wide}} {'STATE':<8} {'BACKEND':<8} {'USABLE':>9} "
            f"{'t_dec':>8} {'t_pre':>8} {'LAYERS':>10}"
        )
        lines = [header, "-" * len(header)]
        for w in self.workers.values():
            caps = w.capabilities
            span = "-" if w.layers == 0 else f"{w.first_layer}-{w.last_layer}"
            if w.holds_lm_head and w.layers:
                span += "*"
            lines.append(
                f"{w.name:<{wide}} {w.state:<8} {caps.backend:<8} "
                f"{caps.usable_bytes / 1024**3:>8.1f}G "
                f"{caps.t_dec_ms:>8.3f} {caps.t_pre_ms:>8.3f} {span:>10}"
            )
        return "\n".join(lines)


def _as_int(value: Any) -> int:
    """Coerce one wire value to int. A bad value is 0, never an exception."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):  # OverflowError: JSON `Infinity`
        return 0


def _as_float(value: Any) -> float:
    """Coerce one wire value to float. A bad value is 0.0, never an exception."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
