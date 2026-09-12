"""Worker registry on the head. PRD 7.2 and 7.6.

M1 interface only. Every body raises NotImplementedError.

An in-memory dict keyed by worker name. It is persisted nowhere: a restarted
head rebuilds it from the workers that reconnect.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

HEALTH_INTERVAL_S = 2.0
HEALTH_TIMEOUT_S = 6.0

WorkerState = Literal["standby", "loading", "loaded", "lost"]


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


@dataclass
class Health:
    """The last `health` frame of one worker (PRD 7.6)."""

    at: float
    free_bytes: int
    kv_used_bytes: int
    queue_depth: int


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

    @property
    def layers(self) -> int:
        """Layer count of the assigned range, or 0 when the worker is standby."""
        raise NotImplementedError


class NameInUse(Exception):
    """A second `hello` used a live name from a different data address (PRD 7.2)."""


@dataclass
class Registry:
    """Every worker the head knows about (PRD 7.2).

    `control` is the write side of a worker's control socket. The annotation is
    Any because the protocol lane owns the frame layer (PRD 8).
    """

    workers: dict[str, Worker] = field(default_factory=dict)

    def hello(
        self,
        name: str,
        data_address: tuple[str, int],
        capabilities: Capabilities,
        control: Any,
    ) -> Worker:
        """Add a worker, or replace the entry of a reconnecting one.

        A reconnect from the same data address replaces the old entry. A second
        `hello` with a name already `loading` or `loaded` and a different data
        address raises NameInUse, and the caller answers `error{"name in use"}`.
        """
        raise NotImplementedError

    def get(self, name: str) -> Worker | None:
        """The worker of that name, or None."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[Worker]:
        """Every worker, in join order."""
        raise NotImplementedError

    def on_health(self, name: str, health: Health) -> None:
        """Record a `health` frame. Workers send one every 2 s (PRD 7.6)."""
        raise NotImplementedError

    def timed_out(self, now: float) -> list[Worker]:
        """Workers with no `health` frame for `HEALTH_TIMEOUT_S` (PRD 7.6)."""
        raise NotImplementedError

    def mark_lost(self, name: str) -> Worker | None:
        """Move a worker to `lost` after a timeout or a control socket close.

        A `link_down{peer}` frame from a ring node is not enough on its own: the
        head waits one health timeout of `peer` first, so that a transient
        socket reset does not force a re-plan (PRD 7.6).
        """
        raise NotImplementedError

    def assign(self, name: str, first_layer: int, last_layer: int, holds_lm_head: bool) -> None:
        """Record the range that the plan gave this worker. Sets state `loading`."""
        raise NotImplementedError

    def mark_loaded(self, name: str) -> None:
        """Record a `loaded` reply. The head moves to READY when all are loaded."""
        raise NotImplementedError

    def ring(self) -> list[Worker]:
        """The loaded workers in ring order, N1 first."""
        raise NotImplementedError

    def roster(self) -> str:
        """The joined-worker table that `baton serve` prints (PRD 7.1 step 5)."""
        raise NotImplementedError
