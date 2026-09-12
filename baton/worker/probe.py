"""Capability probe and compute benchmark (PRD 6.2, 6.3).

The head needs two things from every worker before it can plan: what the device
is able to do, and how fast it actually is. This module answers both and owns
nothing else. It allocates no shard memory and holds no request state, so the
control loop can call it at any time without touching the compute thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Backend = Literal["cuda", "mps", "cpu"]
LinkKind = Literal["wifi", "wired", "unknown"]

#: OS memory held back per backend (PRD 6.3). MPS is unified memory, so the
#: reserve is the largest: the window server and every other app share it.
OS_RESERVE_BYTES: dict[Backend, int] = {
    "cuda": 1 * 1024**3,
    "mps": 2 * 1024**3,
    "cpu": 3 * 1024**3 // 2,
}


@dataclass(slots=True, frozen=True)
class MemoryBudget:
    """What this worker is willing to spend on weights plus KV cache."""

    mem_total_bytes: int
    mem_free_bytes: int
    usable_bytes: int


@dataclass(slots=True, frozen=True)
class BenchResult:
    """Per-layer timings for the planner cost function (PRD 9.2)."""

    t_dec_ms: float
    """Median milliseconds per layer per decoded token."""

    t_pre_ms: float
    """Median milliseconds per layer per 256-token prefill chunk."""

    quant: str
    dtype: str


@dataclass(slots=True, frozen=True)
class Capabilities:
    """The `caps` block of `hello`, refreshed on request (PRD 6.2)."""

    backend: Backend
    compute_dtype: str
    mem_total_bytes: int
    mem_free_bytes: int
    disk_free_bytes: int
    int4_fast_path: bool
    link: LinkKind
    bench: BenchResult | None = None


def pick_device(requested: str = "auto") -> Backend:
    """Resolve `--device` to a concrete backend (PRD 6.1 step 1).

    `auto` prefers `cuda`, then `mps`, then `cpu`. An explicit request is
    returned as given and is not checked for availability: a user who names a
    backend that is absent must see the failure from the first allocation, not
    a silent downgrade to CPU.
    """
    raise NotImplementedError


def memory_total_free(backend: Backend) -> tuple[int, int]:
    """Return `(mem_total_bytes, mem_free_bytes)` for this backend.

    The measurement differs per backend (PRD 6.2). MPS reports a recommended
    maximum rather than a true free figure, so the free value is clamped by the
    host free memory from `psutil`: unified memory is shared with the OS.
    """
    raise NotImplementedError


def int4_fast_path(backend: Backend) -> bool:
    """Test `torch._weight_int4pack_mm` on a 64x64 tensor.

    Any exception means no fast path. The probe must never raise: a missing
    kernel is a capability answer, not an error.
    """
    raise NotImplementedError


def detect_link() -> LinkKind:
    """Classify the default route interface as wireless or wired.

    Best effort per platform: `networksetup` on macOS, `iw` on Linux, `netsh`
    on Windows. Returns `unknown` when the tool is absent or the output does
    not parse. The planner treats `unknown` as `wifi` because that is the
    pessimistic case.
    """
    raise NotImplementedError


def probe_capabilities(
    backend: Backend,
    cache_dir: Path,
    bench: BenchResult | None = None,
) -> Capabilities:
    """Gather the full `caps` block.

    `compute_dtype` comes from the PRD 5.5 rule, which the model lane owns, and
    `disk_free_bytes` is `shutil.disk_usage(cache_dir).free`.

    Cost target: under 200 ms, because this runs on startup and again whenever
    the head asks for a refresh. `bench` is carried through unchanged: the
    benchmark is run separately, on demand, and is not part of the probe.
    """
    raise NotImplementedError


def memory_budget(backend: Backend, max_mem_bytes: int | None = None) -> MemoryBudget:
    """Compute the memory budget (PRD 6.3).

    `usable_bytes = min(mem_free_bytes, max_mem_bytes) - OS_RESERVE_BYTES[backend]`,
    floored at zero. `--max-mem` lets a 64 GB machine simulate a 4 GB one and
    lets a user keep memory for other work.
    """
    raise NotImplementedError


def run_bench(
    spec: object,
    quant: str,
    compute_dtype: str,
    backend: Backend,
) -> BenchResult:
    """Time one decoder layer of the target model's shape (PRD 6.3).

    Builds one layer with random weights in the requested tier, warms up 5
    decode steps against a 512-token synthetic cache, times 30 decode steps
    and 5 prefill passes of a 256-token chunk, then frees everything.

    Runs on the compute thread, never on the control loop: it holds the device
    for several seconds and would stall every heartbeat if it ran inline.
    Budget is under 10 s on any device.

    `spec` is the model spec owned by the kernels lane (PRD 5.1). It stays
    loosely typed here so this module does not import that lane.
    """
    raise NotImplementedError
