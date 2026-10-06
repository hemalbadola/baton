"""Capability probe and compute benchmark (PRD 6.2, 6.3).

The head needs two things from every worker before it can plan: what the device
is able to do, and how fast it actually is. This module answers both and owns
nothing else. It allocates no shard memory and holds no request state, so the
control loop can call it at any time without touching the compute thread.
"""

from __future__ import annotations

import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Backend = Literal["cuda", "mps", "cpu"]
LinkKind = Literal["wifi", "wired", "unknown"]

#: OS memory held back per backend (PRD 6.3). MPS is unified memory, so the
#: reserve is the largest: the window server and every other app share it.
OS_RESERVE_BYTES: dict[Backend, int] = {
    "cuda": 1 * 1024**3,
    "mps": 2 * 1024**3,
    "cpu": 3 * 1024**3 // 2,
}

#: Compute dtype names as they travel in `caps` and `bench`, and the torch
#: attribute each one names.
_TORCH_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

# PRD 6.3 benchmark shape.
BENCH_CACHE_TOKENS = 512
BENCH_WARMUP_STEPS = 5
BENCH_DECODE_STEPS = 30
BENCH_PREFILL_PASSES = 5

_now = time.perf_counter  # a module name, so a test can drive the clock


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


def torch_dtype(name: str) -> Any:
    """The torch dtype for a wire name such as `bf16`."""
    import torch

    return getattr(torch, _TORCH_DTYPES[name])


def dtype_name(dtype: object) -> str:
    """Inverse of `torch_dtype`."""
    return next(k for k, v in _TORCH_DTYPES.items() if str(dtype) == f"torch.{v}")


def _safe(probe: Callable[[], Any], default: Any) -> Any:
    """Run one sub-probe. A failure is an answer, never a lost `hello`."""
    try:
        return probe()
    except Exception:  # noqa: BLE001 - any failure means "unknown", by design
        return default


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=True).stdout


def pick_device(requested: str = "auto") -> Backend:
    """Resolve `--device` to a concrete backend (PRD 6.1 step 1).

    `auto` prefers `cuda`, then `mps`, then `cpu`. An explicit request is
    returned as given and is not checked for availability: a user who names a
    backend that is absent must see the failure from the first allocation, not
    a silent downgrade to CPU.
    """
    if requested != "auto":
        return requested  # type: ignore[return-value]
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def memory_total_free(backend: Backend) -> tuple[int, int]:
    """Return `(mem_total_bytes, mem_free_bytes)` for this backend.

    The measurement differs per backend (PRD 6.2). MPS reports a recommended
    maximum rather than a true free figure, so the free value is clamped by the
    host free memory from `psutil`: unified memory is shared with the OS.
    """
    import psutil
    import torch

    if backend == "cuda":
        free, total = torch.cuda.mem_get_info()
        return total, free
    host = psutil.virtual_memory()
    if backend == "mps":
        total = torch.mps.recommended_max_memory()
        free = total - torch.mps.current_allocated_memory()
        return total, max(0, min(free, host.available))
    return host.total, host.available


def int4_fast_path(backend: Backend) -> bool:
    """True when the packed int4 kernel exists here AND matches the reference.

    The check itself is `quant.int4_fast_ok`: availability is not correctness.
    Any exception means no fast path. The probe must never raise: a missing
    kernel is a capability answer, not an error.
    """

    def check() -> bool:
        from baton.model import quant

        return bool(quant.int4_fast_ok(backend, quant.compute_dtype(backend)))

    return _safe(check, False)


def detect_link() -> LinkKind:
    """Classify the default route interface as wireless or wired.

    Best effort per platform: `route` plus `networksetup` on macOS, `ip` plus
    sysfs on Linux, `netsh` on Windows. Returns `unknown` when the tool is
    absent or the output does not parse: a wrong answer skews every hop the
    planner predicts, so no answer is better. The planner treats `unknown` as
    `wifi` because that is the pessimistic case.
    """

    def classify() -> LinkKind:
        if sys.platform == "darwin":
            iface = re.search(r"interface:\s*(\S+)", _run(["route", "-n", "get", "default"]))
            ports = _run(["networksetup", "-listallhardwareports"])
            port = re.search(rf"Hardware Port: ([^\n]+)\nDevice: {re.escape(iface[1])}\n", ports)
            return "wifi" if re.search(r"wi-?fi|airport", port[1], re.IGNORECASE) else "wired"
        if sys.platform.startswith("linux"):
            iface = re.search(r"\bdev\s+(\S+)", _run(["ip", "route", "show", "default"]))
            return "wifi" if Path(f"/sys/class/net/{iface[1]}/wireless").exists() else "wired"
        if sys.platform == "win32":
            # ponytail: matches English `netsh` output only, and a machine with
            # Wi-Fi and Ethernet both up reads as wifi. Parse `Get-NetRoute` if
            # that bites.
            out = _run(["netsh", "wlan", "show", "interfaces"])
            return (
                "wifi"
                if re.search(r"^\s*State\s*:\s*connected", out, re.IGNORECASE | re.MULTILINE)
                else "wired"
            )
        return "unknown"

    # A failed match subscripts None and lands here as `unknown`.
    return _safe(classify, "unknown")


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
    from baton.model import quant

    def disk_free() -> int:
        path = cache_dir
        while not path.exists() and path != path.parent:
            path = path.parent
        return shutil.disk_usage(path).free

    total, free = _safe(lambda: memory_total_free(backend), (0, 0))
    return Capabilities(
        backend=backend,
        compute_dtype=_safe(lambda: dtype_name(quant.compute_dtype(backend)), "fp32"),
        mem_total_bytes=total,
        mem_free_bytes=free,
        disk_free_bytes=_safe(disk_free, 0),
        int4_fast_path=int4_fast_path(backend),
        link=detect_link(),
        bench=bench,
    )


def memory_budget(backend: Backend, max_mem_bytes: int | None = None) -> MemoryBudget:
    """Compute the memory budget (PRD 6.3, as corrected by BAT-8 and BAT-23).

    The ceiling is `mem_total_bytes - OS_RESERVE_BYTES[backend]`. Without
    `--max-mem` the worker offers its free memory up to that ceiling. With it,
    the worker offers what the user said, up to the same ceiling: the flag lets
    a 64 GB machine simulate a 4 GB one, and lets the owner of an 8 GB laptop
    promise memory that a browser holds right now.

    PRD 6.3 takes the reserve off the free figure. Free memory already leaves
    out what the OS and every open app hold, so that counts the reserve twice:
    a real 8 GB Mac with 1.3 GB free came out at zero bytes.
    """
    total, free = memory_total_free(backend)
    ceiling = total - OS_RESERVE_BYTES[backend]
    usable = min(free if max_mem_bytes is None else max_mem_bytes, ceiling)
    return MemoryBudget(mem_total_bytes=total, mem_free_bytes=free, usable_bytes=max(0, usable))


class _BenchKV:
    """A one-layer KV cache preloaded with random rows (PRD 6.3 step 2)."""

    def __init__(self, spec: Any, max_len: int, filled: int, device: Any, dtype: Any) -> None:
        import torch

        shape = (max_len, spec.n_kv_heads, spec.head_dim)
        self.k = torch.randn(shape).to(device=device, dtype=dtype)
        self.v = torch.randn(shape).to(device=device, dtype=dtype)
        self.length = filled

    def append(self, k: Any, v: Any, pos_start: int) -> tuple[Any, Any]:
        self.length = pos_start + k.shape[0]
        self.k[pos_start : self.length] = k
        self.v[pos_start : self.length] = v
        return self.k[: self.length], self.v[: self.length]


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

    `spec` is a `ModelSpec`, or its `to_dict()` form as it arrives in `bench`.
    """
    import gc

    import torch

    from baton.model.layers import DecoderLayer, build_rope_tables
    from baton.model.spec import ModelSpec
    from baton.worker.engine import CHUNK_TOKENS

    if isinstance(spec, dict):
        spec = ModelSpec(**spec)
    device = torch.device(backend)
    dtype = torch_dtype(compute_dtype)
    max_len = BENCH_CACHE_TOKENS + BENCH_WARMUP_STEPS + BENCH_DECODE_STEPS

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def measure() -> tuple[list[float], list[float]]:
        # Every tensor is a local of this function, so all of it is garbage the
        # moment it returns (PRD 6.3 step 5).
        # ponytail: a dense layer for every tier. The decoder has no quantized
        # matmul yet (BAT-10), so `quant` is echoed back, not exercised.
        layer = DecoderLayer(spec, device=device, dtype=dtype).eval()
        cos, sin = build_rope_tables(spec, max_len, device=device)
        cache = _BenchKV(spec, max_len, BENCH_CACHE_TOKENS, device, dtype)

        def timed_ms(x: torch.Tensor, pos: int) -> float:
            rows = slice(pos, pos + x.shape[0])
            sync()
            start = _now()
            layer(x, cos[rows], sin[rows], pos, cache)
            sync()
            return (_now() - start) * 1000.0

        token = torch.randn(1, spec.hidden).to(device=device, dtype=dtype)
        chunk = torch.randn(CHUNK_TOKENS, spec.hidden).to(device=device, dtype=dtype)
        steps = range(BENCH_WARMUP_STEPS + BENCH_DECODE_STEPS)
        decode = [timed_ms(token, BENCH_CACHE_TOKENS + i) for i in steps][BENCH_WARMUP_STEPS:]
        return decode, [timed_ms(chunk, 0) for _ in range(BENCH_PREFILL_PASSES)]

    with torch.inference_mode():
        decode, prefill = measure()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return BenchResult(
        t_dec_ms=statistics.median(decode),
        t_pre_ms=statistics.median(prefill),
        quant=quant,
        dtype=compute_dtype,
    )
