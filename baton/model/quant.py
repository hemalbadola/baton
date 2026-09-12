"""Weight-only quantization tiers T0, T1, T2 (PRD 5.5).

Every tier is round-to-nearest, done by the worker at first load, tensor by
tensor, then cached. No offline tool and no external quantization library
(decision D8).

=====  ==========  ==================================================  =========================
Tier   Flag        Storage                                             Matmul path
=====  ==========  ==================================================  =========================
T0     ``bf16``    as shipped, fp16 where bf16 is absent               ``x @ W.T``
T1     ``int8``    per-output-channel absmax, fp16 scale               dequant, then matmul
T2     ``int4``    group 128 along the input dim, fp16 scale and zero, ``_weight_int4pack_mm``
                   packed two nibbles per byte                         where it verifies, else
                                                                       dequant
=====  ==========  ==================================================  =========================

The int4 kernel is never trusted on sight. :func:`int4_fast_ok` runs it against
the dequant reference once per device and dtype, and falls back for good if the
two disagree. Quantization that quietly degrades quality is the failure this
guards against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

Tier = Literal["bf16", "int8", "int4"]

TIERS: tuple[Tier, ...] = ("bf16", "int8", "int4")

GROUP = 128
"""int4 group size along the input dimension (PRD 5.5)."""

INT8_MAX = 127
INT4_MAX = 15
EPS = 1e-8
"""Floor for a group scale. A constant group would otherwise divide by zero."""

FAST_RTOL = 2e-2
"""How far the int4 kernel may sit from the dequant reference before we drop it."""

INT4_KERNEL_MAX_ROWS = 32
"""Above this many rows the dequant path beats the packed-int4 kernel.

Measured on Apple-silicon CPU with a 4864 x 896 weight: the kernel is 15x faster
at one row (decode) and 2.5x slower at 256 rows (a prefill chunk, PRD 10.1). The
crossover sits near 32. Backends differ, so this is a constant, not a law.
"""


class QuantError(Exception):
    """A tensor cannot be quantized in the requested tier."""


# --------------------------------------------------------------------------- #
# Compute dtype (PRD 5.5)
# --------------------------------------------------------------------------- #


def compute_dtype(device: str | torch.device = "cpu") -> torch.dtype:
    """Pick the arithmetic dtype for a backend, by the rules in PRD 5.5."""
    kind = torch.device(device).type
    if kind == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if kind == "mps":
        return torch.bfloat16 if _mps_has_bf16() else torch.float16
    return torch.bfloat16 if _cpu_has_bf16() else torch.float32


def _mps_has_bf16() -> bool:
    import platform

    try:
        major = int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        return False
    torch_major, torch_minor = (int(p) for p in torch.__version__.split(".")[:2])
    return major >= 14 and (torch_major, torch_minor) >= (2, 3)


def _cpu_has_bf16() -> bool:
    if torch.backends.cpu.get_cpu_capability() in ("AVX512_BF16", "AVX512"):
        return True
    import platform

    return platform.machine() in ("arm64", "aarch64")


# --------------------------------------------------------------------------- #
# The quantized weight
# --------------------------------------------------------------------------- #


@dataclass
class QuantWeight:
    """One weight matrix ``[out, in]`` in one tier, with everything needed to use it.

    ``qweight`` holds bf16/fp16 values for T0, int8 for T1, and packed nibbles of
    shape ``[out, in // 2]`` for T2. ``scale`` is ``[out]`` for T1 and
    ``[out, in // group]`` for T2. ``zero`` exists for T2 alone.
    """

    tier: Tier
    shape: tuple[int, int]
    qweight: torch.Tensor
    scale: torch.Tensor | None = None
    zero: torch.Tensor | None = None
    group: int = GROUP
    _packed: dict = field(default_factory=dict, repr=False, compare=False)
    """Kernel-ready int4 weights, built once per (device, dtype). Never persisted."""

    @property
    def nbytes(self) -> int:
        """Bytes on disk and in memory, scales and zeros included."""
        total = self.qweight.numel() * self.qweight.element_size()
        for t in (self.scale, self.zero):
            if t is not None:
                total += t.numel() * t.element_size()
        return total

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Reconstruct the full ``[out, in]`` matrix. The reference path for every tier."""
        if self.tier == "bf16":
            out = self.qweight
        elif self.tier == "int8":
            out = dequantize_int8(self.qweight, _require(self.scale, "scale"))
        elif self.tier == "int4":
            out = dequantize_int4(
                self.qweight,
                _require(self.scale, "scale"),
                _require(self.zero, "zero"),
                self.shape[1],
                self.group,
            )
        else:
            raise QuantError(f"unknown tier {self.tier!r}")
        return out if dtype is None else out.to(dtype)

    def to(self, device: str | torch.device) -> QuantWeight:
        """Move every stored tensor to a device. Returns a new view, same storage rules."""

        def move(t):
            return None if t is None else t.to(device)

        return QuantWeight(
            self.tier,
            self.shape,
            self.qweight.to(device),
            move(self.scale),
            move(self.zero),
            self.group,
        )  # the packed cache is rebuilt on the new device, not copied

    def to_tensors(self) -> dict[str, torch.Tensor]:
        """Flatten to named tensors, ready for :func:`safetensors_io.save_safetensors`."""
        out = {"qweight": self.qweight}
        if self.scale is not None:
            out["scale"] = self.scale
        if self.zero is not None:
            out["zero"] = self.zero
        return out

    @classmethod
    def from_tensors(
        cls,
        tier: Tier,
        shape: tuple[int, int],
        tensors: dict[str, torch.Tensor],
        group: int = GROUP,
    ) -> QuantWeight:
        return cls(
            tier, tuple(shape), tensors["qweight"], tensors.get("scale"), tensors.get("zero"), group
        )


def _require(t: torch.Tensor | None, what: str) -> torch.Tensor:
    if t is None:
        raise QuantError(f"missing {what}")
    return t


# --------------------------------------------------------------------------- #
# T1 — int8, per output channel
# --------------------------------------------------------------------------- #


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel absmax int8. Returns ``(q int8 [out, in], scale fp16 [out])``."""
    f = w.float()
    scale = (f.abs().amax(dim=1) / INT8_MAX).clamp(min=EPS)
    q = (f / scale[:, None]).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float()[:, None]


# --------------------------------------------------------------------------- #
# T2 — int4, group 128 along the input dimension
# --------------------------------------------------------------------------- #


def pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """Pack a uint8 tensor of 0..15 values, two per byte, along the last dimension.

    Element ``2i`` goes to the low nibble, element ``2i+1`` to the high nibble.
    """
    if q.shape[-1] % 2:
        raise QuantError(f"cannot pack an odd last dimension {q.shape[-1]}")
    q = q.to(torch.uint8)
    return (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_nibbles`."""
    lo = packed & 0x0F
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def quantize_int4(
    w: torch.Tensor, group: int = GROUP
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Round-to-nearest int4, group ``group`` along the input dim (PRD 5.5).

    Returns ``(packed uint8 [out, in // 2], scale fp16 [out, in // group],
    zero uint8 [out, in // group])``.
    """
    if w.ndim != 2:
        raise QuantError(f"int4 needs a 2-D weight, got shape {tuple(w.shape)}")
    out_f, in_f = w.shape
    if in_f % group:
        raise QuantError(f"input dim {in_f} is not a multiple of the group size {group}")

    f = w.float().reshape(out_f, in_f // group, group)
    lo = f.amin(dim=-1, keepdim=True)
    hi = f.amax(dim=-1, keepdim=True)
    # A constant group has zero range. Spanning [0, v] instead of [v, v] keeps the
    # value representable; a floor alone would collapse it to zero.
    span = torch.where(hi > lo, hi - lo, hi.abs().clamp(min=EPS))
    lo = torch.where(hi > lo, lo, torch.minimum(hi, torch.zeros_like(hi)))
    scale = span / INT4_MAX
    zero = (-lo / scale).round().clamp(0, INT4_MAX)
    q = (f / scale + zero).round().clamp(0, INT4_MAX).to(torch.uint8)

    return (
        pack_nibbles(q.reshape(out_f, in_f)),
        scale.squeeze(-1).to(torch.float16),
        zero.squeeze(-1).to(torch.uint8),
    )


def dequantize_int4(
    packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, in_f: int, group: int = GROUP
) -> torch.Tensor:
    out_f = packed.shape[0]
    q = unpack_nibbles(packed)[:, :in_f].float().reshape(out_f, in_f // group, group)
    w = (q - zero.float()[..., None]) * scale.float()[..., None]
    return w.reshape(out_f, in_f)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def quantize(
    w: torch.Tensor, tier: Tier, group: int = GROUP, store_dtype: torch.dtype | None = None
) -> QuantWeight:
    """Quantize one ``[out, in]`` weight into ``tier``."""
    if tier not in TIERS:
        raise QuantError(f"unknown tier {tier!r}, expected one of {TIERS}")
    if w.ndim != 2:
        raise QuantError(f"expected a 2-D weight, got shape {tuple(w.shape)}")
    shape = (w.shape[0], w.shape[1])

    if tier == "bf16":
        keep = store_dtype or (
            w.dtype if w.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        )
        return QuantWeight("bf16", shape, w.to(keep), group=group)
    if tier == "int8":
        q, scale = quantize_int8(w)
        return QuantWeight("int8", shape, q, scale, group=group)
    packed, scale, zero = quantize_int4(w, group)
    return QuantWeight("int4", shape, packed, scale, zero, group)


# --------------------------------------------------------------------------- #
# Matmul dispatch
# --------------------------------------------------------------------------- #

_FAST_CHECKED: dict[tuple[str, torch.dtype], bool] = {}


def _int4_ops(device_type: str):
    """The pack and mm ops for a backend, or ``None`` when the backend has none."""
    aten = torch.ops.aten
    if device_type == "cpu":
        pack, mm = "_convert_weight_to_int4pack_for_cpu", "_weight_int4pack_mm_for_cpu"
    else:
        pack, mm = "_convert_weight_to_int4pack", "_weight_int4pack_mm"
    if not (hasattr(aten, pack) and hasattr(aten, mm)):
        return None
    return getattr(aten, pack), getattr(aten, mm)


def int4_fast_ok(device: str | torch.device = "cpu", dtype: torch.dtype = torch.bfloat16) -> bool:
    """Verify the int4 kernel against the dequant reference once per device and dtype.

    PRD 5.5 offers the kernel as a fast path "where available". Available is not
    the same as correct, so this measures it before we rely on it.
    """
    device = torch.device(device)
    key = (device.type, dtype)
    if key in _FAST_CHECKED:
        return _FAST_CHECKED[key]

    ok = False
    try:
        torch.manual_seed(0)
        w = (torch.randn(GROUP * 2, GROUP * 2) * 0.02).to(device)
        qw = quantize(w, "int4")
        x = torch.randn(3, GROUP * 2, dtype=dtype, device=device)
        fast = _int4_mm_fast(x, qw)
        ref = x.float() @ qw.dequantize().float().t()
        denom = ref.norm().clamp(min=EPS)
        ok = bool(fast is not None and ((fast.float() - ref).norm() / denom) < FAST_RTOL)
    except Exception:  # noqa: BLE001 - any failure means the backend cannot be trusted
        ok = False

    _FAST_CHECKED[key] = ok
    return ok


def _int4_mm_fast(x: torch.Tensor, qw: QuantWeight) -> torch.Tensor | None:
    """``x @ W.T`` through the packed-int4 kernel, or ``None`` if the backend has none."""
    ops = _int4_ops(x.device.type)
    if ops is None:
        return None
    convert, mm = ops

    # Packing walks the whole weight, so it must happen once per weight, not once
    # per token. Decode calls this with a single row; rebuilding here costs 10x.
    key = (x.device.type, x.dtype)
    cached = qw._packed.get(key)
    if cached is None:
        scale = _require(qw.scale, "scale").float()
        zero = _require(qw.zero, "zero").float()
        # The kernel reconstructs w = (q - 8) * s + z. Ours is w = (q - zero) * s.
        # Matching the two gives z = s * (8 - zero).
        sz = torch.stack([scale, scale * (8 - zero)], dim=-1).to(x.dtype)
        sz = sz.transpose(0, 1).contiguous()
        q = unpack_nibbles(qw.qweight)[:, : qw.shape[1]].to(torch.int32).contiguous()
        cached = (convert(q, 1), sz)
        qw._packed[key] = cached

    packed, sz = cached
    return mm(x.contiguous(), packed, qw.group, sz)


def linear(
    x: torch.Tensor, qw: QuantWeight, bias: torch.Tensor | None = None, fast: bool | None = None
) -> torch.Tensor:
    """``x @ W.T + bias`` for any tier. ``x`` is ``[n, in]``, the result is ``[n, out]``."""
    if x.shape[-1] != qw.shape[1]:
        raise QuantError(f"x has {x.shape[-1]} input features, weight wants {qw.shape[1]}")

    if qw.tier == "int4":
        if fast is None:
            # The kernel wins on decode and loses on prefill. Pick by batch size.
            rows = x.reshape(-1, x.shape[-1]).shape[0]
            fast = rows <= INT4_KERNEL_MAX_ROWS and int4_fast_ok(x.device, x.dtype)
        if fast:
            out = _int4_mm_fast(x, qw)
            if out is not None:
                return out if bias is None else out + bias

    w = qw.dequantize(x.dtype)
    out = x @ w.t()
    return out if bias is None else out + bias


# --------------------------------------------------------------------------- #
# Measurement helpers (PRD 5.6, 16)
# --------------------------------------------------------------------------- #


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Frobenius relative error, the number the selftest table reports."""
    e = exact.float()
    return float((approx.float() - e).norm() / e.norm().clamp(min=EPS))


def tier_bytes(shape: tuple[int, int], tier: Tier, group: int = GROUP) -> int:
    """Predict bytes on disk for a weight of ``shape`` in ``tier``, without allocating it."""
    out_f, in_f = shape
    if tier == "bf16":
        return out_f * in_f * 2
    if tier == "int8":
        return out_f * in_f + out_f * 2
    groups = (in_f + group - 1) // group
    return out_f * in_f // 2 + out_f * groups * 2 + out_f * groups
