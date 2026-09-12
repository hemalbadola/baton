"""Unit tests for baton.model.quant (PRD 5.5)."""

from __future__ import annotations

import pytest
import torch

from baton.model import quant as Q
from baton.model import safetensors_io as sio

OUT, IN = 128, 512


@pytest.fixture
def weight():
    torch.manual_seed(0)
    return torch.randn(OUT, IN) * 0.02


# --------------------------------------------------------------------------- #
# Storage size — the whole point of the tiers
# --------------------------------------------------------------------------- #


def test_int8_halves_bf16_and_int4_quarters_it(weight):
    sizes = {t: Q.quantize(weight, t).nbytes for t in Q.TIERS}
    assert sizes["int8"] < sizes["bf16"] * 0.52
    assert sizes["int4"] < sizes["bf16"] * 0.30
    assert sizes["bf16"] == OUT * IN * 2


def test_tier_bytes_predicts_the_real_size(weight):
    for tier in Q.TIERS:
        assert Q.tier_bytes((OUT, IN), tier) == Q.quantize(weight, tier).nbytes


def test_int4_stores_two_weights_per_byte(weight):
    qw = Q.quantize(weight, "int4")
    assert qw.qweight.shape == (OUT, IN // 2)
    assert qw.qweight.dtype == torch.uint8
    assert qw.scale.shape == (OUT, IN // Q.GROUP)
    assert qw.zero.shape == (OUT, IN // Q.GROUP)


def test_int8_keeps_one_scale_per_output_channel(weight):
    qw = Q.quantize(weight, "int8")
    assert qw.qweight.shape == (OUT, IN)
    assert qw.qweight.dtype == torch.int8
    assert qw.scale.shape == (OUT,)
    assert qw.scale.dtype == torch.float16
    assert qw.zero is None


# --------------------------------------------------------------------------- #
# Accuracy — measured, not assumed
# --------------------------------------------------------------------------- #


def test_int8_beats_int4_and_both_stay_bounded(weight):
    err8 = Q.relative_error(Q.quantize(weight, "int8").dequantize(), weight)
    err4 = Q.relative_error(Q.quantize(weight, "int4").dequantize(), weight)
    assert err8 < err4
    assert err8 < 0.01
    assert err4 < 0.15


def test_bf16_tier_is_lossless_apart_from_the_cast(weight):
    qw = Q.quantize(weight, "bf16")
    assert qw.qweight.dtype == torch.bfloat16
    assert torch.equal(qw.dequantize(torch.bfloat16), weight.to(torch.bfloat16))


def test_int4_error_falls_as_the_group_shrinks(weight):
    errs = [
        Q.relative_error(Q.quantize(weight, "int4", g).dequantize(), weight) for g in (256, 128, 64)
    ]
    assert errs[0] > errs[1] > errs[2]


def test_a_constant_row_survives_both_tiers():
    """A zero-range group would divide by zero without the scale floor."""
    w = torch.full((Q.GROUP, Q.GROUP * 2), 0.25)
    for tier in ("int8", "int4"):
        assert Q.relative_error(Q.quantize(w, tier).dequantize(), w) < 1e-3


def test_an_all_zero_weight_does_not_produce_nan():
    w = torch.zeros(Q.GROUP, Q.GROUP * 2)
    for tier in Q.TIERS:
        assert torch.isfinite(Q.quantize(w, tier).dequantize()).all()


def test_one_large_outlier_costs_int8_more_than_int4():
    """int8 is per row, int4 is per group of 128, so an outlier hurts a whole row."""
    torch.manual_seed(1)
    w = torch.randn(OUT, IN) * 0.02
    w[0, 0] = 50.0
    err8 = Q.relative_error(Q.quantize(w[:1], "int8").dequantize(), w[:1])
    err4 = Q.relative_error(Q.quantize(w[:1], "int4").dequantize(), w[:1])
    assert err8 > err4


def test_quantized_values_stay_inside_their_range(weight):
    q8 = Q.quantize(weight, "int8").qweight
    assert q8.min() >= -127 and q8.max() <= 127
    nib = Q.unpack_nibbles(Q.quantize(weight, "int4").qweight)
    assert nib.min() >= 0 and nib.max() <= 15


# --------------------------------------------------------------------------- #
# Nibble packing
# --------------------------------------------------------------------------- #


def test_pack_then_unpack_is_the_identity():
    q = torch.randint(0, 16, (7, 64), dtype=torch.uint8)
    assert torch.equal(Q.unpack_nibbles(Q.pack_nibbles(q)), q)


def test_low_nibble_holds_the_even_index():
    q = torch.tensor([[1, 2]], dtype=torch.uint8)
    assert Q.pack_nibbles(q).item() == 1 | (2 << 4)


def test_an_odd_last_dimension_cannot_pack():
    with pytest.raises(Q.QuantError):
        Q.pack_nibbles(torch.zeros(3, dtype=torch.uint8))


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_unknown_tier_raises(weight):
    with pytest.raises(Q.QuantError):
        Q.quantize(weight, "int2")


def test_a_non_2d_weight_raises():
    with pytest.raises(Q.QuantError):
        Q.quantize(torch.randn(4, 4, 4), "int4")


def test_an_input_dim_that_is_not_a_multiple_of_the_group_raises():
    with pytest.raises(Q.QuantError):
        Q.quantize(torch.randn(4, 130), "int4")


def test_a_mismatched_input_width_raises(weight):
    qw = Q.quantize(weight, "int8")
    with pytest.raises(Q.QuantError):
        Q.linear(torch.randn(2, IN + 1), qw)


# --------------------------------------------------------------------------- #
# linear dispatch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tier", Q.TIERS)
def test_linear_matches_the_dequantized_reference(weight, tier):
    qw = Q.quantize(weight, tier)
    x = torch.randn(4, IN, dtype=torch.bfloat16)
    got = Q.linear(x, qw)
    ref = x @ qw.dequantize(x.dtype).t()
    assert got.shape == (4, OUT)
    assert Q.relative_error(got, ref) < 2e-2


@pytest.mark.parametrize("tier", Q.TIERS)
def test_linear_tracks_the_exact_matmul_within_the_tier_error(weight, tier):
    x = torch.randn(4, IN)
    exact = x @ weight.t()
    got = Q.linear(x.to(torch.float32), Q.quantize(weight, tier))
    bound = {"bf16": 0.02, "int8": 0.02, "int4": 0.20}[tier]
    assert Q.relative_error(got, exact) < bound


@pytest.mark.parametrize("tier", Q.TIERS)
def test_bias_is_added(weight, tier):
    qw = Q.quantize(weight, tier)
    x = torch.randn(2, IN)
    bias = torch.arange(OUT, dtype=torch.float32)
    assert torch.allclose(Q.linear(x, qw, bias), Q.linear(x, qw) + bias, atol=1e-3)


def test_the_int4_fast_path_agrees_with_the_dequant_path(weight):
    """The fast kernel is only used after it matches. This is that check."""
    if not Q.int4_fast_ok("cpu", torch.bfloat16):
        pytest.skip("no verified int4 kernel on this backend")
    qw = Q.quantize(weight, "int4")
    x = torch.randn(4, IN, dtype=torch.bfloat16)
    assert Q.relative_error(Q.linear(x, qw, fast=True), Q.linear(x, qw, fast=False)) < Q.FAST_RTOL


def test_the_fast_path_verdict_is_cached():
    Q._FAST_CHECKED.clear()
    first = Q.int4_fast_ok("cpu", torch.bfloat16)
    assert ("cpu", torch.bfloat16) in Q._FAST_CHECKED
    assert Q.int4_fast_ok("cpu", torch.bfloat16) is first


def test_a_broken_kernel_is_not_used(monkeypatch, weight):
    """If the backend kernel returns nonsense, linear must fall back, not trust it."""
    monkeypatch.setattr(Q, "_int4_mm_fast", lambda x, qw: torch.zeros(x.shape[0], qw.shape[0]))
    Q._FAST_CHECKED.clear()
    assert Q.int4_fast_ok("cpu", torch.float32) is False

    Q._FAST_CHECKED.clear()
    qw = Q.quantize(weight, "int4")
    x = torch.randn(2, IN)
    out = Q.linear(x, qw)  # fast=None, so it consults the verdict
    assert not torch.equal(out, torch.zeros_like(out))
    Q._FAST_CHECKED.clear()


# --------------------------------------------------------------------------- #
# Persistence and compute dtype
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tier", Q.TIERS)
def test_a_quantized_weight_round_trips_through_safetensors(tmp_path, weight, tier):
    qw = Q.quantize(weight, tier)
    path = sio.save_safetensors(tmp_path / "w.safetensors", qw.to_tensors(), {"tier": tier})
    header, header_len = sio.read_header(path)
    refs = list(sio.refs_from_header(path.name, header, header_len).values())
    tensors = sio.load_tensors(refs, sio.LocalSource(tmp_path))

    back = Q.QuantWeight.from_tensors(tier, qw.shape, tensors, qw.group)
    assert torch.equal(back.dequantize(), qw.dequantize())


def test_to_device_keeps_the_values(weight):
    qw = Q.quantize(weight, "int4")
    assert torch.equal(qw.to("cpu").dequantize(), qw.dequantize())


def test_compute_dtype_is_one_of_the_three_allowed(weight):
    assert Q.compute_dtype("cpu") in (torch.bfloat16, torch.float32)


def test_compute_dtype_rejects_nothing_and_defaults_to_cpu():
    assert Q.compute_dtype() == Q.compute_dtype("cpu")
