"""RoPE tables against the Hugging Face reference (PRD 16.1).

This is the test the PRD singles out: "This is the test that catches a wrong
RoPE constant." The llama3 frequency rescale has four constants and three
branches. Every wrong combination still produces readable text.
"""

from __future__ import annotations

import pytest
import torch

from baton.model.layers import _inv_freq, apply_rope, build_rope_tables, rotate_half

from .conftest import (
    ALL_NAMES,
    CONFIGS,
    TOL_EXACT,
    hf_class,
    hf_function,
    make_hf_config,
    make_spec,
    max_abs_diff,
)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_inv_freq_matches_hf(name):
    """The inverse frequencies, before any position is applied."""
    ours = _inv_freq(make_spec(name))
    theirs = hf_class(name, "RotaryEmbedding")(make_hf_config(name)).inv_freq
    assert max_abs_diff(ours, theirs) < TOL_EXACT


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("max_ctx", [64, 4096])
def test_rope_tables_match_hf(name, max_ctx):
    """cos and sin over a whole position range, which is what a worker caches."""
    cos, sin = build_rope_tables(make_spec(name), max_ctx)

    rotary = hf_class(name, "RotaryEmbedding")(make_hf_config(name))
    positions = torch.arange(max_ctx).unsqueeze(0)
    hf_cos, hf_sin = rotary(torch.zeros(1, max_ctx, 1, dtype=torch.float32), positions)

    assert cos.shape == (max_ctx, CONFIGS[name]["head_dim"])
    assert max_abs_diff(cos, hf_cos[0]) < TOL_EXACT
    assert max_abs_diff(sin, hf_sin[0]) < TOL_EXACT


def test_llama3_rescale_is_actually_applied():
    """A llama3 spec must not produce the plain-rope frequencies.

    Without this, a rescale that silently never runs would pass every other test
    in the file, because plain rope and llama3 rope agree at short wavelengths.
    """
    scaled = _inv_freq(make_spec("llama-3.2-1b"))
    plain = _inv_freq(make_spec("llama-3.2-1b", rope_scaling=None))
    assert scaled.shape == plain.shape
    # High frequencies are left alone, low frequencies are divided by `factor`.
    assert torch.equal(scaled[0], plain[0])
    assert scaled[-1].item() == pytest.approx(plain[-1].item() / 32.0, rel=1e-6)
    assert not torch.allclose(scaled, plain)


def test_llama3_smoothing_band_is_reached():
    """At least one frequency falls between the two wavelengths and is blended.

    If none did, the middle branch of the rescale would never run and a bug
    there would hide until a long prompt arrived.
    """
    scaled = _inv_freq(make_spec("llama-3.2-1b"))
    plain = _inv_freq(make_spec("llama-3.2-1b", rope_scaling=None))
    blended = (scaled != plain) & (scaled != plain / 32.0)
    assert blended.any()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_apply_rope_matches_hf(name):
    """The rotation itself, on query and key tensors at a non-zero offset."""
    spec = make_spec(name)
    n, pos_start = 17, 5
    cos, sin = build_rope_tables(spec, 64)
    q = torch.randn(n, spec.n_heads, spec.head_dim)
    k = torch.randn(n, spec.n_kv_heads, spec.head_dim)

    window = slice(pos_start, pos_start + n)
    ours_q = apply_rope(q, cos[window], sin[window])
    ours_k = apply_rope(k, cos[window], sin[window])

    # The reference lays tensors out as [batch, heads, seq, dim].
    hf_q, hf_k = hf_function(name, "apply_rotary_pos_emb")(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        cos[window].unsqueeze(0),
        sin[window].unsqueeze(0),
    )
    assert max_abs_diff(ours_q, hf_q.squeeze(0).transpose(0, 1)) < TOL_EXACT
    assert max_abs_diff(ours_k, hf_k.squeeze(0).transpose(0, 1)) < TOL_EXACT


def test_rope_is_a_rotation():
    """Norms survive the rotation.

    A cheap invariant that needs no reference model, so it still runs on a
    worker where `transformers` is not installed.
    """
    spec = make_spec("tiny")
    cos, sin = build_rope_tables(spec, 32)
    x = torch.randn(32, spec.n_heads, spec.head_dim)
    rotated = apply_rope(x, cos, sin)
    assert max_abs_diff(x.norm(dim=-1), rotated.norm(dim=-1)) < TOL_EXACT


def test_rope_at_position_zero_is_identity():
    spec = make_spec("tiny")
    cos, sin = build_rope_tables(spec, 4)
    x = torch.randn(1, spec.n_heads, spec.head_dim)
    assert max_abs_diff(x, apply_rope(x, cos[:1], sin[:1])) < TOL_EXACT


def test_rotate_half_matches_hf():
    x = torch.randn(3, 4, 8)
    assert torch.equal(rotate_half(x), hf_function("tiny", "rotate_half")(x))


def test_unsupported_rope_type_is_refused():
    """A yarn or longrope checkpoint must fail loudly, not run plain rope."""
    spec = make_spec("llama-3.2-1b", rope_scaling={"rope_type": "yarn", "factor": 4.0})
    with pytest.raises(NotImplementedError):
        _inv_freq(spec)
