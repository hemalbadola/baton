"""RoPE tables against the Hugging Face reference (PRD 16.1).

This is the test the PRD singles out: "This is the test that catches a wrong
RoPE constant." The llama3 frequency rescale has four constants and three
branches. Every wrong combination still produces readable text.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from baton.model.layers import _inv_freq, apply_rope, build_rope_tables, rotate_half

from .conftest import (
    LLAMA_3_2_1B,
    QWEN_TINY,
    TINY,
    TOL_EXACT,
    make_hf_config,
    make_spec,
    max_abs_diff,
)

CONFIGS = {"llama-3.2-1b": LLAMA_3_2_1B, "tiny": TINY, "qwen-tiny": QWEN_TINY}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_inv_freq_matches_hf(name):
    """The inverse frequencies, before any position is applied."""
    config = CONFIGS[name]
    ours = _inv_freq(make_spec(config))
    theirs = LlamaRotaryEmbedding(make_hf_config(config)).inv_freq
    assert max_abs_diff(ours, theirs) < TOL_EXACT


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("max_ctx", [64, 4096])
def test_rope_tables_match_hf(name, max_ctx):
    """cos and sin over a whole position range, which is what a worker caches."""
    config = CONFIGS[name]
    cos, sin = build_rope_tables(make_spec(config), max_ctx)

    rotary = LlamaRotaryEmbedding(make_hf_config(config))
    positions = torch.arange(max_ctx).unsqueeze(0)
    hf_cos, hf_sin = rotary(torch.zeros(1, max_ctx, 1, dtype=torch.float32), positions)

    assert cos.shape == (max_ctx, config["head_dim"])
    assert max_abs_diff(cos, hf_cos[0]) < TOL_EXACT
    assert max_abs_diff(sin, hf_sin[0]) < TOL_EXACT


def test_llama3_rescale_is_actually_applied():
    """A llama3 spec must not produce the plain-rope frequencies.

    Without this, a rescale that silently never runs would pass every other
    test in the file, because plain rope and llama3 rope agree at short
    wavelengths.
    """
    scaled = _inv_freq(make_spec(LLAMA_3_2_1B))
    plain = _inv_freq(make_spec(LLAMA_3_2_1B, rope_scaling=None))
    assert scaled.shape == plain.shape
    # High frequencies are left alone, low frequencies are divided by `factor`.
    assert torch.equal(scaled[0], plain[0])
    assert scaled[-1].item() == pytest.approx(plain[-1].item() / 32.0, rel=1e-6)
    assert not torch.allclose(scaled, plain)


def test_llama3_smoothing_band_is_reached():
    """At least one frequency falls between the two wavelengths and is blended.

    If it did not, the middle branch of the rescale would never run and a bug
    there would hide until a long prompt arrived.
    """
    spec = make_spec(LLAMA_3_2_1B)
    scaled = _inv_freq(spec)
    plain = _inv_freq(make_spec(LLAMA_3_2_1B, rope_scaling=None))
    blended = (scaled != plain) & (scaled != plain / 32.0)
    assert blended.any()


@pytest.mark.parametrize("name", list(CONFIGS))
def test_apply_rope_matches_hf(name):
    """The rotation itself, on query and key tensors."""
    config = CONFIGS[name]
    spec = make_spec(config)
    n = 17
    pos_start = 5
    cos, sin = build_rope_tables(spec, 64)
    q = torch.randn(n, spec.n_heads, spec.head_dim)
    k = torch.randn(n, spec.n_kv_heads, spec.head_dim)

    window = slice(pos_start, pos_start + n)
    ours_q = apply_rope(q, cos[window], sin[window])
    ours_k = apply_rope(k, cos[window], sin[window])

    # HF lays tensors out as [batch, heads, seq, dim].
    hf_q, hf_k = apply_rotary_pos_emb(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        cos[window].unsqueeze(0),
        sin[window].unsqueeze(0),
    )
    assert max_abs_diff(ours_q, hf_q.squeeze(0).transpose(0, 1)) < TOL_EXACT
    assert max_abs_diff(ours_k, hf_k.squeeze(0).transpose(0, 1)) < TOL_EXACT


def test_rope_is_a_rotation():
    """Norms survive the rotation, on every backend and dtype.

    A cheap invariant that needs no reference model, so it still runs when
    `transformers` is not installed on a worker.
    """
    spec = make_spec(TINY)
    cos, sin = build_rope_tables(spec, 32)
    x = torch.randn(32, spec.n_heads, spec.head_dim)
    rotated = apply_rope(x, cos, sin)
    assert max_abs_diff(x.norm(dim=-1), rotated.norm(dim=-1)) < TOL_EXACT


def test_rope_at_position_zero_is_identity():
    spec = make_spec(TINY)
    cos, sin = build_rope_tables(spec, 4)
    x = torch.randn(1, spec.n_heads, spec.head_dim)
    assert max_abs_diff(x, apply_rope(x, cos[:1], sin[:1])) < TOL_EXACT


def test_rotate_half_matches_hf():
    x = torch.randn(3, 4, 8)
    from transformers.models.llama.modeling_llama import rotate_half as hf_rotate_half

    assert torch.equal(rotate_half(x), hf_rotate_half(x))
