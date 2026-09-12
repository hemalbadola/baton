"""Every sub-block of the decoder layer against Hugging Face (PRD 16.1).

Each test builds our module and the reference module, copies one set of weights
into both, and compares the output for a 64-token random input. Our submodule
names match the checkpoint names, so the copy is a plain `load_state_dict` with
no mapping, and a rename on either side fails loudly here.
"""

from __future__ import annotations

import pytest
import torch

from baton.model.layers import MLP, Attention, DecoderLayer, RMSNorm, build_rope_tables

from .conftest import (
    LLAMA_3_2_1B,
    N_TOKENS,
    SMALL_NAMES,
    TOL_BF16,
    TOL_EXACT,
    TOL_FP32,
    hf_causal_mask,
    hf_class,
    hf_position_embeddings,
    make_hf_config,
    make_spec,
    max_abs_diff,
    randomize,
)

# --- RMSNorm -------------------------------------------------------------


@pytest.mark.parametrize("dtype,tol", [(torch.float32, TOL_EXACT), (torch.bfloat16, TOL_BF16)])
def test_rmsnorm_matches_hf(dtype, tol):
    hidden, eps = 2048, LLAMA_3_2_1B["rms_norm_eps"]
    ours = randomize(RMSNorm(hidden, eps))
    theirs = hf_class("tiny", "RMSNorm")(hidden, eps)
    theirs.load_state_dict(ours.state_dict())

    x = torch.randn(N_TOKENS, hidden).to(dtype)
    assert max_abs_diff(ours.to(dtype)(x), theirs.to(dtype)(x)) < tol


def test_rmsnorm_reduces_in_fp32():
    """A bf16 input must still be normalized by an fp32 mean (PRD 16.5).

    A bf16 sum of squares over 2048 elements loses several bits. The check is
    against the fp32 answer, not against a bf16 one.
    """
    norm = randomize(RMSNorm(2048, 1e-5))
    x = torch.randn(N_TOKENS, 2048) * 8
    reference = norm(x)
    in_bf16 = norm.to(torch.bfloat16)(x.to(torch.bfloat16))
    assert max_abs_diff(in_bf16, reference) < TOL_BF16


# --- MLP -----------------------------------------------------------------


@pytest.mark.parametrize("name", SMALL_NAMES)
def test_mlp_matches_hf(name):
    spec = make_spec(name)
    ours = randomize(MLP(spec))
    theirs = hf_class(name, "MLP")(make_hf_config(name))
    theirs.load_state_dict(ours.state_dict())

    x = torch.randn(N_TOKENS, spec.hidden)
    assert max_abs_diff(ours(x), theirs(x)) < TOL_FP32


# --- attention -----------------------------------------------------------


@pytest.mark.parametrize("name", SMALL_NAMES)
def test_attention_matches_hf(name):
    """Grouped-query attention with a causal mask over a prefill chunk."""
    spec = make_spec(name)
    ours = randomize(Attention(spec))
    theirs = hf_class(name, "Attention")(make_hf_config(name), layer_idx=0)
    theirs.load_state_dict(ours.state_dict())

    x = torch.randn(N_TOKENS, spec.hidden)
    cos, sin = build_rope_tables(spec, N_TOKENS)
    got = ours(x, cos, sin, pos_start=0)

    want, _ = theirs(
        x.unsqueeze(0),
        position_embeddings=hf_position_embeddings(name, N_TOKENS),
        attention_mask=hf_causal_mask(N_TOKENS),
    )
    assert max_abs_diff(got, want.squeeze(0)) < TOL_FP32


def test_attention_is_causal():
    """Changing a later token must not move an earlier output.

    Without this, a missing mask would still pass the reference comparison,
    because the reference would be handed the same missing mask.
    """
    spec = make_spec("tiny")
    attn = randomize(Attention(spec))
    cos, sin = build_rope_tables(spec, N_TOKENS)
    x = torch.randn(N_TOKENS, spec.hidden)
    base = attn(x, cos, sin)

    changed = x.clone()
    changed[-1] += 5.0
    after = attn(changed, cos, sin)
    assert max_abs_diff(base[:-1], after[:-1]) < TOL_EXACT
    assert max_abs_diff(base[-1], after[-1]) > 1e-3


def test_attention_uses_grouped_query_heads():
    """8 query heads share 2 key/value heads, so the group size is 4."""
    spec = make_spec("tiny")
    assert spec.n_kv_groups == 4
    attn = Attention(spec)
    assert attn.k_proj.out_features == spec.n_kv_heads * spec.head_dim
    assert attn.q_proj.out_features == spec.n_heads * spec.head_dim


# --- the whole layer -----------------------------------------------------


@pytest.mark.parametrize("name", SMALL_NAMES)
def test_decoder_layer_matches_hf(name):
    spec = make_spec(name)
    ours = randomize(DecoderLayer(spec))
    theirs = hf_class(name, "DecoderLayer")(make_hf_config(name), layer_idx=0)
    theirs.load_state_dict(ours.state_dict())

    x = torch.randn(N_TOKENS, spec.hidden)
    cos, sin = build_rope_tables(spec, N_TOKENS)
    got = ours(x, cos, sin, pos_start=0)

    want = theirs(
        x.unsqueeze(0),
        position_embeddings=hf_position_embeddings(name, N_TOKENS),
        attention_mask=hf_causal_mask(N_TOKENS),
    )
    assert max_abs_diff(got, want.squeeze(0)) < TOL_FP32


@pytest.mark.slow
def test_decoder_layer_matches_hf_at_llama_1b_geometry():
    """The same comparison at the real shape: hidden 2048, 32 heads, 8 kv heads.

    About 500 MB of fp32 weights across the two copies, so it is one layer and
    not the stack. Anything the size changes -- head count, group count,
    intermediate width -- is covered only here.
    """
    name = "llama-3.2-1b"
    spec = make_spec(name)
    ours = randomize(DecoderLayer(spec))
    theirs = hf_class(name, "DecoderLayer")(make_hf_config(name), layer_idx=0)
    theirs.load_state_dict(ours.state_dict())

    x = torch.randn(N_TOKENS, spec.hidden)
    cos, sin = build_rope_tables(spec, N_TOKENS)
    got = ours(x, cos, sin, pos_start=0)

    want = theirs(
        x.unsqueeze(0),
        position_embeddings=hf_position_embeddings(name, N_TOKENS),
        attention_mask=hf_causal_mask(N_TOKENS),
    )
    assert max_abs_diff(got, want.squeeze(0)) < TOL_FP32


def test_qwen_bias_is_on_qkv_only():
    """Qwen 2.5 carries a bias on q, k and v, and on nothing else (PRD 5.1)."""
    qwen = Attention(make_spec("qwen-tiny"))
    llama = Attention(make_spec("tiny"))
    assert qwen.q_proj.bias is not None
    assert qwen.k_proj.bias is not None
    assert qwen.v_proj.bias is not None
    assert qwen.o_proj.bias is None
    assert llama.q_proj.bias is None


def test_our_state_dict_keys_match_the_reference():
    """The checkpoint loads with no name mapping only while this holds."""
    ours = set(DecoderLayer(make_spec("tiny")).state_dict())
    theirs = set(hf_class("tiny", "DecoderLayer")(make_hf_config("tiny"), 0).state_dict())
    assert ours == theirs


@pytest.mark.parametrize("pos_start", [0, 5, 100])
def test_attention_is_causal_without_a_cache(pos_start):
    """A chunk run with no cache is still causal inside itself.

    The keys then start at `pos_start`, not at 0. Reading their positions as 0
    makes every key look older than every query and masks nothing.
    """
    spec = make_spec("tiny")
    attn = randomize(Attention(spec))
    cos, sin = build_rope_tables(spec, 256)
    window = slice(pos_start, pos_start + N_TOKENS)
    x = torch.randn(N_TOKENS, spec.hidden)

    base = attn(x, cos[window], sin[window], pos_start=pos_start)
    changed = x.clone()
    changed[-1] += 5.0
    after = attn(changed, cos[window], sin[window], pos_start=pos_start)
    assert max_abs_diff(base[:-1], after[:-1]) < TOL_EXACT
