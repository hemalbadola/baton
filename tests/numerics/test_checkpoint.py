"""The gate on a real checkpoint (PRD 16.1).

Everything else in this directory copies one random set of weights into both
implementations. That proves the arithmetic and the module tree, but not that a
shipped checkpoint loads under the names `ModelSpec` predicts.

This file runs only when a checkpoint is on disk. Point `BATON_CHECKPOINT` at a
directory holding `config.json` and `model.safetensors`, or let it find
Llama-3.2-1B in the Hugging Face cache. The tests skip otherwise, so CI on a
machine without the weights still runs green.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from baton.model.layers import DecoderStack
from baton.model.spec import ModelSpec

from .conftest import LLAMA_3_2_1B, TOL_BF16, max_abs_diff

N_TOKENS = 64
REPOS = ("unsloth/Llama-3.2-1B", "meta-llama/Llama-3.2-1B", "NousResearch/Llama-3.2-1B")


def find_checkpoint() -> Path | None:
    """A local directory with `config.json` and `model.safetensors`, or None."""
    named = os.environ.get("BATON_CHECKPOINT")
    if named:
        return Path(named)
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    for repo in REPOS:
        root = hub / ("models--" + repo.replace("/", "--")) / "snapshots"
        for snapshot in sorted(root.glob("*")) if root.is_dir() else []:
            if (snapshot / "config.json").is_file() and (snapshot / "model.safetensors").is_file():
                return snapshot
    return None


CHECKPOINT = find_checkpoint()
needs_weights = pytest.mark.skipif(
    CHECKPOINT is None, reason="no local checkpoint; set BATON_CHECKPOINT"
)


@pytest.fixture(scope="module")
def checkpoint():
    """The spec and the raw tensors, loaded once for the whole module."""
    from safetensors.torch import load_file

    spec = ModelSpec.from_pretrained(CHECKPOINT)
    return spec, load_file(str(CHECKPOINT / "model.safetensors"))


@needs_weights
def test_real_config_matches_the_values_the_tests_assume(checkpoint):
    """The constants in `conftest.py` are the shipped ones, not a guess."""
    spec, _ = checkpoint
    expected = ModelSpec.from_config(LLAMA_3_2_1B)
    for field in ("n_layers", "hidden", "intermediate", "n_heads", "n_kv_heads", "head_dim"):
        assert getattr(spec, field) == getattr(expected, field)
    assert spec.rope_theta == expected.rope_theta
    assert spec.rope_scaling == expected.rope_scaling
    assert spec.tie_embeddings == expected.tie_embeddings


@needs_weights
def test_every_predicted_tensor_name_is_in_the_checkpoint(checkpoint):
    """`range_names` drives the byte-range fetch (PRD 5.3). A name that is not
    in the file means a worker downloads nothing and loads zeros."""
    spec, tensors = checkpoint
    wanted = spec.range_names(0, spec.n_layers, embed=True, head=True)
    missing = [name for name in wanted if name not in tensors]
    assert not missing, missing


@needs_weights
def test_one_real_layer_matches_hf_in_fp32(checkpoint):
    """A single shipped layer, fp32, against `transformers` (PRD 16.1)."""
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer,
        LlamaRotaryEmbedding,
    )

    from baton.model.layers import DecoderLayer, build_rope_tables

    from .conftest import TOL_FP32, hf_causal_mask, make_hf_config

    spec, tensors = checkpoint
    layer_idx = 5
    names = spec.layer_names(layer_idx)
    weights = {k: tensors[v].to(torch.float32) for k, v in names.items()}

    ours = DecoderLayer(spec)
    from baton.model.layers import _LOCAL_SUFFIX

    ours.load_state_dict({_LOCAL_SUFFIX[k]: v for k, v in weights.items()})

    config = make_hf_config("llama-3.2-1b")
    theirs = LlamaDecoderLayer(config, layer_idx=layer_idx)
    theirs.load_state_dict({_LOCAL_SUFFIX[k]: v for k, v in weights.items()})

    x = torch.randn(N_TOKENS, spec.hidden) * 0.5
    cos, sin = build_rope_tables(spec, N_TOKENS)
    got = ours(x, cos, sin, pos_start=0)

    rotary = LlamaRotaryEmbedding(config)
    hf_cos, hf_sin = rotary(torch.zeros(1, N_TOKENS, 1), torch.arange(N_TOKENS).unsqueeze(0))
    want = theirs(
        x.unsqueeze(0),
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=hf_causal_mask(N_TOKENS),
    )
    assert max_abs_diff(got, want.squeeze(0)) < TOL_FP32


@needs_weights
def test_full_model_greedy_tokens_match_hf_in_bf16(checkpoint):
    """The whole shipped model, both sides in bf16.

    fp32 would need about 20 GB across the two copies, so this runs at the
    weight dtype the checkpoint ships in. The assertion is on the greedy pick,
    which is what a user sees, plus the PRD 16.2 logits bound.
    """
    from transformers import AutoModelForCausalLM

    spec, tensors = checkpoint
    ids = torch.randint(0, spec.vocab, (N_TOKENS,))

    ours = DecoderStack(
        spec, 0, spec.n_layers, embed=True, head=True, max_ctx=N_TOKENS, dtype=torch.bfloat16
    )
    ours.load_hf_weights(tensors)
    with torch.no_grad():
        got = ours(ours.embed(ids))
    del ours

    reference = AutoModelForCausalLM.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        want = reference(input_ids=ids.unsqueeze(0)).logits.squeeze(0)
    del reference

    assert torch.equal(got.argmax(-1), want.argmax(-1))
    assert max_abs_diff(got, want) < TOL_BF16


@needs_weights
def test_two_shards_match_one_shard_on_real_weights(checkpoint):
    """Layers 0-7 on one worker and 8-15 on another, as M0 splits them."""
    spec, tensors = checkpoint
    ids = torch.randint(0, spec.vocab, (N_TOKENS,))
    half = spec.n_layers // 2

    whole = DecoderStack(
        spec, 0, spec.n_layers, embed=True, head=True, max_ctx=N_TOKENS, dtype=torch.bfloat16
    )
    whole.load_hf_weights(tensors)
    with torch.no_grad():
        want = whole(whole.embed(ids))
    del whole

    first = DecoderStack(spec, 0, half, embed=True, max_ctx=N_TOKENS, dtype=torch.bfloat16)
    second = DecoderStack(
        spec, half, spec.n_layers, head=True, max_ctx=N_TOKENS, dtype=torch.bfloat16
    )
    first.load_hf_weights(tensors)
    second.load_hf_weights(tensors)
    with torch.no_grad():
        got = second(first(first.embed(ids)))

    # Same operations in the same order, so this is an equality, not a tolerance.
    assert torch.equal(got, want)
