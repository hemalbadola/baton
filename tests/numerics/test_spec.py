"""`ModelSpec` parsing (PRD 5.1).

Every number the decoder uses comes from here. A field read from the wrong key
produces a model that runs and is wrong, so the parse is checked against the
real `config.json` values.
"""

from __future__ import annotations

import json

import pytest

from baton.model.spec import ModelSpec

from .conftest import LLAMA_3_2_1B, QWEN_TINY, make_spec


def test_llama_1b_config_parses():
    spec = make_spec("llama-3.2-1b")
    assert (spec.n_layers, spec.hidden, spec.intermediate) == (16, 2048, 8192)
    assert (spec.n_heads, spec.n_kv_heads, spec.head_dim) == (32, 8, 64)
    assert spec.vocab == 128256
    assert spec.rope_theta == 500000.0
    assert spec.rms_eps == 1e-5
    assert spec.rope_type == "llama3"
    assert spec.tie_embeddings
    assert not spec.attn_bias


def test_derived_geometry():
    spec = make_spec("llama-3.2-1b")
    assert spec.n_kv_groups == 4
    assert spec.q_dim == 2048
    assert spec.kv_dim == 512


def test_head_dim_falls_back_to_hidden_over_heads():
    """Older configs omit `head_dim`."""
    config = {k: v for k, v in LLAMA_3_2_1B.items() if k != "head_dim"}
    assert ModelSpec.from_config(config).head_dim == 64


def test_qwen_flags():
    spec = ModelSpec.from_config(QWEN_TINY)
    assert spec.attn_bias
    assert spec.rope_type == "default"
    assert not spec.tie_embeddings


def test_old_style_rope_type_key_is_read():
    """Pre-2024 configs spell it `type`, not `rope_type`."""
    spec = make_spec("llama-3.2-1b", rope_scaling={"type": "llama3", "factor": 8.0})
    assert spec.rope_type == "llama3"


def test_layer_names_carry_the_index():
    spec = make_spec("llama-3.2-1b")
    names = spec.layer_names(7)
    assert names["q_proj"] == "model.layers.7.self_attn.q_proj.weight"
    assert names["down_proj"] == "model.layers.7.mlp.down_proj.weight"
    assert "q_bias" not in names


def test_bias_names_appear_only_for_qwen():
    assert "q_bias" in ModelSpec.from_config(QWEN_TINY).layer_names(0)


def test_tied_head_points_at_the_embedding():
    """A tied checkpoint has no `lm_head.weight` to fetch (PRD 5.1)."""
    assert make_spec("llama-3.2-1b").head_names()["lm_head"] == "model.embed_tokens.weight"
    assert ModelSpec.from_config(QWEN_TINY).head_names()["lm_head"] == "lm_head.weight"


def test_range_names_cover_one_worker_s_fetch_list():
    """The loader fetches exactly this list for a layer range (PRD 5.3)."""
    spec = make_spec("llama-3.2-1b")
    middle = spec.range_names(4, 8)
    assert len(middle) == 4 * 9  # 2 norms, 4 attention, 3 mlp
    assert spec.tensor_names["embed"] not in middle

    first = spec.range_names(0, 2, embed=True)
    assert first[0] == "model.embed_tokens.weight"

    last = spec.range_names(14, 16, head=True)
    assert "model.norm.weight" in last
    # Tied, so the embedding name appears once and serves as the head.
    assert last.count("model.embed_tokens.weight") == 1


def test_spec_round_trips_through_json():
    """The head sends the spec to every worker inside the `plan` frame."""
    spec = make_spec("llama-3.2-1b")
    assert ModelSpec(**json.loads(json.dumps(spec.to_dict()))) == spec


def test_impossible_geometry_is_refused():
    with pytest.raises(ValueError, match="divisible"):
        make_spec("llama-3.2-1b", num_key_value_heads=7)
    with pytest.raises(ValueError, match="even"):
        make_spec("llama-3.2-1b", head_dim=63)


def test_qwen2_config_without_attention_bias_key_has_biases() -> None:
    """Qwen2 configs carry no `attention_bias`, yet q, k and v have biases. Dropping
    them gave garbage text from a real Qwen2.5-0.5B (BAT-40)."""
    from baton.model.spec import ModelSpec

    config = {
        "model_type": "qwen2",
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    assert ModelSpec.from_config(config).attn_bias
    assert not ModelSpec.from_config(config | {"model_type": "llama"}).attn_bias
