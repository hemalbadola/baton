"""The whole decoder stack (PRD 16.1) and the properties M0 depends on.

Three things are checked here that no single-layer test can reach:

* the full stack's logits against `transformers`, embedding and `lm_head`
  included;
* a KV cache decode, one token at a time, equal to one full-sequence forward;
* a stack split across two workers equal to one stack holding every layer.

The third is the M0 exit condition in miniature. If it fails, two processes will
not agree with one process no matter how good the transport is.
"""

from __future__ import annotations

import pytest
import torch

from baton.model.layers import DecoderStack

from .conftest import (
    N_TOKENS,
    SMALL_NAMES,
    TOL_EXACT,
    TOL_FP32,
    RefKVCache,
    make_caches,
    make_hf_config,
    make_spec,
    max_abs_diff,
    randomize,
)


def build_pair(name: str):
    """One reference model and one `DecoderStack` holding the same weights."""
    spec = make_spec(name)
    reference = randomize(
        __import__(
            "transformers", fromlist=["AutoModelForCausalLM"]
        ).AutoModelForCausalLM.from_config(make_hf_config(name))
    )
    ours = DecoderStack(spec, 0, spec.n_layers, embed=True, head=True, max_ctx=256)
    ours.load_hf_weights(dict(reference.state_dict()))
    return spec, ours, reference.eval()


# --- against the reference ----------------------------------------------


@pytest.mark.parametrize("name", SMALL_NAMES)
def test_stack_logits_match_hf(name):
    """Embedding, every layer, the final norm and `lm_head`."""
    spec, ours, reference = build_pair(name)
    ids = torch.randint(0, spec.vocab, (N_TOKENS,))

    got = ours(ours.embed(ids))
    with torch.no_grad():
        want = reference(input_ids=ids.unsqueeze(0)).logits.squeeze(0)

    assert got.shape == want.shape
    assert max_abs_diff(got, want) < TOL_FP32


@pytest.mark.parametrize("name", SMALL_NAMES)
def test_greedy_tokens_match_hf(name):
    """The argmax at every position, which is what a greedy run emits.

    A tolerance can hide a flipped token. This asserts the decision, not the
    distance.
    """
    spec, ours, reference = build_pair(name)
    ids = torch.randint(0, spec.vocab, (N_TOKENS,))

    got = ours(ours.embed(ids)).argmax(-1)
    with torch.no_grad():
        want = reference(input_ids=ids.unsqueeze(0)).logits.squeeze(0).argmax(-1)
    assert torch.equal(got, want)


def test_tied_lm_head_loads_from_the_embedding():
    """Llama 3.2 1B and 3B tie `lm_head` to the embedding table (PRD 5.1).

    A tied checkpoint has no `lm_head.weight`, so a loader that looks for one
    fails on exactly the model M0 uses.
    """
    spec = make_spec("tiny")
    assert spec.tie_embeddings
    assert spec.head_names()["lm_head"] == "model.embed_tokens.weight"
    _, ours, _ = build_pair("tiny")
    assert torch.equal(ours.lm_head.weight, ours.embed_tokens.weight)


# --- the KV cache path ---------------------------------------------------


def test_decode_one_token_at_a_time_matches_one_forward():
    """Cached decode must reproduce the full-sequence answer.

    This is where a wrong `pos_start` shows up: the rope table is indexed by
    absolute position, so an off-by-one rotates every key in the cache.
    """
    spec = make_spec("tiny")
    stack = randomize(DecoderStack(spec, 0, spec.n_layers, embed=True, head=True, max_ctx=128))
    ids = torch.randint(0, spec.vocab, (N_TOKENS,))
    want = stack(stack.embed(ids))

    caches = make_caches(spec, spec.n_layers, 128)
    rows = [stack(stack.embed(ids[i : i + 1]), pos_start=i, cache=caches) for i in range(N_TOKENS)]
    got = torch.cat(rows, dim=0)
    assert max_abs_diff(got, want) < TOL_FP32


@pytest.mark.parametrize("chunk", [1, 7, 16, 64])
def test_chunked_prefill_matches_one_forward(chunk):
    """Prefill arrives in chunks of 256 in the real pipeline (PRD 10.1).

    Every chunk boundary re-enters attention with `kv_len > n`, which is the one
    case the mask has to get right.
    """
    spec = make_spec("tiny")
    stack = randomize(DecoderStack(spec, 0, spec.n_layers, embed=True, head=True, max_ctx=128))
    ids = torch.randint(0, spec.vocab, (N_TOKENS,))
    want = stack(stack.embed(ids))

    caches = make_caches(spec, spec.n_layers, 128)
    rows = [
        stack(stack.embed(ids[i : i + chunk]), pos_start=i, cache=caches)
        for i in range(0, N_TOKENS, chunk)
    ]
    assert max_abs_diff(torch.cat(rows, dim=0), want) < TOL_FP32


def test_prefill_then_decode_matches_one_forward():
    """The real order: one prefill chunk, then single tokens."""
    spec = make_spec("tiny")
    stack = randomize(DecoderStack(spec, 0, spec.n_layers, embed=True, head=True, max_ctx=128))
    ids = torch.randint(0, spec.vocab, (N_TOKENS,))
    want = stack(stack.embed(ids))

    caches = make_caches(spec, spec.n_layers, 128)
    split = 40
    prefill = stack(stack.embed(ids[:split]), pos_start=0, cache=caches)
    steps = [
        stack(stack.embed(ids[i : i + 1]), pos_start=i, cache=caches)
        for i in range(split, N_TOKENS)
    ]
    assert max_abs_diff(torch.cat([prefill, *steps], dim=0), want) < TOL_FP32


def test_cache_holds_the_positions_it_was_given():
    spec = make_spec("tiny")
    cache = RefKVCache(spec, 32)
    k = torch.randn(4, spec.n_kv_heads, spec.head_dim)
    cache.append(k, k, 0)
    k2 = torch.randn(1, spec.n_kv_heads, spec.head_dim)
    all_k, _ = cache.append(k2, k2, 4)
    assert all_k.shape[0] == 5
    assert torch.equal(all_k[:4], k)
    assert torch.equal(all_k[4:], k2)


# --- sharding ------------------------------------------------------------


@pytest.mark.parametrize("split", [1, 2, 3])
def test_split_across_two_workers_matches_one_worker(split):
    """The M0 exit condition in one process: layers [0, s) then [s, n)."""
    spec = make_spec("tiny")
    whole = randomize(DecoderStack(spec, 0, spec.n_layers, embed=True, head=True, max_ctx=128))
    weights = hf_names(whole, spec)

    first = DecoderStack(spec, 0, split, embed=True, max_ctx=128)
    second = DecoderStack(spec, split, spec.n_layers, head=True, max_ctx=128)
    first.load_hf_weights(weights)
    second.load_hf_weights(weights)

    ids = torch.randint(0, spec.vocab, (N_TOKENS,))
    want = whole(whole.embed(ids))
    got = second(first(first.embed(ids)))
    assert max_abs_diff(got, want) < TOL_EXACT


def hf_names(stack: DecoderStack, spec) -> dict[str, torch.Tensor]:
    """Re-key a stack's own weights under checkpoint names, so another stack
    can load them."""
    from baton.model.layers import _LOCAL_SUFFIX

    state = stack.state_dict()
    out: dict[str, torch.Tensor] = {spec.tensor_names["embed"]: state["embed_tokens.weight"]}
    out[spec.tensor_names["final_norm"]] = state["norm.weight"]
    for local, absolute in enumerate(range(stack.layer_start, stack.layer_end)):
        for key, name in spec.layer_names(absolute).items():
            out[name] = state[f"layers.{local}.{_LOCAL_SUFFIX[key]}"]
    return out


def test_last_only_returns_the_final_row():
    """A prefill chunk needs `lm_head` on its last row alone (PRD 5.2 step 8)."""
    spec = make_spec("tiny")
    stack = randomize(DecoderStack(spec, 0, spec.n_layers, embed=True, head=True, max_ctx=128))
    ids = torch.randint(0, spec.vocab, (N_TOKENS,))
    x = stack.embed(ids)
    assert max_abs_diff(stack(x, last_only=True), stack(x)[-1:]) < TOL_EXACT


def test_middle_worker_returns_hidden_states():
    """A worker without `lm_head` passes activations on, not logits."""
    spec = make_spec("tiny")
    stack = DecoderStack(spec, 1, 3, max_ctx=64)
    out = stack(torch.randn(8, spec.hidden))
    assert out.shape == (8, spec.hidden)
    assert stack.n_local_layers == 2


def test_a_bad_layer_range_is_refused():
    spec = make_spec("tiny")
    with pytest.raises(ValueError):
        DecoderStack(spec, 2, spec.n_layers + 1, max_ctx=64)


def test_running_past_the_rope_table_is_refused():
    """Silent truncation here would corrupt long prompts and nothing else."""
    spec = make_spec("tiny")
    stack = DecoderStack(spec, 0, 1, max_ctx=16)
    with pytest.raises(ValueError, match="rope table"):
        stack(torch.randn(8, spec.hidden), pos_start=12)


def test_tie_flag_controls_weight_sharing():
    """Llama ties, Qwen does not. A wrong tie silently reuses the wrong matrix."""
    llama = DecoderStack(make_spec("tiny"), 0, 1, embed=True, head=True, max_ctx=16)
    qwen = DecoderStack(make_spec("qwen-tiny"), 0, 1, embed=True, head=True, max_ctx=16)
    assert llama.lm_head.weight is llama.embed_tokens.weight
    assert qwen.lm_head.weight is not qwen.embed_tokens.weight
