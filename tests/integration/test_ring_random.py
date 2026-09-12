"""Ring versus one process on random weights: identical is identical.

Random weights first (god's dispatch): a split must reproduce the one-process
tokens bit for bit before real weights are worth loading. The tiny geometry
keeps each run under a few seconds, so every branch of the ring is covered
here: chunked prefill, stop ids, `max_tokens`, interleaved requests, k = 3.
"""

from __future__ import annotations

import pytest
import torch

from tests.integration.pipeline import from_wire, generate_in_process, run_ring, to_wire

from .conftest import SEED

RANDOM = ("random", SEED)
TOKENS = 32


def tokens(gens):
    return [g.tokens for g in gens]


@pytest.mark.parametrize("split", [1, 2, 3])
def test_two_processes_match_one_process(tiny_spec, random_prompts, split):
    """The M0 exit condition on random weights, every split point."""
    n = tiny_spec.n_layers
    one = run_ring(tiny_spec, [(0, n)], RANDOM, random_prompts, TOKENS)
    two = run_ring(tiny_spec, [(0, split), (split, n)], RANDOM, random_prompts, TOKENS)

    assert tokens(two) == tokens(one)
    assert all(len(g.tokens) == TOKENS and g.reason == "length" for g in two)
    # Guard on the test itself: a generation that repeats one token would pass
    # with broken layer maths. Each prompt must wander through the vocabulary.
    assert all(len(set(g.tokens)) >= TOKENS // 4 for g in one)


def test_one_process_ring_matches_a_plain_loop(tiny_spec, random_prompts):
    """The k = 1 ring is the baseline. This pins it to a loop with no engine."""
    one = run_ring(tiny_spec, [(0, tiny_spec.n_layers)], RANDOM, random_prompts, TOKENS)
    plain = generate_in_process(tiny_spec, RANDOM, random_prompts, TOKENS)
    assert tokens(one) == plain


def test_three_processes_match(tiny_spec, random_prompts):
    """A middle node forwards hidden states; the ring closes Nk -> N1."""
    one = run_ring(tiny_spec, [(0, 4)], RANDOM, random_prompts, TOKENS)
    three = run_ring(tiny_spec, [(0, 1), (1, 3), (3, 4)], RANDOM, random_prompts, TOKENS)
    assert tokens(three) == tokens(one)


def test_prefill_chunks_cross_the_wire(tiny_spec):
    """A 600-token prompt is three `act` frames of 256, 256, 88 (PRD 10.1)."""
    gen = torch.Generator().manual_seed(SEED + 1)
    prompt = torch.randint(0, tiny_spec.vocab, (600,), generator=gen).tolist()
    plain = generate_in_process(tiny_spec, RANDOM, [prompt], 8, max_ctx=1024)
    two = run_ring(tiny_spec, [(0, 2), (2, 4)], RANDOM, [prompt], 8, max_ctx=1024)
    assert tokens(two) == plain


def test_stop_id_ends_the_request(tiny_spec, random_prompts):
    """`final=true, reason="stop"` and a `release` around the ring (PRD 10.2)."""
    prompt = random_prompts[:1]
    plain = generate_in_process(tiny_spec, RANDOM, prompt, TOKENS)[0]
    stop = plain[5]
    two = run_ring(tiny_spec, [(0, 2), (2, 4)], RANDOM, prompt, TOKENS, stop_ids=(stop,))
    cut = plain.index(stop) + 1
    assert two[0].tokens == plain[:cut]
    assert two[0].reason == "stop"


def test_interleaved_requests_do_not_mix(tiny_spec, random_prompts):
    """All prompts in flight at once. Per-request KV must not leak (PRD 10.5)."""
    one = run_ring(tiny_spec, [(0, 4)], RANDOM, random_prompts, TOKENS)
    two = run_ring(tiny_spec, [(0, 2), (2, 4)], RANDOM, random_prompts, TOKENS, concurrent=True)
    assert tokens(two) == tokens(one)


def test_wire_payload_round_trips(tiny_spec):
    """The payload path does what its name says: fp32 is exact, bf16 is half
    the bytes and equal to a bf16 cast. A bf16 hop on this model flipped no
    token over 8 x 32 greedy steps, so tokens cannot prove the dtype reached
    the wire; the bytes can."""
    x = torch.randn(3, tiny_spec.hidden)
    fp32 = to_wire(x, "fp32")
    assert len(fp32) == 3 * tiny_spec.hidden * 4
    assert torch.equal(from_wire(fp32, 3, tiny_spec.hidden, "fp32"), x)
    bf16 = to_wire(x, "bf16")
    assert len(bf16) == 3 * tiny_spec.hidden * 2
    assert torch.equal(from_wire(bf16, 3, tiny_spec.hidden, "bf16"), x.to(torch.bfloat16))
