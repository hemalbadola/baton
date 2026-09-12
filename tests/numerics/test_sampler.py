"""Sampling on Nk (PRD 10.3).

Sampling has no reference implementation to compare against, so these check the
properties the pipeline relies on: greedy is the argmax, a seed repeats, and
each filter keeps exactly the tokens it claims to keep.
"""

from __future__ import annotations

import pytest
import torch

from baton.worker.sampler import (
    Sampler,
    SamplingParams,
    apply_repetition_penalty,
    top_k_filter,
    top_p_filter,
)

VOCAB = 512


@pytest.fixture
def logits() -> torch.Tensor:
    return torch.randn(VOCAB)


def test_temperature_zero_is_argmax(logits):
    sampler = Sampler(SamplingParams(temperature=0))
    assert sampler.sample(logits) == int(logits.argmax())


def test_greedy_accepts_a_two_dimensional_row(logits):
    """Nk slices one row out of the prefill chunk, so `[1, vocab]` arrives."""
    sampler = Sampler(SamplingParams(temperature=0))
    assert sampler.sample(logits.unsqueeze(0)) == int(logits.argmax())


def test_same_seed_gives_the_same_sequence(logits):
    params = SamplingParams(temperature=0.8, top_k=50, top_p=0.9, seed=42)
    first = [Sampler(params).sample(logits) for _ in range(1)]
    a = Sampler(params)
    b = Sampler(params)
    assert [a.sample(logits) for _ in range(20)] == [b.sample(logits) for _ in range(20)]
    assert first  # the single-draw case is the decode step


def test_different_seeds_diverge(logits):
    a = Sampler(SamplingParams(temperature=1.0, seed=1))
    b = Sampler(SamplingParams(temperature=1.0, seed=2))
    assert [a.sample(logits) for _ in range(20)] != [b.sample(logits) for _ in range(20)]


def test_unseeded_sampler_still_records_its_seed():
    """A run must be repeatable after the fact, so the drawn seed is kept."""
    sampler = Sampler(SamplingParams(temperature=1.0))
    assert isinstance(sampler.seed, int)
    repeat = Sampler(SamplingParams(temperature=1.0, seed=sampler.seed))
    x = torch.randn(VOCAB)
    assert [sampler.sample(x) for _ in range(5)] == [repeat.sample(x) for _ in range(5)]


def test_top_k_keeps_exactly_k(logits):
    for k in (1, 5, 50):
        kept = top_k_filter(logits.clone(), k) > float("-inf")
        assert int(kept.sum()) == k
        assert kept[logits.argmax()]


def test_top_k_zero_keeps_everything(logits):
    assert torch.equal(top_k_filter(logits.clone(), 0), logits)


def test_top_p_keeps_the_smallest_set_that_reaches_p(logits):
    probs = logits.softmax(-1)
    ordered = probs.sort(descending=True).values
    for p in (0.3, 0.6, 0.95):
        kept = top_p_filter(logits.clone(), p) > float("-inf")
        expected = int((ordered.cumsum(0) < p).sum()) + 1
        assert int(kept.sum()) == expected
        assert kept[logits.argmax()]


def test_top_p_one_keeps_everything(logits):
    assert torch.equal(top_p_filter(logits.clone(), 1.0), logits)


def test_sampling_never_draws_a_filtered_token(logits):
    """The whole point of the filters. A drawn token outside top_k is a bug."""
    allowed = set(torch.topk(logits, 5).indices.tolist())
    sampler = Sampler(SamplingParams(temperature=1.0, top_k=5, seed=3))
    assert {sampler.sample(logits) for _ in range(200)} <= allowed


def test_repetition_penalty_lowers_both_signs():
    """Dividing a negative logit would raise it, so the penalty is sign aware."""
    x = torch.tensor([2.0, -2.0, 0.5])
    out = apply_repetition_penalty(x.clone(), [0, 1], 2.0)
    assert out[0] < x[0]
    assert out[1] < x[1]
    assert out[2] == x[2]


def test_repetition_penalty_can_change_the_greedy_pick():
    """Nk applies the penalty over the prompt ids plus the tokens so far."""
    x = torch.tensor([3.0, 2.9, 0.0])
    assert Sampler(SamplingParams(temperature=0)).sample(x) == 0
    penalized = Sampler(SamplingParams(temperature=0, repetition_penalty=2.0))
    assert penalized.sample(x, seen=[0]) == 1


def test_repetition_penalty_does_not_mutate_the_caller_logits():
    """Nk keeps the logits row; a destructive penalty would poison a retry."""
    x = torch.tensor([3.0, 2.9, 0.0])
    Sampler(SamplingParams(temperature=0, repetition_penalty=2.0)).sample(x, seen=[0])
    assert torch.equal(x, torch.tensor([3.0, 2.9, 0.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"repetition_penalty": 0.0},
    ],
)
def test_bad_parameters_are_refused(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)
