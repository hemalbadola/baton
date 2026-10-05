"""The M0 exit test on the real Llama-3.2-1B (PRD 16.2, PRD 19).

Two processes on one machine must generate 64 greedy tokens identical to one
process, for 20 prompts, with `--wire fp32` and fp32 compute. Then the same
split with `--wire bf16`: greedy agreement of at least 99 % over the first 64
positions and a logits gap under 0.05 at position 0.

Everything here skips when the weights are not on disk. Nothing downloads.
Set `BATON_M0_PROMPTS` / `BATON_M0_TOKENS` to shrink a local run.
"""

from __future__ import annotations

import statistics

import pytest
import torch

from tests.integration.pipeline import (
    NodeConfig,
    build_stack,
    generate_in_process,
    overhead_seconds,
    run_ring,
)

from .conftest import M0_TOKENS

pytestmark = pytest.mark.slow

SPLIT = 8  # of 16 layers
# The checkpoint is bf16. Keep it that way in memory and compute in fp32: the
# upcast is exact and the machine has 8 GB (see `pipeline.UpcastLinear`).
STORE = {"storage": "bf16"}
BF16_AGREEMENT = 0.99  # PRD 16.2
BF16_LOGIT_GAP = 0.05  # PRD 16.2, position 0
# Secondary checks run on a few prompts: at ~0.5 s per token on this CPU the
# 20-prompt runs are for the gates, and these only confirm a path works.
FEW = 3


def tokens(gens):
    return [g.tokens for g in gens]


def report_overhead(label: str, stats: dict[str, list]) -> None:
    """Node-side per-frame overhead and stage compute, printed for the report."""
    assert stats, "no node wrote its stats file"
    print(f"\n[{label}] node-side per-frame overhead (wall on node minus compute):")
    for node, samples in sorted(stats.items()):
        xs = sorted(overhead_seconds(samples))
        p95 = xs[int(0.95 * (len(xs) - 1))]
        compute = statistics.median(c for _, _, c in samples)
        print(
            f"  {node}: n={len(xs)} median {statistics.median(xs) * 1e6:.0f} us"
            f"  p95 {p95 * 1e6:.0f} us  max {xs[-1] * 1e6:.0f} us"
            f"  | compute median {compute * 1e3:.1f} ms"
        )


def prefix_agreement(gens, base) -> str:
    """Matching prefix length summed over prompts. A free-running comparison:
    one early flip changes every later token, so this is a report figure."""
    matched = 0
    for g, b in zip(gens, base, strict=True):
        prefix = 0
        while prefix < M0_TOKENS and g.tokens[prefix] == b.tokens[prefix]:
            prefix += 1
        matched += prefix
    return f"{matched}/{len(gens) * M0_TOKENS}"


@pytest.fixture(scope="module")
def baseline(llama):
    """One process holding every layer, through the same engine loop."""
    spec, weights, prompts = llama
    stats: dict = {}
    gens = run_ring(spec, [(0, spec.n_layers)], weights, prompts, M0_TOKENS, stats=stats, **STORE)
    report_overhead("k=1 cpu fp32", stats)
    return gens


def test_two_processes_identical_fp32(llama, baseline):
    """PRD 16.2 first run, and the PRD 19 exit condition. Identical, not close."""
    spec, weights, prompts = llama
    stats: dict = {}
    two = run_ring(
        spec,
        [(0, SPLIT), (SPLIT, spec.n_layers)],
        weights,
        prompts,
        M0_TOKENS,
        stats=stats,
        **STORE,
    )
    report_overhead("k=2 cpu fp32 wire fp32", stats)

    assert all(len(g.tokens) == M0_TOKENS for g in two)
    assert tokens(two) == tokens(baseline)


def test_one_process_ring_matches_a_plain_loop(llama, baseline):
    """The baseline is the engine loop. A plain loop with no engine agrees too."""
    spec, weights, prompts = llama
    plain = generate_in_process(spec, weights, prompts[:FEW], M0_TOKENS, **STORE)
    assert plain == tokens(baseline[:FEW])


def test_wire_bf16_agreement(llama, baseline):
    """PRD 16.2 second run, teacher-forced in one process.

    The ring generates freely, so one early flip changes every later token
    and per-position agreement stops meaning anything. Teacher forcing is what
    the 99 % figure describes: at every position, given the fp32 context, does
    a bf16 hop pick the same token? One forward per prompt answers it exactly.
    """
    spec, weights, prompts = llama
    first = build_stack(NodeConfig("a", 0, "", "", spec, 0, SPLIT, weights, **STORE))
    second = build_stack(NodeConfig("b", 0, "", "", spec, SPLIT, spec.n_layers, weights, **STORE))

    agree = total = 0
    worst_gap = 0.0
    with torch.inference_mode():
        for prompt, base in zip(prompts, baseline, strict=True):
            ids = torch.tensor([*prompt, *base.tokens])
            h = first(first.embed(ids).float())
            want_logits = second(h)
            got_logits = second(h.to(torch.bfloat16).to(torch.float32))
            # Position 0 of the generation is the last prompt row.
            p0 = len(prompt) - 1
            window = slice(p0, p0 + M0_TOKENS)
            want = want_logits[window].argmax(-1)
            got = got_logits[window].argmax(-1)
            agree += int((want == got).sum())
            total += M0_TOKENS
            gap = (want_logits[p0] - got_logits[p0]).abs().max().item()
            worst_gap = max(worst_gap, gap)

    print(f"\nwire bf16 teacher-forced agreement {agree}/{total} = {agree / total:.4f}")
    print(f"wire bf16 logits max abs gap at position 0: {worst_gap:.4f}")
    assert agree / total >= BF16_AGREEMENT
    assert worst_gap < BF16_LOGIT_GAP


def test_wire_bf16_ring_runs(llama, baseline):
    """The free-running bf16 ring, reported. Divergence after a low-margin
    token is expected and is not a failure (PRD 16.2)."""
    spec, weights, prompts = llama
    stats: dict = {}
    two = run_ring(
        spec,
        [(0, SPLIT), (SPLIT, spec.n_layers)],
        weights,
        prompts[:FEW],
        M0_TOKENS,
        wire="bf16",
        stats=stats,
        **STORE,
    )
    report_overhead("k=2 cpu fp32 wire bf16", stats)
    print(f"wire bf16 free-running prefix agreement {prefix_agreement(two, baseline[:FEW])}")
    assert all(len(g.tokens) == M0_TOKENS for g in two)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_two_processes_identical_mps(llama, baseline):
    """Same backend, two processes versus one: identical is still the bar.

    Also reports MPS against the CPU baseline. That is the cross-backend
    figure of PRD 16.3, not a gate here.
    """
    spec, weights, prompts = llama
    one = run_ring(spec, [(0, spec.n_layers)], weights, prompts, M0_TOKENS, device="mps", **STORE)
    stats: dict = {}
    two = run_ring(
        spec,
        [(0, SPLIT), (SPLIT, spec.n_layers)],
        weights,
        prompts,
        M0_TOKENS,
        device="mps",
        stats=stats,
        **STORE,
    )
    report_overhead("k=2 mps fp32 wire fp32", stats)
    print(f"\nmps vs cpu free-running prefix agreement {prefix_agreement(one, baseline)}")
    assert tokens(two) == tokens(one)
