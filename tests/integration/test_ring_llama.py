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


def tokens(gens):
    return [g.tokens for g in gens]


def report_overhead(label: str, gens) -> None:
    """Node-side per-frame overhead from the traces, printed for the report."""
    by_node: dict[str, list[float]] = {}
    for g in gens:
        for trace in g.traces:
            for e in trace:
                by_node.setdefault(e["node"], []).append(e["t_send"] - e["t_recv"] - e["compute"])
    print(f"\n[{label}] node-side per-frame overhead (wall on node minus compute):")
    for node, xs in by_node.items():
        xs.sort()
        p95 = xs[int(0.95 * (len(xs) - 1))]
        print(
            f"  {node}: n={len(xs)} median {statistics.median(xs) * 1e6:.0f} us"
            f"  p95 {p95 * 1e6:.0f} us  max {xs[-1] * 1e6:.0f} us"
        )
    compute = [e["compute"] for g in gens for t in g.traces for e in t]
    print(f"  compute per frame: median {statistics.median(compute) * 1e3:.1f} ms")
    assert overhead_seconds(gens)  # the traces reached the head


@pytest.fixture(scope="module")
def baseline(llama):
    """One process holding every layer, through the same engine loop."""
    spec, weights, prompts = llama
    gens = run_ring(spec, [(0, spec.n_layers)], weights, prompts, M0_TOKENS, **STORE)
    report_overhead("k=1 cpu fp32", gens)
    return gens


def test_two_processes_identical_fp32(llama, baseline):
    """PRD 16.2 first run, and the PRD 19 exit condition. Identical, not close."""
    spec, weights, prompts = llama
    two = run_ring(spec, [(0, SPLIT), (SPLIT, spec.n_layers)], weights, prompts, M0_TOKENS, **STORE)
    report_overhead("k=2 cpu fp32 wire fp32", two)

    assert all(len(g.tokens) == M0_TOKENS for g in two)
    assert tokens(two) == tokens(baseline)


def test_one_process_ring_matches_a_plain_loop(llama, baseline):
    """The baseline is the engine loop. A plain loop with no engine agrees too."""
    spec, weights, prompts = llama
    plain = generate_in_process(spec, weights, prompts, M0_TOKENS, **STORE)
    assert plain == tokens(baseline)


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
    two = run_ring(
        spec,
        [(0, SPLIT), (SPLIT, spec.n_layers)],
        weights,
        prompts,
        M0_TOKENS,
        wire="bf16",
        **STORE,
    )
    report_overhead("k=2 cpu fp32 wire bf16", two)
    matched = 0
    for g, b in zip(two, baseline, strict=True):
        prefix = 0
        while prefix < M0_TOKENS and g.tokens[prefix] == b.tokens[prefix]:
            prefix += 1
        matched += prefix
    print(f"wire bf16 free-running prefix agreement {matched}/{len(two) * M0_TOKENS}")
    assert all(len(g.tokens) == M0_TOKENS for g in two)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_two_processes_identical_mps(llama, baseline):
    """Same backend, two processes versus one: identical is still the bar.

    Also reports MPS against the CPU baseline. That is the cross-backend
    figure of PRD 16.3, not a gate here.
    """
    spec, weights, prompts = llama
    one = run_ring(spec, [(0, spec.n_layers)], weights, prompts, M0_TOKENS, device="mps", **STORE)
    two = run_ring(
        spec,
        [(0, SPLIT), (SPLIT, spec.n_layers)],
        weights,
        prompts,
        M0_TOKENS,
        device="mps",
        **STORE,
    )
    report_overhead("k=2 mps fp32 wire fp32", two)
    assert tokens(two) == tokens(one)

    matched = 0
    for g, b in zip(one, baseline, strict=True):
        prefix = 0
        while prefix < M0_TOKENS and g.tokens[prefix] == b.tokens[prefix]:
            prefix += 1
        matched += prefix
    print(f"\nmps vs cpu free-running prefix agreement {matched}/{len(one) * M0_TOKENS}")
