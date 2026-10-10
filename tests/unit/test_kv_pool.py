"""KV cache accounting tests (PRD 6.5). These protect the one place where a
worker can silently over-commit and run out of memory mid-generation."""

import pytest
import torch

from baton.worker.engine import KVPool, OutOfMemoryOnKV

LAYERS, PER_TOKEN = 4, 2 * 2 * 16 * 4  # 4 layers; K and V, 2 kv heads, head_dim 16, fp32


def pool(budget: int) -> KVPool:
    return KVPool(budget, LAYERS, PER_TOKEN)


def alloc(p: KVPool, req: str, max_len: int):
    return p.allocate(req, max_len, 2, 16, "cpu", torch.float32)


def test_cost_is_layers_times_bytes_times_length() -> None:
    assert pool(10**9).cost_bytes(100) == LAYERS * PER_TOKEN * 100


def test_the_whole_charge_is_taken_at_allocation_and_credited_on_release() -> None:
    p = pool(10**9)
    alloc(p, "a", 50)
    assert p.used_bytes == p.cost_bytes(50)
    assert p.free_bytes == 10**9 - p.cost_bytes(50)
    assert p.release("a") == p.cost_bytes(50)
    assert p.used_bytes == 0


def test_used_bytes_sums_live_requests() -> None:
    p = pool(10**9)
    alloc(p, "a", 10)
    alloc(p, "b", 20)
    assert p.used_bytes == p.cost_bytes(30)
    assert len(p) == 2


def test_an_allocation_over_budget_raises_and_charges_nothing() -> None:
    p = pool(1000)
    with pytest.raises(OutOfMemoryOnKV) as err:
        alloc(p, "a", 100)
    assert err.value.req == "a"
    assert p.used_bytes == 0 and p.get("a") is None


def test_a_retried_first_frame_does_not_charge_twice() -> None:
    p = pool(10**9)
    first = alloc(p, "a", 10)
    assert alloc(p, "a", 10) is first
    assert p.used_bytes == p.cost_bytes(10)


def test_release_of_an_unknown_request_is_a_no_op() -> None:
    assert pool(10).release("never") == 0
