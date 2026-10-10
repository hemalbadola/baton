"""Admission arithmetic and request building on the head (PRD 7.3, 7.5)."""

import pytest

from baton.head.driver import Admission, Driver, Rejected, Sampling
from baton.head.planner import Assignment, Plan
from baton.head.registry import Registry


def plan() -> Plan:
    rows = (
        Assignment("a", 0, 10, 0, 9, False, 1.0, 100),
        Assignment("b", 1, 6, 10, 15, True, 1.0, 100),
    )
    return Plan(feasible=True, objective="latency", assignments=rows, kv_fraction=0.2)


def admission() -> Admission:
    # 1 KB per token per layer. Node a: 10 layers, so 10 KB per token.
    return Admission(
        plan(),
        kv_budget_bytes={"a": 100_000, "b": 100_000},
        per_token={"a": 1000, "b": 1000},
    )


def test_cost_is_layers_times_per_token_times_length() -> None:
    assert admission().cost_bytes(5) == {"a": 50_000, "b": 30_000}


def test_a_request_must_fit_on_every_node() -> None:
    adm = admission()
    assert adm.fits(10) and not adm.fits(11)  # node a: 10 layers x 1000 x 10 = 100_000


def test_hold_and_release_move_the_usage() -> None:
    adm = admission()
    adm.hold("r1", 6)
    assert adm.usage() == {"a": (60_000, 100_000), "b": (36_000, 100_000)}
    assert not adm.fits(5)
    adm.release("r1")
    assert adm.fits(10)
    adm.release("r1")  # a second release is a no-op


class Tok:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text)))


def driver(ctx: int = 16) -> Driver:
    return Driver(Registry(), admission(), Tok(), ctx)


async def test_a_prompt_longer_than_the_context_is_a_400() -> None:
    with pytest.raises(Rejected) as err:
        driver(8).build_request(None, "x" * 8, 4, Sampling())
    assert err.value.status == 400


async def test_max_len_is_capped_at_the_context() -> None:
    request = driver(16).build_request(None, "x" * 10, 100, Sampling(), stop=())
    assert request.max_len == 16 and len(request.ids) == 10
