"""Planner tests. PRD 9, and the M2 exit test in PRD 19.

Two layers of checks:

1. The worked example of PRD 9.9, number for number.
2. Twenty synthetic fleets, feasible and infeasible, checked against
   invariants that must hold for every plan, and against a brute-force
   optimum where the fleet is small enough to enumerate.
"""

from __future__ import annotations

import random
from itertools import pairwise, permutations

import pytest

from baton.head import planner as P

GB = 1_000_000_000
MB = 1_000_000

# PRD 9.9: 70B at int4.
W70B = 455 * MB
LM_HEAD_70B = 525 * MB
MODEL_70B = P.Model(layers=80, weight_bytes_per_layer=W70B, lm_head_bytes=LM_HEAD_70B)

WORKED_DEVICES = (
    P.Device("pc", 11 * GB, t_dec_ms=1.5, t_pre_ms=25),
    P.Device("mac", 13 * GB, t_dec_ms=3.0, t_pre_ms=120),
    P.Device("desk", 13 * GB, t_dec_ms=12.0, t_pre_ms=900),
    P.Device("old", 9 * GB, t_dec_ms=15.0, t_pre_ms=1200),
)

# mac and old are on Wi-Fi, pc and desk are wired. A Wi-Fi to Wi-Fi hop pays
# two airtime transmissions, so it is the expensive one (PRD 9.6).
WIFI = {"mac", "old"}


def _rtt(names: list[str]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for a in names:
        matrix[a] = {}
        for b in names:
            if a == b:
                matrix[a][b] = 0.0
            elif a in WIFI and b in WIFI:
                matrix[a][b] = 12.0
            elif a in WIFI or b in WIFI:
                matrix[a][b] = 6.0
            else:
                matrix[a][b] = 1.0
    return matrix


def counts_of(plan: P.Plan) -> dict[str, int]:
    return {a.name: a.layers for a in plan.assignments}


def _ring_ms(fleet: P.Fleet, ring: tuple[str, ...]) -> float:
    """A one-member ring costs nothing: N1 and Nk are one process (PRD 10.2)."""
    if len(ring) == 1:
        return 0.0
    hops = sum(fleet.hop_ms(ring[j], ring[j + 1]) for j in range(len(ring) - 1))
    return hops + fleet.hop_ms(ring[-1], ring[0])


# --- PRD 9.9 worked example ---------------------------------------------------


def test_caps_match_the_worked_example():
    fleet = P.Fleet(WORKED_DEVICES)
    assert P._caps(fleet, MODEL_70B, 20) == {"pc": 19, "mac": 22, "desk": 22, "old": 15}
    assert P._caps(fleet, MODEL_70B, 10) == {"pc": 21, "mac": 25, "desk": 25, "old": 17}


def test_costs_match_the_worked_example():
    fleet = P.Fleet(WORKED_DEVICES)
    costs = P._costs(fleet, P.Workload())
    assert round(costs["pc"]) == 349
    assert round(costs["mac"]) == 834
    assert round(costs["desk"]) == 4158
    assert round(costs["old"]) == 5344


def test_worked_example_is_infeasible_at_kv_fraction_020_alone():
    """80 layers plus 2 lm_head equivalents need 82. The caps sum to 78."""
    fleet = P.Fleet(WORKED_DEVICES)
    caps = P._caps(fleet, MODEL_70B, 20)
    equiv = 2
    assert sum(caps.values()) == 78 < MODEL_70B.layers + equiv


def test_worked_example_plan_at_kv_fraction_010():
    """The PRD table: pc 19 plus the head, mac 25, desk 25, old 11."""
    fleet = P.Fleet(WORKED_DEVICES, rtt_ms=_rtt([d.name for d in WORKED_DEVICES]))
    plan = P.plan(fleet, MODEL_70B, P.Workload(kv_fraction=0.10))

    assert plan.feasible
    assert counts_of(plan) == {"pc": 19, "mac": 25, "desk": 25, "old": 11}
    assert plan.nk == "pc"
    assert plan.ring == ("mac", "desk", "old", "pc")
    stages = {a.name: a.stage_ms for a in plan.assignments}
    assert stages == {"mac": 75.0, "desk": 300.0, "old": 165.0, "pc": 28.5}
    ranges = {a.name: (a.first_layer, a.last_layer) for a in plan.assignments}
    assert ranges == {"mac": (0, 24), "desk": (25, 49), "old": (50, 60), "pc": (61, 79)}
    assert plan.guaranteed_ctx == pytest.approx(12695, abs=1)


def test_worked_example_token_ms_with_a_flat_3ms_matrix():
    """PRD: 568.5 ms of compute plus 4 hops of 3 ms = 580.5 ms, 1.7 tok/s."""
    fleet = P.Fleet(WORKED_DEVICES, default_rtt_ms=3.0)
    plan = P.plan(fleet, MODEL_70B, P.Workload(kv_fraction=0.10))
    assert plan.token_ms == pytest.approx(580.5)
    assert plan.tokens_per_second == pytest.approx(1.72, abs=0.01)


def test_relaxation_ladder_lowers_kv_fraction_until_the_fleet_fits():
    """PRD 9.9: 0.20 does not fit. The planner steps down by 0.05."""
    fleet = P.Fleet(WORKED_DEVICES, default_rtt_ms=3.0)
    plan = P.plan(fleet, MODEL_70B, P.Workload(kv_fraction=0.20))
    assert plan.feasible
    # 0.15 is the first rung that fits: caps 20+24+24+16 = 84 >= 82.
    assert plan.kv_fraction == pytest.approx(0.15)
    assert sum(counts_of(plan).values()) == MODEL_70B.layers


def test_wifi_devices_land_between_wired_ones():
    """PRD 9.6 heuristic outcome, checked against every ring order."""
    fleet = P.Fleet(WORKED_DEVICES, rtt_ms=_rtt([d.name for d in WORKED_DEVICES]))
    plan = P.plan(fleet, MODEL_70B, P.Workload(kv_fraction=0.10))
    assert plan.ring == ("mac", "desk", "old", "pc")
    assert _ring_ms(fleet, plan.ring) == min(
        _ring_ms(fleet, order + ("pc",)) for order in permutations(("mac", "desk", "old"))
    )


# --- twenty synthetic fleets --------------------------------------------------


def _synthetic_fleets() -> list[tuple[str, P.Fleet, P.Model, P.Workload, bool]]:
    """Twenty named cases. The last element says whether a plan must exist."""
    cases: list[tuple[str, P.Fleet, P.Model, P.Workload, bool]] = []
    small = P.Model(layers=32, weight_bytes_per_layer=200 * MB, lm_head_bytes=300 * MB)

    def fleet(*devs: P.Device, **kw) -> P.Fleet:
        return P.Fleet(devs, **kw)

    cases.append(
        ("single_device_fits", fleet(P.Device("solo", 16 * GB, 2.0, 40)), small, P.Workload(), True)
    )
    cases.append(
        (
            "single_device_too_small",
            fleet(P.Device("solo", 2 * GB, 2.0, 40)),
            small,
            P.Workload(),
            False,
        )
    )
    cases.append(
        (
            "two_even_devices",
            fleet(P.Device("a", 5 * GB, 2.0, 40), P.Device("b", 5 * GB, 2.0, 40)),
            small,
            P.Workload(),
            True,
        )
    )
    cases.append(
        (
            "fast_gpu_plus_slow_cpu",
            fleet(P.Device("gpu", 8 * GB, 0.8, 12), P.Device("cpu", 8 * GB, 20.0, 1500)),
            small,
            P.Workload(),
            True,
        )
    )
    cases.append(
        (
            "gpu_holds_everything_cpu_stands_by",
            fleet(P.Device("gpu", 24 * GB, 0.8, 12), P.Device("cpu", 8 * GB, 20.0, 1500)),
            small,
            P.Workload(),
            True,
        )
    )
    cases.append(
        (
            "four_mixed_devices",
            fleet(
                P.Device("m1", 6 * GB, 3.0, 120),
                P.Device("m2", 6 * GB, 4.0, 200),
                P.Device("m3", 4 * GB, 9.0, 700),
                P.Device("m4", 4 * GB, 11.0, 900),
            ),
            small,
            P.Workload(),
            True,
        )
    )
    cases.append(
        (
            "four_mixed_throughput",
            fleet(
                P.Device("m1", 6 * GB, 3.0, 120),
                P.Device("m2", 6 * GB, 4.0, 200),
                P.Device("m3", 4 * GB, 9.0, 700),
                P.Device("m4", 4 * GB, 11.0, 900),
            ),
            small,
            P.Workload(objective=P.THROUGHPUT),
            True,
        )
    )
    cases.append(
        (
            "70b_worked_example",
            P.Fleet(WORKED_DEVICES, rtt_ms=_rtt([d.name for d in WORKED_DEVICES])),
            MODEL_70B,
            P.Workload(),
            True,
        )
    )
    cases.append(
        (
            "70b_worked_example_throughput",
            P.Fleet(WORKED_DEVICES, rtt_ms=_rtt([d.name for d in WORKED_DEVICES])),
            MODEL_70B,
            P.Workload(objective=P.THROUGHPUT),
            True,
        )
    )
    cases.append(
        (
            "70b_on_two_laptops_infeasible",
            fleet(P.Device("mac", 13 * GB, 3.0, 120), P.Device("pc", 11 * GB, 1.5, 25)),
            MODEL_70B,
            P.Workload(),
            False,
        )
    )
    cases.append(
        (
            "exactly_at_capacity",
            fleet(P.Device("a", 8_200 * MB, 2.0, 40), P.Device("b", 8_200 * MB, 3.0, 60)),
            small,
            P.Workload(kv_fraction=0.0),
            True,
        )
    )
    cases.append(
        (
            "one_layer_short",
            fleet(P.Device("a", 3 * GB, 2.0, 40), P.Device("b", 3 * GB, 3.0, 60)),
            small,
            P.Workload(kv_fraction=0.0),
            False,
        )
    )
    cases.append(
        (
            "eight_tiny_devices",
            fleet(*(P.Device(f"d{i}", 2 * GB, 2.0 + i, 40 * (i + 1)) for i in range(8))),
            small,
            P.Workload(),
            True,
        )
    )
    cases.append(
        (
            "nine_devices_beyond_enumeration",
            fleet(*(P.Device(f"d{i}", 2 * GB, 2.0 + i, 40 * (i + 1)) for i in range(9))),
            small,
            P.Workload(),
            True,
        )
    )
    cases.append(
        (
            "one_huge_device_many_idle",
            fleet(
                P.Device("big", 40 * GB, 1.0, 20),
                *(P.Device(f"idle{i}", 1 * GB, 30.0, 2000) for i in range(4)),
            ),
            small,
            P.Workload(),
            True,
        )
    )
    cases.append(
        (
            "long_prompt_workload",
            fleet(
                P.Device("gpu", 8 * GB, 0.8, 12),
                P.Device("mac", 8 * GB, 3.0, 120),
                P.Device("cpu", 8 * GB, 20.0, 1500),
            ),
            small,
            P.Workload(prompt_tokens=8000, gen_tokens=50),
            True,
        )
    )
    cases.append(
        (
            "long_generation_workload",
            fleet(
                P.Device("gpu", 8 * GB, 0.8, 12),
                P.Device("mac", 8 * GB, 3.0, 120),
                P.Device("cpu", 8 * GB, 20.0, 1500),
            ),
            small,
            P.Workload(prompt_tokens=50, gen_tokens=2000),
            True,
        )
    )
    cases.append(
        (
            "kv_fraction_floor_blocks_the_plan",
            fleet(
                P.Device("a", 3_400 * MB, 2.0, 40),
                P.Device("b", 3_400 * MB, 3.0, 60),
                P.Device("c", 3_400 * MB, 4.0, 80),
            ),
            P.Model(layers=48, weight_bytes_per_layer=200 * MB, lm_head_bytes=300 * MB),
            P.Workload(kv_fraction=0.20),
            False,
        )
    )
    cases.append(
        (
            "heavy_lm_head",
            fleet(
                P.Device("a", 8 * GB, 2.0, 40),
                P.Device("b", 8 * GB, 3.0, 60),
            ),
            P.Model(layers=32, weight_bytes_per_layer=200 * MB, lm_head_bytes=2_000 * MB),
            P.Workload(),
            True,
        )
    )

    rng = random.Random(9)
    names = [f"r{i}" for i in range(5)]
    devices = tuple(
        P.Device(
            n,
            rng.randrange(4, 20) * GB,
            round(rng.uniform(0.8, 20.0), 2),
            round(rng.uniform(10.0, 1500.0), 1),
        )
        for n in names
    )
    rtt = {
        a: {b: 0.0 if a == b else round(rng.uniform(0.5, 15.0), 2) for b in names} for a in names
    }
    cases.append(
        ("random_five_device_fleet", P.Fleet(devices, rtt_ms=rtt), small, P.Workload(), True)
    )

    assert len(cases) == 20
    return cases


SYNTHETIC = _synthetic_fleets()
IDS = [c[0] for c in SYNTHETIC]


@pytest.mark.parametrize("name,fleet,model,workload,feasible", SYNTHETIC, ids=IDS)
def test_synthetic_fleet_invariants(name, fleet, model, workload, feasible):
    plan = P.plan(fleet, model, workload)
    assert plan.feasible is feasible, f"{name}: {plan.reason}"
    if not plan.feasible:
        assert plan.reason
        assert not plan.assignments
        return

    equiv = -(-model.lm_head_bytes // model.weight_bytes_per_layer)
    caps = P._caps(fleet, model, P._percent(plan.kv_fraction))
    counts = counts_of(plan)

    assert sum(counts.values()) == model.layers
    assert all(c > 0 for c in counts.values()), "a ring member with no layers"
    assert set(counts) | set(plan.standby) == {d.name for d in fleet.devices}
    assert not set(counts) & set(plan.standby)

    for a in plan.assignments:
        charge = equiv if a.holds_lm_head else 0
        assert a.layers + charge <= caps[a.name], f"{name}: {a.name} over capacity"
        assert a.last_layer - a.first_layer + 1 == a.layers

    spans = [(a.first_layer, a.last_layer) for a in plan.assignments]
    assert spans[0][0] == 0
    assert spans[-1][1] == model.layers - 1
    for left, right in pairwise(spans):
        assert right[0] == left[1] + 1, f"{name}: layer ranges are not contiguous"

    assert sum(a.holds_lm_head for a in plan.assignments) == 1
    assert plan.token_ms > 0
    assert plan.hop_ms == pytest.approx(_ring_ms(fleet, plan.ring))
    assert plan.guaranteed_ctx == min(a.guaranteed_ctx for a in plan.assignments)


@pytest.mark.parametrize("name,fleet,model,workload,feasible", SYNTHETIC, ids=IDS)
def test_ring_order_is_optimal(name, fleet, model, workload, feasible):
    plan = P.plan(fleet, model, workload)
    if not plan.feasible or len(plan.ring) > P.RING_ENUMERATE_MAX:
        pytest.skip("infeasible, or too many members to enumerate")
    others = tuple(n for n in plan.ring if n != plan.nk)
    if not others:
        return
    best = min(_ring_ms(fleet, order + (plan.nk,)) for order in permutations(others))
    assert plan.hop_ms == pytest.approx(best)


# --- objectives ---------------------------------------------------------------


def test_throughput_minimizes_the_slowest_stage():
    devices = (
        P.Device("gpu", 8 * GB, 1.0, 20),
        P.Device("mac", 8 * GB, 4.0, 200),
        P.Device("cpu", 8 * GB, 16.0, 1200),
    )
    fleet = P.Fleet(devices, default_rtt_ms=3.0)
    model = P.Model(layers=32, weight_bytes_per_layer=200 * MB, lm_head_bytes=300 * MB)

    fast = P.plan(fleet, model, P.Workload(objective=P.THROUGHPUT))
    low = P.plan(fleet, model, P.Workload(objective=P.LATENCY))
    assert max(a.stage_ms for a in fast.assignments) < max(a.stage_ms for a in low.assignments)


def _splits(total: int, caps: list[int]):
    if not caps:
        if total == 0:
            yield ()
        return
    head, rest = caps[0], caps[1:]
    for n in range(min(head, total) + 1):
        for tail in _splits(total - n, rest):
            yield (n, *tail)


def test_throughput_matches_a_brute_force_optimum():
    devices = (
        P.Device("a", 4 * GB, 1.0, 20),
        P.Device("b", 4 * GB, 3.0, 90),
        P.Device("c", 4 * GB, 7.0, 400),
    )
    fleet = P.Fleet(devices, default_rtt_ms=3.0)
    model = P.Model(layers=24, weight_bytes_per_layer=400 * MB, lm_head_bytes=400 * MB)
    plan = P.plan(fleet, model, P.Workload(objective=P.THROUGHPUT))

    caps = P._caps(fleet, model, P._percent(plan.kv_fraction))
    t_dec = {d.name: d.t_dec_ms for d in devices}
    best = min(
        max(n * t_dec[name] for name, n in zip(caps, combo) if n)
        for combo in _splits(model.layers, [caps[n] for n in caps])
    )
    assert max(a.stage_ms for a in plan.assignments) == pytest.approx(best)


def test_latency_keeps_layers_on_the_cheapest_device():
    devices = (
        P.Device("gpu", 20 * GB, 1.0, 20),
        P.Device("cpu", 20 * GB, 30.0, 2500),
    )
    fleet = P.Fleet(devices, default_rtt_ms=3.0)
    model = P.Model(layers=32, weight_bytes_per_layer=200 * MB, lm_head_bytes=300 * MB)
    plan = P.plan(fleet, model, P.Workload())
    assert plan.ring == ("gpu",)
    assert plan.standby == ("cpu",)


def test_local_search_drops_a_member_whose_hop_costs_more_than_its_layers():
    """A device that can hold one layer is not worth a ring hop (PRD 9.4)."""
    w = 1 * GB
    devices = (
        P.Device("gpu", 40 * GB, 1.0, 10),
        P.Device("scrap", 2 * GB, 1.05, 11),
    )
    fleet = P.Fleet(devices, default_rtt_ms=400.0)
    model = P.Model(layers=8, weight_bytes_per_layer=w, lm_head_bytes=w)
    plan = P.plan(fleet, model, P.Workload(kv_fraction=0.0))
    assert plan.ring == ("gpu",)
    assert plan.standby == ("scrap",)


# --- infeasible reporting -----------------------------------------------------


def test_infeasible_plan_names_the_shortfall():
    fleet = P.Fleet((P.Device("mac", 13 * GB, 3.0, 120), P.Device("pc", 11 * GB, 1.5, 25)))
    plan = P.plan(fleet, MODEL_70B)
    assert not plan.feasible
    assert plan.shortfall_bytes > 0
    assert "Need" in plan.reason and "add a device" in plan.reason
    assert P.format_plan(plan).startswith("INFEASIBLE:")


def test_context_floor_stops_the_relaxation_ladder():
    fleet = P.Fleet(
        (
            P.Device("a", 3_400 * MB, 2.0, 40),
            P.Device("b", 3_400 * MB, 3.0, 60),
            P.Device("c", 3_400 * MB, 4.0, 80),
        )
    )
    model = P.Model(layers=48, weight_bytes_per_layer=200 * MB, lm_head_bytes=300 * MB)
    plan = P.plan(fleet, model, P.Workload(kv_fraction=0.20))
    assert not plan.feasible
    assert "token floor" in plan.reason


# --- input validation and the manual override ---------------------------------


@pytest.mark.parametrize(
    "fleet,model,workload",
    [
        (P.Fleet(()), MODEL_70B, P.Workload()),
        (
            P.Fleet((P.Device("a", GB, 1.0, 1), P.Device("a", GB, 1.0, 1))),
            MODEL_70B,
            P.Workload(),
        ),
        (P.Fleet((P.Device("a", GB, 1.0, 1),)), MODEL_70B, P.Workload(objective="cheapest")),
        (P.Fleet((P.Device("a", GB, 1.0, 1),)), MODEL_70B, P.Workload(kv_fraction=1.0)),
        (P.Fleet((P.Device("a", GB, 0.0, 1),)), MODEL_70B, P.Workload()),
    ],
)
def test_bad_input_raises(fleet, model, workload):
    with pytest.raises(P.PlannerError):
        P.plan(fleet, model, workload)


def test_manual_override_keeps_ranges_and_orders_the_ring():
    fleet = P.Fleet(WORKED_DEVICES, rtt_ms=_rtt([d.name for d in WORKED_DEVICES]))
    plan = P.plan_manual(
        fleet,
        MODEL_70B,
        [("mac", 0, 23), ("desk", 24, 45), ("old", 46, 62), ("pc", 63, 79)],
        P.Workload(kv_fraction=0.10),
    )
    assert plan.feasible
    assert counts_of(plan) == {"mac": 24, "desk": 22, "old": 17, "pc": 17}
    assert plan.nk == "pc"


def test_manual_override_rejects_a_range_over_a_device_capacity():
    fleet = P.Fleet(WORKED_DEVICES)
    plan = P.plan_manual(
        fleet,
        MODEL_70B,
        [("mac", 0, 23), ("desk", 24, 41), ("old", 42, 59), ("pc", 60, 79)],
        P.Workload(kv_fraction=0.10),
    )
    assert not plan.feasible
    assert "old" in plan.reason


def test_manual_override_rejects_a_gap():
    fleet = P.Fleet(WORKED_DEVICES)
    with pytest.raises(P.PlannerError):
        P.plan_manual(fleet, MODEL_70B, [("mac", 0, 23), ("pc", 30, 79)])


def test_format_plan_prints_one_row_per_member():
    fleet = P.Fleet(WORKED_DEVICES, default_rtt_ms=3.0)
    text = P.format_plan(P.plan(fleet, MODEL_70B, P.Workload(kv_fraction=0.10)))
    assert "N1" in text and "N4=Nk" in text
    assert text.count("\n") == 2 + len(WORKED_DEVICES)
