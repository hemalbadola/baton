"""Partitioner for the head node. PRD section 9.

Pure logic over a fleet description: no hardware, no I/O, no other lane. The
planner turns a list of probed devices plus a model shape into a ring of layer
ranges, or reports why the fleet cannot hold the model.

Cost model (9.2). Every layer on device i costs `c_i` milliseconds of
user-visible time for one request:

    c_i = t_dec_i * G + t_pre_i * (P / 256)

The `latency` objective minimizes the sum of layer costs plus the hop cost of
the ring, because a single user feels the sum of the stages. The `throughput`
objective minimizes the slowest stage instead, because with many concurrent
requests the ring behaves like a pipeline (PRD D7, 10.5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import permutations
from math import ceil

KV_BYTES_PER_TOKEN_PER_LAYER = 4096
PREFILL_CHUNK = 256
MIN_GUARANTEED_CTX = 2048
KV_FRACTION_STEP = 0.05
RING_ENUMERATE_MAX = 8

LATENCY = "latency"
THROUGHPUT = "throughput"
OBJECTIVES = (LATENCY, THROUGHPUT)


class PlannerError(ValueError):
    """The planner inputs are malformed. An infeasible fleet is not an error."""


@dataclass(frozen=True)
class Device:
    """One probed device (PRD 6.4 `bench` reply, PRD 9.1 inputs).

    `t_dec_ms` is milliseconds per layer per decoded token. `t_pre_ms` is
    milliseconds per layer per 256-token prefill chunk. `int4_fast_path` is
    recorded by the probe but the v1 cost function does not use it: the
    measured `t_dec_ms` already contains the effect of the fast path.
    """

    name: str
    usable_bytes: int
    t_dec_ms: float
    t_pre_ms: float
    int4_fast_path: bool = False


@dataclass(frozen=True)
class Model:
    """Model shape at the chosen quantization tier (PRD 9.1)."""

    layers: int
    weight_bytes_per_layer: int
    lm_head_bytes: int
    kv_bytes_per_token_per_layer: int = KV_BYTES_PER_TOKEN_PER_LAYER


@dataclass(frozen=True)
class Workload:
    """Planning knobs. `gen_tokens` is G and `prompt_tokens` is P (PRD 9.1)."""

    objective: str = LATENCY
    gen_tokens: int = 200
    prompt_tokens: int = 500
    ctx: int = 4096
    kv_fraction: float = 0.2


DEFAULT_WORKLOAD = Workload()


@dataclass(frozen=True)
class Fleet:
    """The devices plus the measured round-trip times between them (PRD 9.1).

    `rtt_ms[a][b]` is the median round trip from a to b in milliseconds. A pair
    that the probe did not measure falls back to `default_rtt_ms`.
    """

    devices: tuple[Device, ...]
    rtt_ms: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    default_rtt_ms: float = 3.0

    def hop_ms(self, src: str, dst: str) -> float:
        row = self.rtt_ms.get(src)
        if row is None:
            return self.default_rtt_ms
        value = row.get(dst)
        return self.default_rtt_ms if value is None else float(value)


@dataclass(frozen=True)
class Assignment:
    """One ring member and the contiguous layer range it holds (PRD 9.6)."""

    name: str
    position: int
    layers: int
    first_layer: int
    last_layer: int
    holds_lm_head: bool
    stage_ms: float
    guaranteed_ctx: int

    @property
    def role(self) -> str:
        return f"N{self.position + 1}"


@dataclass(frozen=True)
class Plan:
    """The planner result. An infeasible fleet returns `feasible=False`."""

    feasible: bool
    objective: str
    assignments: tuple[Assignment, ...] = ()
    standby: tuple[str, ...] = ()
    kv_fraction: float = 0.0
    guaranteed_ctx: int = 0
    hop_ms: float = 0.0
    token_ms: float = 0.0
    reason: str = ""
    shortfall_bytes: int = 0

    @property
    def ring(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.assignments)

    @property
    def nk(self) -> str | None:
        for a in self.assignments:
            if a.holds_lm_head:
                return a.name
        return None

    @property
    def tokens_per_second(self) -> float:
        return 1000.0 / self.token_ms if self.token_ms > 0 else 0.0


def plan(fleet: Fleet, model: Model, workload: Workload = DEFAULT_WORKLOAD) -> Plan:
    """Partition `model` over `fleet`. Never raises on an infeasible fleet.

    The planner lowers `kv_fraction` in steps of 0.05 until the fleet fits, and
    refuses to lower it past the point where the guaranteed context of the
    resulting plan drops under 2048 tokens (PRD 9.9).
    """
    _validate(fleet, model, workload)

    lm_head_equiv = ceil(model.lm_head_bytes / model.weight_bytes_per_layer)
    need = model.layers + lm_head_equiv
    costs = _costs(fleet, workload)
    requested = _percent(workload.kv_fraction)
    total_usable = sum(d.usable_bytes for d in fleet.devices)
    caps: dict[str, int] = {}
    percent = requested

    for percent in range(requested, -1, -_percent(KV_FRACTION_STEP)):
        caps = _caps(fleet, model, percent)
        if sum(caps.values()) < need:
            continue
        counts, nk = _assign(fleet, model, workload, caps, costs, lm_head_equiv)
        built = _build(fleet, model, workload, counts, nk, percent)
        if percent == requested or built.guaranteed_ctx >= MIN_GUARANTEED_CTX:
            return built
        return Plan(
            feasible=False,
            objective=workload.objective,
            kv_fraction=percent / 100.0,
            reason=(
                f"Fleet fits the model only at kv_fraction {percent / 100.0:.2f}, which "
                f"guarantees {built.guaranteed_ctx} tokens of context, under the "
                f"{MIN_GUARANTEED_CTX} token floor. Try --quant int4 or add a device."
            ),
        )

    short_layers = need - sum(caps.values())
    shortfall = int(short_layers * model.weight_bytes_per_layer * 100 / max(100 - percent, 1))
    model_bytes = need * model.weight_bytes_per_layer
    return Plan(
        feasible=False,
        objective=workload.objective,
        kv_fraction=percent / 100.0,
        shortfall_bytes=shortfall,
        reason=(
            f"Need {_gb(shortfall)} GB more. Fleet usable: {_gb(total_usable)} GB. "
            f"Model: {_gb(model_bytes)} GB. Try --quant int4 or add a device."
        ),
    )


def plan_manual(
    fleet: Fleet,
    model: Model,
    ranges: Sequence[tuple[str, int, int]],
    workload: Workload = DEFAULT_WORKLOAD,
) -> Plan:
    """Build a plan from a hand-written `--plan` override (PRD 9.8).

    The solver is skipped. Feasibility, ring ordering and role assignment still
    run. `ranges` is a sequence of `(device_name, first_layer, last_layer)`.
    """
    _validate(fleet, model, workload)
    known = {d.name for d in fleet.devices}
    counts: dict[str, int] = {name: 0 for name in known}
    covered: set[int] = set()
    for name, first, last in ranges:
        if name not in known:
            raise PlannerError(f"unknown device in --plan: {name}")
        if first > last:
            raise PlannerError(f"empty range in --plan for {name}: {first}-{last}")
        counts[name] = last - first + 1
        covered |= set(range(first, last + 1))
    if covered != set(range(model.layers)):
        raise PlannerError(f"--plan must cover layers 0-{model.layers - 1} exactly once")

    percent = _percent(workload.kv_fraction)
    caps = _caps(fleet, model, percent)
    lm_head_equiv = ceil(model.lm_head_bytes / model.weight_bytes_per_layer)
    over = [n for n, c in counts.items() if c > caps[n]]
    if over:
        return Plan(
            feasible=False,
            objective=workload.objective,
            kv_fraction=percent / 100.0,
            reason=f"--plan exceeds the memory of: {', '.join(sorted(over))}",
        )
    costs = _costs(fleet, workload)
    # A hand plan fixes every range, so Nk must fit without moving a layer.
    nk = _pick_nk(counts, caps, costs, lm_head_equiv)
    if nk is None:
        return Plan(
            feasible=False,
            objective=workload.objective,
            kv_fraction=percent / 100.0,
            reason="--plan leaves no device with room for the lm_head",
        )
    return _build(fleet, model, workload, counts, nk, percent)


def format_plan(plan_: Plan) -> str:
    """Render the plan table that `baton serve` prints (PRD 9.9)."""
    if not plan_.feasible:
        return f"INFEASIBLE: {plan_.reason}"
    header = (
        f"objective={plan_.objective} kv_fraction={plan_.kv_fraction:.2f} "
        f"guaranteed_ctx={plan_.guaranteed_ctx}"
    )
    lines = [header, f"{'Device':<12}{'Role':<8}{'Layers':<12}{'Stage ms':>10}"]
    for a in plan_.assignments:
        role = f"{a.role}=Nk" if a.holds_lm_head else a.role
        span = f"{a.first_layer}-{a.last_layer}"
        lines.append(f"{a.name:<12}{role:<8}{span:<12}{a.stage_ms:>10.1f}")
    lines.append(
        f"per-token {plan_.token_ms:.1f} ms "
        f"({len(plan_.assignments)} hops {plan_.hop_ms:.1f} ms) "
        f"-> {plan_.tokens_per_second:.2f} tok/s"
    )
    if plan_.standby:
        lines.append(f"standby: {', '.join(plan_.standby)}")
    return "\n".join(lines)


# --- internals ---------------------------------------------------------------


def _validate(fleet: Fleet, model: Model, workload: Workload) -> None:
    if not fleet.devices:
        raise PlannerError("fleet has no devices")
    names = [d.name for d in fleet.devices]
    if len(set(names)) != len(names):
        raise PlannerError("duplicate device names in fleet")
    if model.layers < 1:
        raise PlannerError("model must have at least one layer")
    if model.weight_bytes_per_layer < 1:
        raise PlannerError("weight_bytes_per_layer must be positive")
    if workload.objective not in OBJECTIVES:
        raise PlannerError(f"objective must be one of {OBJECTIVES}")
    if not 0.0 <= workload.kv_fraction < 1.0:
        raise PlannerError("kv_fraction must be in [0, 1)")
    for d in fleet.devices:
        if d.t_dec_ms <= 0 or d.t_pre_ms < 0 or d.usable_bytes < 0:
            raise PlannerError(f"device {d.name} has an impossible probe result")


def _percent(fraction: float) -> int:
    """Resolve a fraction to whole percent so that caps use integer maths."""
    return round(fraction * 100)


def _gb(nbytes: int) -> str:
    return f"{nbytes / 1e9:.1f}"


def _costs(fleet: Fleet, workload: Workload) -> dict[str, float]:
    chunks = workload.prompt_tokens / PREFILL_CHUNK
    return {d.name: d.t_dec_ms * workload.gen_tokens + d.t_pre_ms * chunks for d in fleet.devices}


def _caps(fleet: Fleet, model: Model, percent: int) -> dict[str, int]:
    keep = 100 - percent
    return {
        d.name: (d.usable_bytes * keep) // (100 * model.weight_bytes_per_layer)
        for d in fleet.devices
    }


def _assign(
    fleet: Fleet,
    model: Model,
    workload: Workload,
    caps: dict[str, int],
    costs: dict[str, float],
    lm_head_equiv: int,
) -> tuple[dict[str, int], str]:
    if workload.objective == THROUGHPUT:
        counts = _assign_throughput(fleet, model, caps)
    else:
        counts = _assign_latency(fleet, model, caps, costs)
    counts, nk = _place_lm_head(fleet, counts, caps, costs, lm_head_equiv, workload.objective)
    if workload.objective == LATENCY:
        counts, nk = _refine(fleet, workload, counts, nk, caps, costs, lm_head_equiv)
    return counts, nk


def _assign_latency(
    fleet: Fleet, model: Model, caps: dict[str, int], costs: dict[str, float]
) -> dict[str, int]:
    """Give layers to the cheapest device first (PRD 9.4)."""
    counts = {d.name: 0 for d in fleet.devices}
    remaining = model.layers
    for name in _by_cost(costs):
        take = min(caps[name], remaining)
        counts[name] = take
        remaining -= take
        if remaining == 0:
            break
    return counts


def _assign_throughput(fleet: Fleet, model: Model, caps: dict[str, int]) -> dict[str, int]:
    """Minimize the slowest stage by binary search on the stage time (PRD 9.5)."""
    t_dec = {d.name: d.t_dec_ms for d in fleet.devices}
    candidates = sorted({n * t_dec[name] for name in caps for n in range(1, caps[name] + 1)})

    def fits(stage_ms: float) -> bool:
        return sum(min(caps[n], int(stage_ms / t_dec[n])) for n in caps) >= model.layers

    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if fits(candidates[mid]):
            hi = mid
        else:
            lo = mid + 1
    stage = candidates[lo]
    counts = {n: min(caps[n], int(stage / t_dec[n])) for n in caps}

    surplus = sum(counts.values()) - model.layers
    for name in sorted(counts, key=lambda n: (-t_dec[n], n)):
        if surplus <= 0:
            break
        drop = min(surplus, counts[name])
        counts[name] -= drop
        surplus -= drop
    return counts


def _by_cost(costs: dict[str, float]) -> list[str]:
    return sorted(costs, key=lambda n: (costs[n], n))


def _pick_nk(
    counts: dict[str, int], caps: dict[str, int], costs: dict[str, float], lm_head_equiv: int
) -> str | None:
    """The cheapest ring member that already has room for the lm_head (PRD 9.4)."""
    for name in _by_cost(costs):
        if counts[name] > 0 and caps[name] - counts[name] >= lm_head_equiv:
            return name
    return None


def _try_nk(
    counts: dict[str, int], caps: dict[str, int], nk: str, lm_head_equiv: int, order: list[str]
) -> dict[str, int] | None:
    """Counts with room for the lm_head on `nk`, or None if it cannot be made.

    A full `nk` sheds layers to the next device with room, cheapest first
    (PRD 9.4, and the worked example of 9.9 where pc keeps 19 layers and the 2
    displaced ones go to old).
    """
    if counts[nk] == 0:
        return None
    left = lm_head_equiv - (caps[nk] - counts[nk])
    if left <= 0:
        return dict(counts)
    out = dict(counts)
    movable = out[nk] - 1
    for name in order:
        if name == nk or left == 0:
            continue
        take = min(caps[name] - out[name], left, movable)
        if take > 0:
            out[name] += take
            out[nk] -= take
            left -= take
            movable -= take
    return None if left else out


def _place_lm_head(
    fleet: Fleet,
    counts: dict[str, int],
    caps: dict[str, int],
    costs: dict[str, float],
    lm_head_equiv: int,
    objective: str,
) -> tuple[dict[str, int], str]:
    """Choose Nk and charge the lm_head to it.

    Nk runs `lm_head` on every token, so under `latency` it is the cheapest ring
    member, exactly as PRD 9.4 states. Under `throughput` the cheapest member is
    the wrong choice: making room on a full device pushes its layers onto a
    slower one and raises the slowest stage, which is the only thing that
    objective minimizes. So `throughput` picks the Nk that leaves the slowest
    stage lowest. Both rules agree whenever the cheapest member already has room.

    Total spare capacity is at least `lm_head_equiv` whenever the fleet passed
    the feasibility check, so one candidate always succeeds.
    """
    order = _by_cost(costs)
    if objective == THROUGHPUT:
        t_dec = {d.name: d.t_dec_ms for d in fleet.devices}
        best: tuple[float, str, dict[str, int]] | None = None
        for nk in order:
            candidate = _try_nk(counts, caps, nk, lm_head_equiv, order)
            if candidate is None:
                continue
            stage = max(candidate[n] * t_dec[n] for n in candidate if candidate[n])
            if best is None or (stage, nk) < (best[0], best[1]):
                best = (stage, nk, candidate)
        if best is None:
            raise PlannerError("no device can hold the lm_head")
        return best[2], best[1]

    for nk in order:
        candidate = _try_nk(counts, caps, nk, lm_head_equiv, order)
        if candidate is not None:
            return candidate, nk
    raise PlannerError("no device can hold the lm_head")


def _ring_cost(fleet: Fleet, members: tuple[str, ...], nk: str) -> tuple[float, tuple[str, ...]]:
    """Cheapest closed-ring order with Nk fixed last (PRD 9.6)."""
    if len(members) == 1:
        return 0.0, members
    others = tuple(sorted(n for n in members if n != nk))

    def cost(order: tuple[str, ...]) -> float:
        chain = order + (nk,)
        hops = sum(fleet.hop_ms(chain[j], chain[j + 1]) for j in range(len(chain) - 1))
        return hops + fleet.hop_ms(nk, chain[0])

    if len(members) <= RING_ENUMERATE_MAX:
        best = min(permutations(others), key=lambda o: (cost(o), o))
        return cost(best), best + (nk,)

    # Above 8 members the (k-1)! enumeration stops being cheap. Nearest
    # neighbour from every start is good enough and stays deterministic.
    best_order, best_cost = others, float("inf")
    for start in others:
        order, pool = [start], set(others) - {start}
        while pool:
            nxt = min(pool, key=lambda n: (fleet.hop_ms(order[-1], n), n))
            order.append(nxt)
            pool.remove(nxt)
        candidate = tuple(order)
        c = cost(candidate)
        if c < best_cost:
            best_order, best_cost = candidate, c
    return best_cost, best_order + (nk,)


def _total_cost(
    fleet: Fleet,
    workload: Workload,
    counts: dict[str, int],
    nk: str,
    costs: dict[str, float],
    cache: dict[tuple[frozenset[str], str], float],
) -> float:
    """User-visible milliseconds for one request: compute plus every hop.

    Each generated token and each prefill chunk travels the ring once, so the
    hop cost of a ring member is paid `G + ceil(P / 256)` times. This is the
    term that lets the local search drop a device whose last layers cost more
    than its hop is worth (PRD 9.4).
    """
    members = frozenset(n for n, c in counts.items() if c > 0)
    key = (members, nk)
    if key not in cache:
        cache[key] = _ring_cost(fleet, tuple(sorted(members)), nk)[0]
    passes = workload.gen_tokens + ceil(workload.prompt_tokens / PREFILL_CHUNK)
    compute = sum(counts[n] * costs[n] for n in counts)
    return compute + cache[key] * passes


def _refine(
    fleet: Fleet,
    workload: Workload,
    counts: dict[str, int],
    nk: str,
    caps: dict[str, int],
    costs: dict[str, float],
    lm_head_equiv: int,
) -> tuple[dict[str, int], str]:
    """Move one layer between devices while the total cost drops (PRD 9.4)."""
    cache: dict[tuple[frozenset[str], str], float] = {}
    best = _total_cost(fleet, workload, counts, nk, costs, cache)
    names = sorted(counts)
    improved = True
    while improved:
        improved = False
        for x in names:
            for y in names:
                if x == y or counts[x] == 0:
                    continue
                if x == nk and counts[x] == 1:
                    continue
                charge = lm_head_equiv if y == nk else 0
                if counts[y] + 1 + charge > caps[y]:
                    continue
                counts[x] -= 1
                counts[y] += 1
                candidate = _total_cost(fleet, workload, counts, nk, costs, cache)
                if candidate < best - 1e-9:
                    best = candidate
                    improved = True
                else:
                    counts[x] += 1
                    counts[y] -= 1
    return counts, nk


def _build(
    fleet: Fleet,
    model: Model,
    workload: Workload,
    counts: dict[str, int],
    nk: str,
    percent: int,
) -> Plan:
    by_name = {d.name: d for d in fleet.devices}
    members = tuple(sorted(n for n, c in counts.items() if c > 0))
    hop_ms, order = _ring_cost(fleet, members, nk)

    assignments: list[Assignment] = []
    first = 0
    for position, name in enumerate(order):
        dev = by_name[name]
        layers = counts[name]
        kv_per_token = layers * model.kv_bytes_per_token_per_layer
        assignments.append(
            Assignment(
                name=name,
                position=position,
                layers=layers,
                first_layer=first,
                last_layer=first + layers - 1,
                holds_lm_head=(name == nk),
                stage_ms=layers * dev.t_dec_ms,
                guaranteed_ctx=(dev.usable_bytes * percent) // (100 * kv_per_token),
            )
        )
        first += layers

    token_ms = sum(a.stage_ms for a in assignments) + hop_ms
    return Plan(
        feasible=True,
        objective=workload.objective,
        assignments=tuple(assignments),
        standby=tuple(sorted(n for n, c in counts.items() if c == 0)),
        kv_fraction=percent / 100.0,
        guaranteed_ctx=min(a.guaranteed_ctx for a in assignments),
        hop_ms=hop_ms,
        token_ms=token_ms,
        reason="",
    )
