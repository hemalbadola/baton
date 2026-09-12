"""Unit tests for the head registry. PRD 7.2 and 7.6."""

from __future__ import annotations

import pytest

from baton.head.registry import (
    HEALTH_TIMEOUT_S,
    OS_RESERVE_BYTES,
    Capabilities,
    Health,
    NameInUse,
    Registry,
    Worker,
)

GB = 1024**3


def caps(usable: int = 8 * GB, t_dec: float = 0.5, t_pre: float = 4.0, **kw) -> Capabilities:
    return Capabilities(
        backend=kw.pop("backend", "cuda"),
        usable_bytes=usable,
        t_dec_ms=t_dec,
        t_pre_ms=t_pre,
        **kw,
    )


class FakeClock:
    """A clock the test moves by hand, so no test ever sleeps."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def registry() -> Registry:
    return Registry(clock=FakeClock())


def join(registry: Registry, name: str, port: int = 9001, **kw) -> Worker:
    return registry.hello(name, ("10.0.0.1", port), caps(**kw), control=object())


# --------------------------------------------------------------------------- #
# Capabilities: the wire is a trust boundary
# --------------------------------------------------------------------------- #


def test_from_caps_uses_the_workers_own_usable_bytes():
    """Only the worker knows its `--max-mem`, so its own figure wins."""
    c = Capabilities.from_caps({"backend": "mps", "mem_free_bytes": 9 * GB, "usable_bytes": 4 * GB})
    assert c.usable_bytes == 4 * GB


def test_from_caps_applies_the_prd_63_reserve_when_usable_is_absent():
    c = Capabilities.from_caps({"backend": "mps", "mem_free_bytes": 9 * GB})
    assert c.usable_bytes == 9 * GB - OS_RESERVE_BYTES["mps"]


def test_from_caps_floors_usable_at_zero():
    """A device with less free memory than the reserve offers nothing, not a negative."""
    c = Capabilities.from_caps({"backend": "cuda", "mem_free_bytes": 100})
    assert c.usable_bytes == 0


def test_from_caps_survives_a_junk_block():
    """A malformed caps block must not raise: the head answers `error`, it does not crash."""
    c = Capabilities.from_caps({"mem_free_bytes": "lots", "bench": "soon", "int4_fast_path": None})
    assert c.usable_bytes == 0
    assert c.t_dec_ms == 0.0
    assert c.backend == "cpu"


def test_from_caps_reads_the_bench_block():
    c = Capabilities.from_caps({"backend": "cpu", "bench": {"t_dec_ms": 1.5, "t_pre_ms": 30.0}})
    assert (c.t_dec_ms, c.t_pre_ms) == (1.5, 30.0)


# --------------------------------------------------------------------------- #
# Data shaping
# --------------------------------------------------------------------------- #


def test_layers_is_zero_before_the_plan_assigns_a_range(registry):
    assert join(registry, "a").layers == 0


def test_layers_counts_both_ends_of_the_range(registry):
    join(registry, "a")
    registry.assign("a", 0, 15, holds_lm_head=False)
    assert registry.get("a").layers == 16


def test_get_returns_none_for_an_unknown_name(registry):
    assert registry.get("ghost") is None


def test_iter_keeps_join_order(registry):
    for name in ("a", "b", "c"):
        join(registry, name)
    assert [w.name for w in registry] == ["a", "b", "c"]
    assert len(registry) == 3


def test_ring_is_layer_order_not_join_order(registry):
    """N1 holds layer 0. The ring is layer order, whatever order the workers joined."""
    for name, first, last in (("c", 20, 31), ("a", 0, 9), ("b", 10, 19)):
        join(registry, name)
        registry.assign(name, first, last, holds_lm_head=(name == "c"))
        registry.mark_loaded(name)
    assert [w.name for w in registry.ring()] == ["a", "b", "c"]


def test_ring_holds_only_loaded_workers(registry):
    join(registry, "a")
    registry.assign("a", 0, 9, holds_lm_head=False)
    registry.mark_loaded("a")
    join(registry, "b")
    registry.assign("b", 10, 19, holds_lm_head=True)  # still loading
    join(registry, "spare")  # standby, no range
    assert [w.name for w in registry.ring()] == ["a"]


def test_roster_returns_objects_so_it_can_back_a_cluster_snapshot(registry):
    """`HeadSeam.snapshot` builds one NodeSnapshot per entry. A string cannot do that."""
    join(registry, "a")
    row = registry.roster()[0]
    assert isinstance(row, Worker)
    assert (row.name, row.state, row.capabilities.backend) == ("a", "standby", "cuda")


def test_roster_table_prints_every_worker(registry):
    join(registry, "a")
    join(registry, "b")
    registry.assign("b", 0, 7, holds_lm_head=True)
    table = registry.roster_table()
    assert "a" in table and "b" in table
    assert "0-7*" in table  # the star marks the lm_head holder


def test_devices_skips_a_worker_that_has_not_benched(registry):
    """A zero cost makes the planner reject the whole fleet, so an unbenched worker is left out."""
    join(registry, "benched")
    join(registry, "fresh", t_dec=0.0, t_pre=0.0)
    assert [row[0] for row in registry.devices()] == ["benched"]


def test_devices_skips_a_lost_worker(registry):
    join(registry, "a")
    join(registry, "b")
    registry.mark_lost("b")
    assert [row[0] for row in registry.devices()] == ["a"]


def test_devices_row_matches_the_planner_device_fields(registry):
    join(registry, "a", usable=7 * GB, t_dec=0.5, t_pre=4.0, int4_fast_path=True)
    assert registry.devices() == [("a", 7 * GB, 0.5, 4.0, True)]


# --------------------------------------------------------------------------- #
# Lifecycle: join, go quiet, be declared lost
# --------------------------------------------------------------------------- #


def test_a_new_worker_lands_in_standby(registry):
    assert join(registry, "a").state == "standby"


def test_hello_rejects_an_empty_name(registry):
    with pytest.raises(ValueError):
        registry.hello("", ("10.0.0.1", 1), caps(), control=None)


def test_a_second_hello_from_a_new_address_is_refused_while_the_name_is_loaded(registry):
    join(registry, "a", port=9001)
    registry.assign("a", 0, 9, holds_lm_head=True)
    registry.mark_loaded("a")
    with pytest.raises(NameInUse):
        registry.hello("a", ("10.0.0.9", 9002), caps(), control=object())


def test_the_same_refusal_applies_while_the_name_is_loading(registry):
    join(registry, "a")
    registry.assign("a", 0, 9, holds_lm_head=True)
    with pytest.raises(NameInUse):
        registry.hello("a", ("10.0.0.9", 9002), caps(), control=object())


def test_a_standby_name_is_free_for_a_different_address(registry):
    """Nothing is loaded, so the name carries no shard and no claim."""
    join(registry, "a", port=9001)
    replaced = registry.hello("a", ("10.0.0.9", 9002), caps(), control=object())
    assert replaced.data_address == ("10.0.0.9", 9002)
    assert len(registry) == 1


def test_a_lost_name_is_free_for_a_different_address(registry):
    join(registry, "a", port=9001)
    registry.mark_lost("a")
    assert registry.hello("a", ("10.0.0.9", 9002), caps(), control=object()).state == "standby"


def test_a_reconnect_from_the_same_address_keeps_the_range_but_not_the_state(registry):
    """PRD 11.3: the head keeps the range so it can skip planning on a matching rev.

    The state still drops, because a live socket proves the process is up and
    proves nothing about the shard.
    """
    join(registry, "a", port=9001)
    registry.assign("a", 4, 11, holds_lm_head=True)
    registry.mark_loaded("a")
    back = registry.hello("a", ("10.0.0.1", 9001), caps(), control=object())
    assert (back.first_layer, back.last_layer, back.holds_lm_head) == (4, 11, True)
    assert back.state == "standby"


def test_a_reconnect_replaces_the_control_writer(registry):
    old_control = object()
    registry.hello("a", ("10.0.0.1", 9001), caps(), control=old_control)
    new_control = object()
    registry.hello("a", ("10.0.0.1", 9001), caps(), control=new_control)
    assert registry.get("a").control is new_control


def test_a_reconnect_keeps_its_slot_in_the_join_order(registry):
    join(registry, "a")
    join(registry, "b")
    registry.hello("a", ("10.0.0.1", 9001), caps(), control=object())
    assert [w.name for w in registry] == ["a", "b"]


def test_on_health_records_the_frame(registry):
    join(registry, "a")
    registry.on_health("a", Health(at=1002.0, free_bytes=5, kv_used_bytes=6, queue_depth=7))
    health = registry.get("a").health
    assert (health.free_bytes, health.kv_used_bytes, health.queue_depth) == (5, 6, 7)


def test_on_health_from_an_unknown_worker_is_dropped(registry):
    registry.on_health("ghost", Health(at=1.0, free_bytes=0, kv_used_bytes=0, queue_depth=0))
    assert len(registry) == 0


def test_health_from_a_lost_worker_does_not_revive_it(registry):
    """Only a fresh `hello` revives a worker, after the head checks `loaded_rev`."""
    join(registry, "a")
    registry.mark_lost("a")
    registry.on_health("a", Health(at=1010.0, free_bytes=0, kv_used_bytes=0, queue_depth=0))
    assert registry.get("a").state == "lost"


def test_health_from_frame_stamps_arrival_not_the_peer_clock(registry):
    """PRD 17.2: absolute clocks are never compared across nodes."""
    health = Health.from_frame(
        {"mem_free": 3, "kv_used": 2, "queue_depth": 1, "loaded_rev": 4}, at=1234.0
    )
    assert health.at == 1234.0
    assert (health.free_bytes, health.kv_used_bytes, health.queue_depth) == (3, 2, 1)
    assert health.loaded_rev == 4


def test_the_timeout_clock_starts_at_hello_not_at_the_first_health(registry):
    """A worker that joins and never heartbeats must still time out."""
    join(registry, "a")  # clock is at 1000.0
    assert registry.timed_out(now=1000.0 + HEALTH_TIMEOUT_S) == []
    assert [w.name for w in registry.timed_out(now=1000.0 + HEALTH_TIMEOUT_S + 0.1)] == ["a"]


def test_a_heartbeat_pushes_the_timeout_out(registry):
    join(registry, "a")
    registry.on_health("a", Health(at=1005.0, free_bytes=0, kv_used_bytes=0, queue_depth=0))
    assert registry.timed_out(now=1010.0) == []
    assert [w.name for w in registry.timed_out(now=1012.0)] == ["a"]


def test_timed_out_ignores_a_worker_already_lost(registry):
    join(registry, "a")
    registry.mark_lost("a")
    assert registry.timed_out(now=99999.0) == []


def test_mark_lost_on_an_unknown_name_returns_none(registry):
    assert registry.mark_lost("ghost") is None


def test_assign_sets_loading_and_the_range(registry):
    join(registry, "a")
    registry.assign("a", 0, 15, holds_lm_head=True)
    w = registry.get("a")
    assert (w.state, w.first_layer, w.last_layer, w.holds_lm_head) == ("loading", 0, 15, True)


def test_assign_rejects_a_backwards_range(registry):
    join(registry, "a")
    with pytest.raises(ValueError):
        registry.assign("a", 9, 4, holds_lm_head=False)


def test_assign_rejects_an_unknown_worker(registry):
    with pytest.raises(KeyError):
        registry.assign("ghost", 0, 1, holds_lm_head=False)


def test_mark_loaded_refuses_a_worker_with_no_range(registry):
    """A `loaded` reply for a shard nobody asked for is a protocol error, not a state."""
    join(registry, "a")
    with pytest.raises(ValueError):
        registry.mark_loaded("a")


def test_release_returns_a_worker_to_standby_with_no_range(registry):
    join(registry, "a")
    registry.assign("a", 0, 9, holds_lm_head=True)
    registry.mark_loaded("a")
    registry.release("a")
    w = registry.get("a")
    assert (w.state, w.first_layer, w.holds_lm_head, w.layers) == ("standby", None, False, 0)


def test_release_does_not_revive_a_lost_worker(registry):
    join(registry, "a")
    registry.assign("a", 0, 9, holds_lm_head=False)
    registry.mark_lost("a")
    registry.release("a")
    assert registry.get("a").state == "lost"


def test_data_addr_is_the_string_a_ring_peer_dials(registry):
    assert join(registry, "a", port=9001).data_addr == "10.0.0.1:9001"


def test_the_full_join_quiet_lost_path(registry):
    """One worker, end to end: join, load, go quiet, be declared lost."""
    clock = registry.clock
    join(registry, "a")
    assert registry.get("a").state == "standby"
    registry.assign("a", 0, 15, holds_lm_head=True)
    registry.mark_loaded("a")
    assert [w.name for w in registry.ring()] == ["a"]

    clock.now += 2.0
    registry.on_health("a", Health(at=clock.now, free_bytes=1, kv_used_bytes=0, queue_depth=0))
    assert registry.timed_out(now=clock.now) == []

    clock.now += HEALTH_TIMEOUT_S + 0.1
    late = registry.timed_out(now=clock.now)
    assert [w.name for w in late] == ["a"]
    registry.mark_lost("a")
    assert registry.ring() == []
    assert registry.roster()[0].state == "lost"
