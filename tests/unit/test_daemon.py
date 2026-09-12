"""Worker daemon startup and reconnect tests (PRD 6.1, 11).

Skeleton for M1. These use a fake head over a loopback socket, so none of them
need a second machine.
"""

import pytest

pytestmark = pytest.mark.skip(reason="M1 interface only; WorkerDaemon not implemented")


class TestStartupOrder:
    def test_data_server_binds_before_hello_is_sent(self) -> None:
        """`hello.data_addr` carries a real port, and an ephemeral port is only
        known after the bind (PRD 6.1 steps 3 and 5)."""

    def test_probe_runs_before_hello(self) -> None:
        """The head plans from `caps`, so `hello` must already carry them."""

    def test_explicit_head_skips_the_mdns_browse(self) -> None:
        """`--head HOST:PORT` must not pay the 10 s browse."""

    def test_discovery_retries_forever_and_prints_once_a_minute(self) -> None:
        """A worker started before the head is the normal case."""

    def test_default_name_is_the_hostname(self) -> None:
        """The reconnect path matches on name, so it must be stable."""


class TestHeartbeat:
    def test_health_is_sent_every_two_seconds(self) -> None:
        """Six seconds of silence marks this worker lost (PRD 8.6)."""

    def test_health_fields_come_from_engine_state(self) -> None:
        """`kv_used`, `queue_depth`, `active_reqs` are read, not measured."""

    def test_health_carries_loaded_rev(self) -> None:
        """A restarted head skips planning when every rev matches (PRD 11.3)."""

    def test_heartbeat_survives_a_long_running_compute_job(self) -> None:
        """The regression test for the thread split. A 10 s bench must not
        stop the heartbeat."""


class TestReconnect:
    def test_control_drop_unloads_nothing(self) -> None:
        """The shard stays resident during the 30 s window (PRD 6.1, 11.3)."""

    def test_reconnect_retries_every_two_seconds_with_the_same_name(self) -> None:
        """Same name, so the head can match the returning worker."""

    def test_same_plan_rev_inside_thirty_seconds_resumes_with_no_reload(self) -> None:
        """This is what makes a Wi-Fi blip cost seconds, not minutes."""

    def test_different_plan_rev_triggers_a_reload(self) -> None:
        """The head re-planned while this worker was away."""

    def test_thirty_seconds_elapsed_unloads_and_returns_to_discovery(self) -> None:
        """Holding a shard for a dead head wastes the machine."""


class TestControlDispatch:
    def test_bench_runs_on_the_compute_thread(self) -> None:
        """Inline it would hold the loop for 10 s and lose the worker."""

    def test_load_frees_a_differing_resident_plan_first(self) -> None:
        """Two plans must never be resident at once (PRD 6.4 step 1)."""

    def test_load_progress_is_reported_every_500_ms(self) -> None:
        """The dashboard bar depends on it (PRD 6.4 step 2)."""

    def test_data_connect_failure_after_thirty_seconds_fails_the_load(self) -> None:
        """The head then re-plans (PRD 6.4 step 4, 8.6)."""

    def test_abort_reaches_the_engine_without_awaiting_it(self) -> None:
        """An abort must not queue behind the request it cancels."""

    def test_unknown_message_type_is_logged_not_fatal(self) -> None:
        """A newer head must not crash an older worker."""


class TestFailureReporting:
    def test_kv_oom_replies_error_with_code_oom(self) -> None:
        """The head answers HTTP 503 (PRD 6.5, 11.1)."""

    def test_peer_data_socket_close_reports_link_down(self) -> None:
        """The worker reports and keeps its shard. The head decides (PRD 11.2)."""

    def test_sleep_inhibitor_is_released_on_stop(self) -> None:
        """A leaked `caffeinate` child keeps the laptop awake forever."""

    def test_unsupported_platform_warns_once_and_continues(self) -> None:
        """No sleep inhibitor is a warning, not a startup failure."""
