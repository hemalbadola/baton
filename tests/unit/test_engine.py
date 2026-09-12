"""Forward engine and compute thread tests (PRD 6.6).

Skeleton for M1. The engine tests use a fake `ShardRunner`, so none of them
need a GPU or a real model.
"""

import pytest

pytestmark = pytest.mark.skip(reason="M1 interface only; ForwardEngine not implemented")


class TestThreadSeparation:
    def test_control_loop_never_blocks_on_a_running_job(self) -> None:
        """`submit` returns while the compute thread is mid-frame. This is the
        property the 2 s heartbeat depends on."""

    def test_exactly_one_compute_thread_runs(self) -> None:
        """One thread means no lock protects the model itself."""

    def test_start_is_idempotent(self) -> None:
        """A second `start` must not create a second compute thread."""

    def test_outbound_frames_cross_threads_through_the_loop(self) -> None:
        """Frames reach asyncio through `call_soon_threadsafe`, never by a
        direct `asyncio.Queue.put_nowait` from the compute thread."""

    def test_an_exception_in_a_job_does_not_kill_the_thread(self) -> None:
        """One bad request must not stop every other request on this worker."""

    def test_stop_wakes_the_thread_from_a_blocking_get(self) -> None:
        """The sentinel, not a poll timeout, ends the loop."""


class TestJobOrder:
    def test_jobs_run_in_arrival_order(self) -> None:
        """One frame at a time, first in first out (PRD 6.6)."""

    def test_queue_depth_reports_waiting_jobs(self) -> None:
        """The head reads this for admission control (PRD 7.5)."""


class TestPromptJob:
    def test_prompt_splits_into_chunks_of_256(self) -> None:
        """Chunk width is a constant by decision D10."""

    def test_a_short_prompt_emits_one_act_frame(self) -> None:
        """Fewer than 256 ids must not pad to a full chunk."""

    def test_prompt_is_rejected_off_n1(self) -> None:
        """Only the worker holding layer 0 embeds (PRD 4.3, D5)."""

    def test_kv_is_allocated_on_the_first_frame_only(self) -> None:
        """Later chunks of the same prompt reuse the entry."""


class TestActJob:
    def test_payload_is_viewed_in_compute_dtype(self) -> None:
        """The wire dtype is bf16; a fp16 node casts on send, not on receive
        interpretation (PRD 8.4)."""

    def test_middle_node_emits_act_to_the_next_node(self) -> None:
        """Ni forwards activations to Ni+1."""

    def test_last_node_samples_and_emits_token_and_next(self) -> None:
        """Nk sends `token` to the head and `next` to N1 (D6)."""

    def test_last_node_runs_the_head_on_the_last_row_only(self) -> None:
        """A 256-row prefill chunk needs one logits row, not 256."""


class TestReleaseAndAbort:
    def test_release_frees_kv_then_forwards_the_frame(self) -> None:
        """Free first: the next node may be waiting on this memory."""

    def test_release_stops_when_it_returns_to_its_originator(self) -> None:
        """Nk recognises its own `release` and does not send it round twice
        (PRD 6.6, 8.3)."""

    def test_abort_stops_at_the_last_node(self) -> None:
        """`abort` starts at the head, so its stop rule is 'I am Nk', not
        'this frame is mine'. See the note to god on PRD 6.6 versus 8.3."""


class TestTrace:
    def test_receive_appends_a_node_entry(self) -> None:
        """`{node, t_recv}` per hop (PRD 6.6)."""

    def test_send_adds_t_send_to_the_same_entry(self) -> None:
        """One entry per node, two timestamps."""

    def test_timestamps_are_monotonic_within_a_node(self) -> None:
        """`perf_counter_ns`, never a wall clock: clocks are not compared
        across nodes (PRD 17.2)."""
