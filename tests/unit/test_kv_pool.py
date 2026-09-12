"""KV cache accounting tests (PRD 6.5).

Skeleton for M1. These are the tests that protect the memory charge, which is
the one place a worker can silently over-commit and OOM mid-generation.
"""

import pytest

pytestmark = pytest.mark.skip(reason="M1 interface only; KVPool not implemented")


class TestMaxLen:
    def test_max_len_is_prompt_plus_max_tokens(self) -> None:
        """The tensor covers the whole generation, allocated once."""

    def test_max_len_is_capped_at_ctx_max(self) -> None:
        """A request asking beyond the loaded context is clamped, not refused
        here: the head owns refusal."""


class TestCharging:
    def test_cost_matches_the_prd_worked_example(self) -> None:
        """24 local layers at max_len 4096 charges 403 MB (PRD 6.5)."""

    def test_charge_is_taken_in_full_at_allocation(self) -> None:
        """Charging as the sequence grows would let two admitted requests both
        fit at admission and both fail at token 2000."""

    def test_release_credits_the_exact_charge(self) -> None:
        """used_bytes returns to its prior value, so no charge leaks."""

    def test_used_bytes_is_the_sum_over_live_requests(self) -> None:
        """This is the `kv_used` field of health (PRD 8.3)."""

    def test_free_bytes_ignores_the_allocator_pool(self) -> None:
        """Freed device blocks stay in the caching allocator, so a device query
        would under-report free budget (PRD 6.5)."""


class TestOom:
    def test_allocation_over_budget_raises_out_of_memory_on_kv(self) -> None:
        """The daemon turns this into `error{req, code="oom"}`."""

    def test_failed_allocation_charges_nothing(self) -> None:
        """A rejected request must not leave a phantom charge behind."""

    def test_device_oom_is_wrapped_not_propagated(self) -> None:
        """A backend OOM and a budget OOM reach the head as the same code."""


class TestLifecycle:
    def test_allocate_is_idempotent_for_a_known_request(self) -> None:
        """A retried first frame must not double charge."""

    def test_release_of_an_unknown_request_is_a_no_op(self) -> None:
        """`release` rings the whole cluster; a second visit is normal."""

    def test_abort_frees_the_tensor_like_release(self) -> None:
        """Both paths must reach zero charge for the request."""
