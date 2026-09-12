"""Capability probe and benchmark tests (PRD 6.2, 6.3).

Skeleton for M1. Every test is named for the behaviour it must pin down. The
bodies land with the implementation.
"""

import pytest

pytestmark = pytest.mark.skip(reason="M1 interface only; probe not implemented")


class TestPickDevice:
    def test_auto_prefers_cuda_then_mps_then_cpu(self) -> None:
        """`auto` walks the preference order and stops at the first available."""

    def test_explicit_device_is_returned_unchanged(self) -> None:
        """An explicit `--device cuda` never silently downgrades to cpu."""


class TestMemoryBudget:
    def test_usable_is_free_minus_os_reserve(self) -> None:
        """usable_bytes = mem_free_bytes - os_reserve for the backend."""

    def test_max_mem_caps_free_memory(self) -> None:
        """`--max-mem 4G` on a 64 GB box yields a 4 GB base, so the simulation
        of a small machine is honest."""

    def test_usable_never_goes_negative(self) -> None:
        """Free memory below the OS reserve gives 0, not a negative budget."""

    def test_os_reserve_is_largest_on_mps(self) -> None:
        """Unified memory is shared with the OS, so mps reserves 2 GB."""


class TestCapabilities:
    def test_int4_fast_path_false_when_kernel_missing(self) -> None:
        """A missing `_weight_int4pack_mm` is a capability answer, not a raise."""

    def test_probe_never_raises_on_a_failed_sub_probe(self) -> None:
        """One unavailable field must not lose the whole `hello`."""

    def test_link_unknown_when_platform_tool_is_absent(self) -> None:
        """No `networksetup` / `iw` / `netsh` gives `unknown`."""

    def test_mps_free_memory_is_clamped_by_host_free(self) -> None:
        """`recommended_max_memory` alone over-reports on unified memory."""


class TestBench:
    def test_reports_median_not_mean(self) -> None:
        """One scheduler hiccup in 30 decode steps must not move the number."""

    def test_frees_every_tensor_it_allocated(self) -> None:
        """Device memory after the bench equals memory before it."""

    def test_completes_under_ten_seconds(self) -> None:
        """PRD 6.3 cost budget, on any backend."""
