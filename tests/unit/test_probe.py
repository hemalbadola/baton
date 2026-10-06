"""Capability probe and benchmark tests (PRD 6.2, 6.3)."""

import gc
import time
from types import SimpleNamespace

import psutil
import pytest
import torch

from baton.model import quant
from baton.model.spec import ModelSpec
from baton.worker import probe

GB = 1024**3

TINY = ModelSpec.from_config(
    {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 512,
    }
)


class TestPickDevice:
    def test_auto_prefers_cuda_then_mps_then_cpu(self, monkeypatch) -> None:
        """`auto` walks the preference order and stops at the first available."""
        have = {"cuda": False, "mps": False}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: have["cuda"])
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: have["mps"])
        assert probe.pick_device() == "cpu"
        have["mps"] = True
        assert probe.pick_device() == "mps"
        have["cuda"] = True
        assert probe.pick_device() == "cuda"

    def test_explicit_device_is_returned_unchanged(self, monkeypatch) -> None:
        """An explicit `--device cuda` never silently downgrades to cpu."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert probe.pick_device("cuda") == "cuda"


class TestMemoryBudget:
    @pytest.fixture(autouse=True)
    def _machine(self, monkeypatch) -> None:
        self.total, self.free = 64 * GB, 60 * GB
        monkeypatch.setattr(probe, "memory_total_free", lambda backend: (self.total, self.free))

    def test_usable_is_free_memory_up_to_total_minus_reserve(self) -> None:
        """Free memory already leaves out the OS, so the reserve caps the total."""
        assert probe.memory_budget("cpu").usable_bytes == 60 * GB
        self.free = 64 * GB
        assert probe.memory_budget("cpu").usable_bytes == 64 * GB - 3 * GB // 2

    def test_an_8gb_mac_with_little_free_memory_still_has_a_budget(self) -> None:
        """BAT-23. The measured case: 5.33 GB recommended, 1.3 GB free. It was 0."""
        self.total, self.free = int(5.33 * GB), int(1.3 * GB)
        assert probe.memory_budget("mps").usable_bytes == int(1.3 * GB)
        assert probe.memory_budget("mps", 3 * GB).usable_bytes == 3 * GB

    def test_max_mem_sets_the_budget_up_to_the_ceiling(self) -> None:
        """`--max-mem 1G` leaves 1 GB to plan with, as the PRD 18.1 dev loop
        needs (BAT-8). A value above the machine is cut to the ceiling."""
        assert probe.memory_budget("cpu", 1 * GB).usable_bytes == 1 * GB
        assert probe.memory_budget("mps", 100 * GB).usable_bytes == 62 * GB

    def test_usable_never_goes_negative(self) -> None:
        """A machine smaller than the OS reserve gives 0, not a negative budget."""
        self.total = self.free = 1 * GB
        assert probe.memory_budget("mps").usable_bytes == 0

    def test_os_reserve_is_largest_on_mps(self) -> None:
        """Unified memory is shared with the OS, so mps reserves 2 GB."""
        assert probe.OS_RESERVE_BYTES["mps"] == 2 * GB == max(probe.OS_RESERVE_BYTES.values())


class TestCapabilities:
    def test_int4_fast_path_false_when_kernel_missing(self, monkeypatch) -> None:
        """A missing `_weight_int4pack_mm` is a capability answer, not a raise."""

        def missing(*_args):
            raise AttributeError("no _weight_int4pack_mm")

        monkeypatch.setattr(quant, "int4_fast_ok", missing)
        assert probe.int4_fast_path("cpu") is False

    def test_probe_never_raises_on_a_failed_sub_probe(self, monkeypatch, tmp_path) -> None:
        """One unavailable field must not lose the whole `hello`."""

        def broken(_backend):
            raise RuntimeError("driver fell over")

        monkeypatch.setattr(probe, "memory_total_free", broken)
        caps = probe.probe_capabilities("cpu", tmp_path / "not" / "made" / "yet")
        assert (caps.mem_total_bytes, caps.mem_free_bytes) == (0, 0)
        assert caps.disk_free_bytes > 0  # falls back to the nearest existing parent
        assert caps.compute_dtype in ("bf16", "fp16", "fp32")

    def test_link_unknown_when_platform_tool_is_absent(self, monkeypatch) -> None:
        """No `networksetup` / `ip` / `netsh` gives `unknown`."""

        def absent(_cmd):
            raise FileNotFoundError

        monkeypatch.setattr(probe, "_run", absent)
        assert probe.detect_link() == "unknown"

    def test_link_reads_the_default_route_interface_on_macos(self, monkeypatch) -> None:
        ports = (
            "Hardware Port: Ethernet\nDevice: en5\nEthernet Address: aa\n\n"
            "Hardware Port: Wi-Fi\nDevice: en0\nEthernet Address: bb\n"
        )
        route = {"iface": "en0"}

        def fake(cmd):
            return f"  interface: {route['iface']}\n" if cmd[0] == "route" else ports

        monkeypatch.setattr(probe.sys, "platform", "darwin")
        monkeypatch.setattr(probe, "_run", fake)
        assert probe.detect_link() == "wifi"
        route["iface"] = "en5"
        assert probe.detect_link() == "wired"
        route["iface"] = "utun3"  # a VPN: no hardware port, so no guess
        assert probe.detect_link() == "unknown"

    def test_mps_free_memory_is_clamped_by_host_free(self, monkeypatch) -> None:
        """`recommended_max_memory` alone over-reports on unified memory."""
        monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 16 * GB)
        monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 0)
        host = SimpleNamespace(total=16 * GB, available=3 * GB)
        monkeypatch.setattr(psutil, "virtual_memory", lambda: host)
        assert probe.memory_total_free("mps") == (16 * GB, 3 * GB)


class TestBench:
    def test_reports_median_not_mean(self, monkeypatch) -> None:
        """One scheduler hiccup in 30 decode steps must not move the number."""
        calls = probe.BENCH_WARMUP_STEPS + probe.BENCH_DECODE_STEPS + probe.BENCH_PREFILL_PASSES
        seconds = [0.001] * calls
        seconds[probe.BENCH_WARMUP_STEPS + 3] = 1.0  # one decode step stalls for a second
        ticks, at = [], 0.0
        for s in seconds:
            ticks += [at, at + s]
            at += s
        clock = iter(ticks)
        monkeypatch.setattr(probe, "_now", lambda: next(clock))
        result = probe.run_bench(TINY, "bf16", "fp32", "cpu")
        assert result.t_dec_ms == pytest.approx(1.0)
        assert result.t_pre_ms == pytest.approx(1.0)

    def test_frees_every_tensor_it_allocated(self) -> None:
        """No tensor outlives the bench (PRD 6.3 step 5)."""

        def live_tensors() -> int:
            gc.collect()
            return sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))

        probe.run_bench(TINY, "bf16", "fp32", "cpu")  # warm any lazy torch state first
        before = live_tensors()
        probe.run_bench(TINY, "bf16", "fp32", "cpu")
        assert live_tensors() == before

    def test_completes_under_ten_seconds(self) -> None:
        """PRD 6.3 cost budget. Also takes the spec in its wire form."""
        start = time.perf_counter()
        result = probe.run_bench(TINY.to_dict(), "bf16", "fp32", "cpu")
        assert time.perf_counter() - start < 10
        assert result.t_dec_ms > 0 and result.t_pre_ms > 0
        assert (result.quant, result.dtype) == ("bf16", "fp32")


def test_dtype_names_round_trip() -> None:
    for name in ("bf16", "fp16", "fp32"):
        assert probe.dtype_name(probe.torch_dtype(name)) == name
