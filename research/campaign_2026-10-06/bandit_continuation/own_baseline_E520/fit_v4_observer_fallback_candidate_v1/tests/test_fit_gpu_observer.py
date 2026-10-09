"""Synthetic checks for E520 GPU-observer gates and startup handoff receipts."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fit_worker import FitAbort, FitResourceGuard, start_active_guard, stop_pre_cuda_guard
from support import gpu_inventory, gpu_observer, input_identity


class FakeCuda:
    """Small Torch CUDA stand-in for pure resource-observer tests."""

    def __init__(self, *, total: int, free: int) -> None:
        self.total = total
        self.free = free

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def get_device_properties(self, _index: int) -> Any:
        return SimpleNamespace(total_memory=self.total, major=7, minor=5)

    def mem_get_info(self, _index: int) -> tuple[int, int]:
        return self.free, self.total

    def memory_allocated(self, _index: int) -> int:
        return 100

    def memory_reserved(self, _index: int) -> int:
        return 200

    def max_memory_allocated(self, _index: int) -> int:
        return 300

    def max_memory_reserved(self, _index: int) -> int:
        return 400

    def device_count(self) -> int:
        return 1

    def get_device_name(self, _index: int) -> str:
        return "Synthetic GPU"


class FakeTorch:
    """Torch-like namespace sufficient for observer identity capture."""

    __version__ = "synthetic"
    version = SimpleNamespace(cuda="synthetic")
    backends = SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 1))

    def __init__(self, *, total: int, free: int) -> None:
        self.cuda = FakeCuda(total=total, free=free)


def _active_snapshot(used_upper_bound: int) -> dict[str, Any]:
    total = 16 * 1024**3
    free = total - used_upper_bound
    return {
        "cuda_initialized": True,
        "cuda_allocated_bytes": 100,
        "cuda_reserved_bytes": 200,
        "cuda_allocated_peak_bytes": 300,
        "cuda_reserved_peak_bytes": 400,
        "gpu_process_bytes": None,
        "gpu_process_query_status": "pid_not_reported",
        "gpu_process_query_raw": "",
        "gpu_memory_totals_match": True,
        "gpu_device_used_upper_bound_bytes": used_upper_bound,
        "gpu_total_bytes": total,
        "gpu_memory_info_total_bytes": total,
        "gpu_resource_query_status": "ok",
        "gpu_free_bytes": free,
    }


def test_missing_pid_uses_only_a_conservative_total_minus_free_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_observer,
        "nvidia_process_measurement",
        lambda _pid: {
            "bytes": None,
            "status": "pid_not_reported",
            "raw": "unmatched process table",
            "raw_byte_length": 25,
            "raw_sha256": "a" * 64,
            "raw_truncated": False,
            "stderr": "",
            "stderr_byte_length": 0,
            "stderr_sha256": "b" * 64,
            "stderr_truncated": False,
            "return_code": 0,
            "diagnostic": None,
        },
    )
    total = 16 * 1024**3
    observed = gpu_observer.cuda_resource_snapshot(
        FakeTorch(total=total, free=total - 2_000_000_000), 10
    )
    assert observed["gpu_process_bytes"] is None
    assert observed["gpu_process_query_status"] == "pid_not_reported"
    assert observed["gpu_device_used_upper_bound_bytes"] == 2_000_000_000
    gpu_observer.enforce_active_gpu_observation(observed, 3_000_000_000, 6 * 1024**3)

    over_cap = _active_snapshot(3_000_000_001)
    with pytest.raises(ValueError, match="upper bound exceeded"):
        gpu_observer.enforce_active_gpu_observation(over_cap, 3_000_000_000, 6 * 1024**3)


def test_missing_pid_fails_when_device_total_and_free_disagree() -> None:
    mismatched = _active_snapshot(1_000_000_000)
    mismatched["gpu_memory_totals_match"] = False
    mismatched["gpu_resource_query_status"] = "total_mismatch"
    with pytest.raises(ValueError, match="measurement is unavailable or inconsistent"):
        gpu_observer.enforce_active_gpu_observation(mismatched, 3_000_000_000, 6 * 1024**3)


def test_active_cuda_rejects_the_pre_cuda_idle_marker() -> None:
    invalid = _active_snapshot(1_000_000_000)
    invalid["gpu_process_query_status"] = "pre_cuda_idle"
    with pytest.raises(ValueError, match="not initialized"):
        gpu_observer.enforce_active_gpu_observation(invalid, 3_000_000_000, 6 * 1024**3)


def test_process_query_error_retains_raw_response_without_zero_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="driver query failed",
            stderr="no process table",
        ),
    )
    result = gpu_observer.nvidia_process_measurement(123)
    assert result["bytes"] is None
    assert result["status"] == "query_error"
    assert result["raw"] == "driver query failed"
    assert result["stderr"] == "no process table"
    assert result["return_code"] == 1


def test_physical_inventory_binds_uuid_and_cuda_memory_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    inventory = {
        "status": "ok",
        "rows": [
            {
                "index": 0,
                "name": "Synthetic GPU",
                "driver_version": "1",
                "uuid": "GPU-synthetic",
                "memory_used_mib": 100,
                "memory_total_mib": 16_384,
                "utilization_gpu_percent": 20,
                "utilization_memory_percent": 10,
            }
        ],
    }
    monkeypatch.setattr(gpu_inventory, "nvidia_smi_gpu_sample", lambda: inventory)
    identity = gpu_inventory.runtime_identity(FakeTorch(total=16 * 1024**3, free=14 * 1024**3))
    gpu_inventory.require_physical_gpu_inventory(identity)
    assert identity["selected_physical_gpu_uuid"] == "GPU-synthetic"
    assert (
        identity["physical_gpu_inventory_and_utilization"]["rows"][0]["utilization_gpu_percent"]
        == 20
    )


def test_sync_guard_persists_failed_snapshot_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    snapshot = {
        "rss_current_bytes": 1,
        "rss_peak_bytes": 1,
        "host_available_bytes": 8 * 1024**3,
        "host_total_bytes": 16 * 1024**3,
        "host_available_fraction": 0.5,
        "cgroup_available_bytes": 8 * 1024**3,
        "cgroup_limit_bytes": 16 * 1024**3,
        "cgroup_available_fraction": 0.5,
        "disk_free_bytes": 16 * 1024**3,
        **_active_snapshot(3_000_000_001),
        "gpu_process_query_status": "query_error",
        "gpu_process_query_raw": "raw nvidia-smi failure",
        "gpu_process_query_raw_sha256": "c" * 64,
    }
    monkeypatch.setattr(input_identity, "resource_snapshot", lambda *_args: snapshot)
    guard = FitResourceGuard(
        FakeTorch(total=16 * 1024**3, free=10 * 1024**3),
        tmp_path,
        receipts,
        60,
        started_at=time.monotonic(),
    )
    with pytest.raises(FitAbort, match="upper bound exceeded"):
        guard.check_full()
    resource = json.loads((receipts / "resource.json").read_text())
    abort = json.loads((receipts / "resource_abort.json").read_text())
    assert resource["snapshot"] == snapshot
    assert abort["last_snapshot"] == snapshot
    assert abort["last_snapshot"]["gpu_process_query_raw"] == "raw nvidia-smi failure"


def test_sync_guard_persists_last_snapshot_when_sampling_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    last_snapshot = {"phase": "last_good_sample", "gpu_process_bytes": None}
    guard = FitResourceGuard(None, tmp_path, receipts, 60, started_at=time.monotonic())
    guard.latest = last_snapshot

    def fail_sampling(*_args: Any) -> dict[str, Any]:
        raise OSError("synthetic resource observation failure")

    monkeypatch.setattr(input_identity, "resource_snapshot", fail_sampling)
    with pytest.raises(OSError, match="synthetic resource observation failure"):
        guard.check_full()
    abort = json.loads((receipts / "resource_abort.json").read_text())
    assert abort["last_snapshot"] == last_snapshot
    assert "synthetic resource observation failure" in abort["reason"]


def test_cuda_handoff_stops_idle_monitor_before_initialization_and_active_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    environment = receipts / "environment.json"
    events: list[str] = []
    pre_guard = FitResourceGuard(None, tmp_path, receipts, 60, started_at=0.0)
    pre_guard.latest = {"cuda_initialized": False}
    monkeypatch.setattr(pre_guard, "stop", lambda: events.append("idle_stopped"))

    def fake_start(guard: FitResourceGuard) -> None:
        events.append("active_started")
        guard.latest = {"cuda_initialized": True, "gpu_process_query_status": "matched_pid"}

    monkeypatch.setattr(FitResourceGuard, "start", fake_start)
    started = time.monotonic()
    stop_pre_cuda_guard(pre_guard, receipts, started)
    assert not environment.exists()
    events.append("cuda_initialized")
    runtime = {"physical_gpu_inventory_and_utilization": {"status": "synthetic"}}
    environment.write_text(json.dumps(runtime) + "\n")
    active = start_active_guard(
        FakeTorch(total=16 * 1024**3, free=14 * 1024**3),
        tmp_path,
        receipts,
        60,
        started,
        runtime,
    )
    handoff = json.loads((receipts / "resource_monitor_handoff.json").read_text())
    assert events == ["idle_stopped", "cuda_initialized", "active_started"]
    assert handoff["phase"] == "active_monitor_running"
    assert handoff["status"] == "active_resource_gate_passed"
    assert handoff["initial_active_snapshot"]["cuda_initialized"] is True
    assert active.torch is not None
