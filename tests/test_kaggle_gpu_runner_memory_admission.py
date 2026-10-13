from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.kaggle_gpu_runner import _memory_capacity_snapshot, admit_specs

GIB = 1024**3
HOST_CAPACITY = 32_212_254_720
POST_PROFILE_AVAILABLE = 8_097_878_016
T4_TOTAL = 16_106_127_360
T4_STOP = 13 * GIB


def _device() -> dict[str, int | str]:
    return {
        "name": "Tesla T4",
        "total_bytes": T4_TOTAL,
        "used_bytes": 0,
        "index": 0,
        "uuid": "test-device",
    }


def _spec(name: str, ram_bytes: int, *, cpus: int = 2) -> dict:
    return {
        "name": name,
        "limits": {
            "cpus": cpus,
            "ram_bytes": ram_bytes,
            "gpu_stop_bytes": T4_STOP,
        },
    }


def _host(available_bytes: int = POST_PROFILE_AVAILABLE) -> dict:
    return {
        "capacity_bytes": HOST_CAPACITY,
        "available_bytes": available_bytes,
        "affinity": [0, 1, 2, 3],
    }


def test_post_profile_fit_fits_measured_headroom_with_one_gb_reserve() -> None:
    fit = _spec("fit", 6_000_000_000)

    concurrency = admit_specs([fit], _host(), [_device(), _device()])

    assert concurrency == 2


def test_r2_eight_gb_fit_reservation_still_fails_closed() -> None:
    fit = _spec("fit", 8_000_000_000)

    with pytest.raises(ValueError, match="operational budget"):
        admit_specs([fit], _host(), [_device(), _device()])


def test_reservation_must_leave_full_one_gb_operational_headroom() -> None:
    fit = _spec("fit", 6_000_000_000)

    with pytest.raises(ValueError, match="operational budget"):
        admit_specs(
            [fit],
            _host(available_bytes=6_999_999_999),
            [_device(), _device()],
        )


def test_combined_worker_reservations_are_serialized_at_live_headroom() -> None:
    workers = [_spec("worker-a", 4_000_000_000), _spec("worker-b", 4_000_000_000)]

    concurrency = admit_specs(workers, _host(), [_device(), _device()])

    assert concurrency == 1


def test_physical_capacity_fraction_remains_a_hard_runner_bound() -> None:
    worker = _spec("worker", int(HOST_CAPACITY * 0.8) + 1)
    host = _host(available_bytes=HOST_CAPACITY)

    with pytest.raises(ValueError, match="operational budget"):
        admit_specs([worker], host, [_device()])


def test_memory_snapshot_counts_inactive_file_as_reclaimable_but_reports_it() -> None:
    snapshot = _memory_capacity_snapshot(
        physical_capacity_bytes=32 * GIB,
        mem_available_bytes=24 * GIB,
        cgroup_limit_bytes=32 * GIB,
        cgroup_current_bytes=30 * GIB,
        cgroup_memory_stat={
            "anon": 8 * GIB,
            "file": 22 * GIB,
            "shmem": GIB,
            "inactive_file": 20 * GIB,
            "active_file": 2 * GIB,
        },
    )

    assert snapshot["cgroup_free_bytes"] == 2 * GIB
    assert snapshot["cgroup_reclaimable_available_bytes"] == 22 * GIB
    assert snapshot["cgroup_inactive_file_bytes"] == 20 * GIB
    assert snapshot["cgroup_anon_bytes"] == 8 * GIB
    assert snapshot["available_bytes"] == 22 * GIB


def test_missing_cgroup_cache_measurement_gets_no_reclaimable_credit() -> None:
    snapshot = _memory_capacity_snapshot(
        physical_capacity_bytes=32 * GIB,
        mem_available_bytes=24 * GIB,
        cgroup_limit_bytes=32 * GIB,
        cgroup_current_bytes=30 * GIB,
        cgroup_memory_stat={},
    )

    assert snapshot["cgroup_reclaimable_available_bytes"] == 2 * GIB
    assert snapshot["available_bytes"] == 2 * GIB
