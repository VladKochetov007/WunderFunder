from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fit_headroom_gate_r4 import (
    EXPECTED_TPU_DEVICE_COUNT,
    HEADROOM_POLICY_SCHEMA,
    _cgroup_memory,
    evaluate_actual_device_inventory,
    evaluate_live_resources,
    evaluate_profile_headroom,
)


def _policy(**overrides: Any) -> dict[str, Any]:
    return {
        "schema": HEADROOM_POLICY_SCHEMA,
        "minimum_device_free_bytes": 100,
        "minimum_host_available_bytes": 200,
        "minimum_cgroup_headroom_bytes": 300,
        "minimum_disk_free_bytes": 400,
        **overrides,
    }


def _live(**overrides: Any) -> dict[str, Any]:
    return {
        "host_available_bytes": 1_000,
        "cgroup": {
            "version": "v2",
            "current_bytes": 0,
            "ceiling_bytes": 1_000,
            "limit_sample_complete": True,
            "headroom_bytes": 1_000,
        },
        "disk_free_bytes": 1_000,
        **overrides,
    }


def _profile(*, device_peak: int = 800) -> dict[str, Any]:
    snapshot = {
        "device_memory_stats": [
            {"bytes_limit": 1_000, "bytes_in_use": 700, "peak_bytes_in_use": device_peak}
            for _ in range(2)
        ]
    }
    return {
        "wmse": {"shape": [32, 2000, 211], "finite": True, "memory_after_full_shape": snapshot},
        "huber": {"shape": [32, 2000, 211], "finite": True, "memory_after_full_shape": snapshot},
    }


def test_profile_and_live_headroom_pass_only_with_actual_capacity_margins() -> None:
    result = evaluate_profile_headroom(_policy(), _profile(), _live(), 2)

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["device_checks"]["wmse"]["devices"][0]["free_after_peak_bytes"] == 200


def test_missing_device_stats_or_live_limits_fail_closed() -> None:
    result = evaluate_profile_headroom(_policy(), {}, _live(), 2)

    assert result["passed"] is False
    assert "missing_or_incomplete_wmse_device_memory_stats" in result["errors"]
    assert "missing_or_incomplete_huber_device_memory_stats" in result["errors"]

    no_live_capacity = evaluate_live_resources(_policy(), {"disk_free_bytes": 1_000})
    assert no_live_capacity["passed"] is False
    assert "insufficient_or_missing_live_host_available_bytes" in no_live_capacity["errors"]
    assert "insufficient_or_missing_live_cgroup_headroom_bytes" in no_live_capacity["errors"]


def test_unbounded_cgroup_requires_host_reserve_and_is_recorded_explicitly() -> None:
    result = evaluate_live_resources(
        _policy(),
        {
            "host_available_bytes": 1_000,
            "cgroup": {
                "version": "v2",
                "current_bytes": 500,
                "ceiling_bytes": None,
                "limit_sample_complete": True,
                "headroom_bytes": None,
            },
            "disk_free_bytes": 1_000,
        },
    )

    assert result["passed"] is True
    assert result["live_checks"]["cgroup_headroom_bytes"]["basis"] == (
        "no_finite_cgroup_ceiling; host_available_reserve_enforced"
    )


def test_incomplete_cgroup_limit_sample_does_not_qualify_as_unbounded() -> None:
    result = evaluate_live_resources(
        _policy(),
        {
            "host_available_bytes": 1_000,
            "cgroup": {
                "version": "v2",
                "current_bytes": 500,
                "ceiling_bytes": 800,
                "limit_sample_complete": False,
                "headroom_bytes": 300,
            },
            "disk_free_bytes": 1_000,
        },
    )

    assert result["passed"] is False
    assert "insufficient_or_missing_live_cgroup_headroom_bytes" in result["errors"]


def test_cgroup_v2_requires_both_limit_files_to_identify_unbounded_memory(
    tmp_path: Path,
) -> None:
    (tmp_path / "memory.current").write_text("500\n")
    (tmp_path / "memory.high").write_text("max\n")
    (tmp_path / "memory.max").write_text("max\n")

    complete = _cgroup_memory(tmp_path)

    assert complete["limit_sample_complete"] is True
    assert complete["ceiling_bytes"] is None
    assert complete["headroom_bytes"] is None

    (tmp_path / "memory.max").unlink()
    incomplete = _cgroup_memory(tmp_path)

    assert incomplete["limit_sample_complete"] is False


def test_device_host_cgroup_or_disk_below_root_reserve_defers_fit() -> None:
    low_device = evaluate_profile_headroom(_policy(), _profile(device_peak=950), _live(), 2)
    assert low_device["passed"] is False
    assert "insufficient_wmse_device_0_headroom" in low_device["errors"]

    low_live = evaluate_profile_headroom(
        _policy(),
        _profile(),
        _live(
            host_available_bytes=199,
            cgroup={
                "version": "v2",
                "current_bytes": 0,
                "ceiling_bytes": 299,
                "limit_sample_complete": True,
                "headroom_bytes": 299,
            },
            disk_free_bytes=399,
        ),
        2,
    )
    assert low_live["passed"] is False
    assert "insufficient_or_missing_live_host_available_bytes" in low_live["errors"]
    assert "insufficient_or_missing_live_cgroup_headroom_bytes" in low_live["errors"]
    assert "insufficient_or_missing_live_disk_free_bytes" in low_live["errors"]


def test_unapproved_or_incomplete_policy_cannot_admit_fit() -> None:
    no_policy = evaluate_profile_headroom(None, _profile(), _live(), 2)
    missing_reserve = evaluate_profile_headroom(
        {"schema": HEADROOM_POLICY_SCHEMA}, _profile(), _live(), 2
    )

    assert no_policy["passed"] is False
    assert "missing_or_invalid_headroom_policy" in no_policy["errors"]
    assert missing_reserve["passed"] is False
    assert any(
        error.startswith("invalid_or_missing_policy:") for error in missing_reserve["errors"]
    )


def test_actual_tpu_device_inventory_must_match_the_eight_device_packet() -> None:
    inventory = [
        {"id": f"TPU:{index}", "platform": "tpu", "kind": "TPU v5e"}
        for index in range(EXPECTED_TPU_DEVICE_COUNT)
    ]
    assert evaluate_actual_device_inventory("tpu", 8, inventory)["passed"] is True
    failed = evaluate_actual_device_inventory("tpu", 7, inventory[:7])
    assert failed["passed"] is False
    assert "actual_accelerator_count_differs_from_tpu_v5e8" in failed["errors"]
    assert "actual_device_inventory_is_incomplete" in failed["errors"]
