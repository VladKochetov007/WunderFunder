from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import scripts.kaggle_gpu_runner as runner


def write_receipt(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")


def valid_cohort(**overrides: object) -> dict[str, object]:
    return {
        "packet_id": runner.E540_V7_PACKET_ID,
        "precision": "fp32",
        "phase": "profile_linear",
        "status": "profile_pending",
        "started_unix": 1_700_000_000.0,
        **overrides,
    }


def test_profile_phase_projects_only_allowlisted_receipt_fields(tmp_path: Path) -> None:
    write_receipt(tmp_path / "e540_cohort/cohort.json", valid_cohort(api_token="SECRET"))
    write_receipt(
        tmp_path / "e540_cohort/receipts/linear.json",
        {
            "mode": "linear",
            "status": "profile_pass",
            "error": "SECRET_ERROR",
            "profile": {
                "profile_wall_seconds": 12.25,
                "step_wall_ms": 8.5,
                "profile_update_evidence": {
                    "optimizer_completed_updates": 316,
                    "extra_private_value": "SECRET_NESTED",
                },
            },
        },
    )

    result = runner._e540_worker_phase(tmp_path, now_unix=1_700_000_010.0)

    assert result == {
        "status": "available",
        "stage": "profile",
        "arm_id": "linear",
        "cohort_status": "profile_pending",
        "worker_elapsed_seconds": 10.0,
        "arm_status": "profile_pass",
        "phase_wall_seconds": 12.25,
        "optimizer_completed_updates": 316,
        "step_wall_ms": 8.5,
    }
    assert "SECRET" not in json.dumps(result)


@pytest.mark.parametrize(
    "payload",
    [
        valid_cohort(phase="profile_linear_SECRET"),
        valid_cohort(status="SECRET_STATUS"),
        valid_cohort(packet_id="OTHER_PACKET"),
        valid_cohort(started_unix=float("nan")),
        valid_cohort(started_unix=10**1000),
        valid_cohort(started_unix=1_700_006_001.0),
    ],
)
def test_untrusted_cohort_fields_fail_closed_without_echo(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    write_receipt(tmp_path / "e540_cohort/cohort.json", payload)
    result = runner._e540_worker_phase(tmp_path, now_unix=1_700_000_000.0)

    assert result == {"status": "unavailable"}
    assert "SECRET" not in json.dumps(result)


def test_malformed_oversized_and_symlink_receipts_are_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "e540_cohort/cohort.json"
    path.parent.mkdir(parents=True)
    path.write_text("{malformed", encoding="utf-8")
    assert runner._e540_worker_phase(tmp_path) == {"status": "unavailable"}

    path.write_text(" " * (64 * 1024 + 1), encoding="utf-8")
    assert runner._e540_worker_phase(tmp_path) == {"status": "unavailable"}

    path.unlink()
    outside = tmp_path / "outside.json"
    write_receipt(outside, valid_cohort())
    path.symlink_to(outside)
    assert runner._e540_worker_phase(tmp_path) == {"status": "unavailable"}

    linked_root = tmp_path / "linked-root"
    linked_root.mkdir()
    real_directory = tmp_path / "real-e540-cohort"
    write_receipt(real_directory / "cohort.json", valid_cohort())
    (linked_root / "e540_cohort").symlink_to(real_directory, target_is_directory=True)
    assert runner._e540_worker_phase(linked_root) == {"status": "unavailable"}


def test_numeric_metrics_are_bounded_and_nonfinite_values_are_not_emitted(tmp_path: Path) -> None:
    write_receipt(tmp_path / "e540_cohort/cohort.json", valid_cohort())
    write_receipt(
        tmp_path / "e540_cohort/receipts/linear.json",
        {
            "mode": "linear",
            "status": "profile_pass",
            "profile": {
                "profile_wall_seconds": 5_201,
                "step_wall_ms": float("inf"),
                "profile_update_evidence": {"optimizer_completed_updates": True},
            },
        },
    )

    result = runner._e540_worker_phase(tmp_path, now_unix=1_700_000_001.0)

    assert result["status"] == "available"
    assert "phase_wall_seconds" not in result
    assert "step_wall_ms" not in result
    assert "optimizer_completed_updates" not in result
    assert "NaN" not in json.dumps(result)
    assert "Infinity" not in json.dumps(result)


def test_gate_receipt_reports_only_bounded_projection_fields(tmp_path: Path) -> None:
    write_receipt(
        tmp_path / "e540_cohort/cohort.json",
        valid_cohort(phase="profile_gate_rejected", status="profile_gate_rejected"),
    )
    write_receipt(
        tmp_path / "e540_cohort/profile_gate.json",
        {
            "status": "reject",
            "projected_total_worker_seconds": 6_000,
            "maximum_worker_seconds": 5_200,
            "worker_elapsed_seconds_at_gate": 800,
            "private_input_hash": "SECRET",
        },
    )

    result = runner._e540_worker_phase(tmp_path, now_unix=1_700_000_010.0)

    assert result["stage"] == "profile_gate"
    assert result["profile_gate_status"] == "reject"
    assert result["projected_total_worker_seconds"] == 6_000.0
    assert result["maximum_worker_seconds"] == 5_200.0
    assert result["worker_elapsed_seconds_at_gate"] == 800.0
    assert "SECRET" not in json.dumps(result)


def test_outer_heartbeat_surfaces_safe_e540_phase_but_generic_jobs_stay_unchanged(
    tmp_path: Path,
) -> None:
    write_receipt(
        tmp_path / "e540_cohort/cohort.json",
        valid_cohort(started_unix=time.time() - 10),
    )
    job = {
        "spec": {"name": "e540-v7-fp32-20261008"},
        "device": {"index": 0, "uuid": "GPU-test"},
        "started": time.monotonic() - 1,
        "worker_timeout_seconds": 5_200,
        "peak_ram_bytes": 10,
        "peak_gpu_bytes": 20,
        "e540_phase_root": tmp_path,
    }
    heartbeat = runner.resource_heartbeat([job], [], [], time.monotonic())

    assert heartbeat["active"][0]["worker_phase"]["stage"] == "profile"
    assert heartbeat["active"][0]["worker_phase"]["arm_id"] == "linear"
    generic = {key: value for key, value in job.items() if key != "e540_phase_root"}
    generic["spec"] = {"name": "unrelated"}
    generic_heartbeat = runner.resource_heartbeat([generic], [], [], time.monotonic())
    assert "worker_phase" not in generic_heartbeat["active"][0]
