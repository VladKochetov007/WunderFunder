from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

PACKET = Path(__file__).with_name("h217_packet_v4")
SPEC = importlib.util.spec_from_file_location("h217_v4_worker", PACKET / "worker.py")
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


@pytest.mark.parametrize("value", ["1", "600", "7200", "21600"])
def test_accepts_positive_incoming_runtime_under_reviewed_ceiling(value: str) -> None:
    assert WORKER.validate_incoming_runtime(value, 21600) == int(value)


@pytest.mark.parametrize("value", ["0", "-1", "21601", "", "not-a-number"])
def test_rejects_invalid_or_over_ceiling_incoming_runtime(value: str) -> None:
    with pytest.raises(ValueError):
        WORKER.validate_incoming_runtime(value, 21600)


def test_phase_deadline_uses_actual_incoming_runtime_and_reserves() -> None:
    deadline = WORKER.cohort_phase_deadline(100.0, 5400, 450, 600)
    assert deadline == 4450.0
    assert deadline < 100.0 + 21600 - 450 - 600


def test_profile_projection_runs_tcn_and_defers_unaffordable_gru(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    output = tmp_path / "output"
    output.mkdir()
    fits = {
        "tcn": {
            "argv": [],
            "output_path": "runs/tcn",
            "run_name": "tcn-run",
            "prediction_export": {"output_name": "tcn"},
        },
        "gru": {
            "argv": [],
            "output_path": "runs/gru",
            "run_name": "gru-run",
            "prediction_export": {"output_name": "gru"},
        },
    }
    recipe = {
        "profile_sequence": [
            {"arm": "tcn", "argv": []},
            {"arm": "gru", "argv": []},
        ],
        "fit_branches": {"batch8": [{"arm": arm, **branch} for arm, branch in fits.items()]},
    }
    calls: list[str] = []

    def phase_runner(
        argv: list[str],
        name: str,
        timeout_seconds: int,
        project_root: Path,
        output_root: Path,
        deadline: float,
    ) -> dict[str, object]:
        calls.append(name)
        if name.startswith("fit_"):
            arm = name.removeprefix("fit_")
            checkpoint = project_root / fits[arm]["output_path"] / "model_ep3.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"fixed epoch 3 checkpoint")
        return {"exit_code": 0, "error": None, "elapsed_seconds": 0.01}

    def gate_runner(path: Path, fit: dict[str, object]) -> dict[str, object]:
        projection = 700 if "tcn" in path.name else 5000
        return {"status": "passed", "projected_fit_and_export_seconds": projection}

    monkeypatch.setattr(
        WORKER,
        "validate_prediction",
        lambda project_root, export, checkpoint: {"verified": True},
    )
    result: dict[str, object] = {}
    deadline = time.monotonic() + 2600
    WORKER.execute_arms(recipe, project, output, deadline, result, phase_runner, gate_runner)

    arms = result["arms"]
    assert arms["tcn"]["delivery_status"] == "delivered_ungraded"
    assert arms["gru"]["fit_status"] == "deferred_cohort_runtime"
    assert "fit_gru" not in calls
    assert arms["tcn"]["runtime_admission"]["later_arm_projected_delivery_seconds"] == 6800


def _profile_log(path: Path, evidence: dict[str, object]) -> Path:
    path.write_text(
        '{"precision":"fp16","amp_initial_scale":4096}\n'
        'PROFILE {"step_wall":100}\n'
        f"PROFILE_UPDATE_EVIDENCE {json.dumps(evidence)}\n"
    )
    return path


def _valid_profile_evidence() -> dict[str, object]:
    return {
        "optimizer_attempts": 230,
        "optimizer_completed_updates": 230,
        "amp_skipped_updates": 0,
        "loss_checks": 230,
        "loss_finite": True,
        "gradient_finite": True,
        "counts_consistent": True,
        "valid_for_throughput": True,
        "scaler_initial_scale": 4096,
    }


def test_profile_gate_accepts_exact_230_update_evidence(tmp_path: Path) -> None:
    log = _profile_log(tmp_path / "profile.log", _valid_profile_evidence())
    gate = WORKER.profile_gate(log, {"batch_size": 8})
    assert gate["status"] == "passed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("optimizer_attempts", None),
        ("optimizer_attempts", 0),
        ("optimizer_attempts", True),
        ("optimizer_completed_updates", 229),
        ("amp_skipped_updates", 1),
        ("loss_checks", 229),
    ],
)
def test_profile_gate_rejects_missing_or_mismatched_update_counts(
    tmp_path: Path, field: str, value: object
) -> None:
    evidence = _valid_profile_evidence()
    if value is None:
        evidence.pop(field)
    else:
        evidence[field] = value
    log = _profile_log(tmp_path / "profile.log", evidence)
    with pytest.raises(ValueError, match="profile update counter evidence"):
        WORKER.profile_gate(log, {"batch_size": 8})


def test_short_positive_runtime_is_packaged_as_deferred_without_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = json.loads((PACKET / "runtime_resources.json").read_text())
    output = tmp_path / "output"
    monkeypatch.setenv("WNN_OUTPUT_DIR", str(output))
    monkeypatch.setenv("WNN_ATTEMPT_ID", "short-runtime-test")
    monkeypatch.setenv("WNN_PACKET_SHA256", "test-packet")
    monkeypatch.setenv("WNN_MAX_RUNTIME_SECONDS", "1")
    for environment, field in {
        "WNN_THREADS": "cpu_threads",
        "WNN_LOADER_WORKERS": "loader_workers",
        "WNN_WORKER_RAM_BYTES": "worker_memory_max_bytes",
        "WNN_WORKER_VRAM_BYTES": "observed_total_vram_max_bytes",
        "WNN_GPU_MEMORY_BYTES": "cuda_allocator_max_bytes",
    }.items():
        monkeypatch.setenv(environment, str(runtime[field]))

    assert WORKER.main() == 0
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "deferred_insufficient_runtime"
    assert result["runtime_budget"]["incoming_runtime_seconds"] == 1
    assert result["runtime_budget"]["phase_deadline_offset_seconds"] == -1049
    assert all(
        arm["profile_status"] == "deferred_cohort_runtime"
        and arm["fit_status"] == "deferred_cohort_runtime"
        for arm in result["arms"].values()
    )
    assert (output / "artifacts.tar").is_file()
