from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

import scripts.e540_kaggle_v8_worker as e540_v8_worker
from scripts.e540_kaggle_v8_worker import (
    PACKET_ID,
    RUN_NAMES,
    fit_update_from_provenance_record,
    trainer_argv,
    validate_fit_update_contract,
    validate_recipe_cli,
)

FIXTURE = Path(__file__).parent / "fixtures/E540_V7_ACTUAL_UPDATE_DIAGNOSTICS_ROW_0.json"
ORIGIN = Path(__file__).parent / "fixtures/E540_V7_ACTUAL_UPDATE_DIAGNOSTICS_ROW_0.origin.json"
FIXTURE_SHA256 = "48bc6d062606fb3e82a83174d286241b3a69e660a089a6b5a027f4e5aad9db02"
PRODUCER_PROVENANCE_SHA256 = "8082735f26a9989f5f87f8e51b7499e8653b201d60bcfba64d6116e5e0c80975"


def test_v8_merges_grad_stats_from_the_actual_nested_provenance_row() -> None:
    row = json.loads(FIXTURE.read_text(encoding="utf-8"))
    origin = json.loads(ORIGIN.read_text(encoding="utf-8"))

    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert origin["fixture_sha256"] == FIXTURE_SHA256
    assert origin["producer_sha256"] == PRODUCER_PROVENANCE_SHA256
    assert origin["producer_json_pointer"] == "/update_diagnostics/0"
    assert row["epoch_index"] == 0
    assert "grad_stats" not in row["epoch"]
    assert row["grad_stats"]["steps"] == 72_996
    assert row["grad_stats"]["nonfinite_steps"] == 0

    update = fit_update_from_provenance_record(row)
    validate_fit_update_contract(update, 72_996)

    assert update["optimizer_attempts"] == 72_996
    assert update["optimizer_completed_updates"] == 72_996
    assert update["amp_skipped_updates"] == 0
    assert update["grad_stats"] == row["grad_stats"]


def test_v8_keeps_exact_update_and_finiteness_gates_strict() -> None:
    row = json.loads(FIXTURE.read_text(encoding="utf-8"))
    update = fit_update_from_provenance_record(row)

    with pytest.raises(ValueError, match="exact finite FP32"):
        validate_fit_update_contract({**update, "optimizer_completed_updates": 72_995}, 72_996)
    with pytest.raises(ValueError, match="exact finite FP32"):
        validate_fit_update_contract(
            {**update, "grad_stats": {**update["grad_stats"], "nonfinite_steps": 1}}, 72_996
        )
    with pytest.raises(ValueError, match="missing GradStats"):
        validate_fit_update_contract(fit_update_from_provenance_record({**row, "grad_stats": None}), 72_996)


def test_v8_actual_cli_parsers_accept_both_arm_profiles_fits_and_predictions() -> None:
    checks = validate_recipe_cli()

    assert PACKET_ID == "E540_FP32_V8"
    assert len(checks) == 6
    assert all(item["status"] == "parsed" for item in checks)
    assert all(item["script_token_removed"] for item in checks)
    for mode, run_name in RUN_NAMES.items():
        argv = trainer_argv(mode, profile=False)
        assert argv[argv.index("--name") + 1] == run_name
        assert argv[argv.index("--precision") + 1] == "fp32"
        assert "--grad-stats" in argv


def test_v8_pending_authorization_stops_before_mounted_input_hashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt_path = tmp_path / "pending-auth.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "wnn.e540-conditional-fit-authorization.v1",
                "owner": "root",
                "status": "pending_root_review",
                "authorized": False,
                "conditional_fit_after_profile_gate": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(e540_v8_worker, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(e540_v8_worker, "AUTHORIZATION_PATH", receipt_path)
    monkeypatch.setenv("WNN_KERNEL_DEADLINE_MONOTONIC", str(time.monotonic() + 6000))
    monkeypatch.setenv("WNN_E540_PACKET_ID", PACKET_ID)
    monkeypatch.setenv("WNN_E540_CODE_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("WNN_E540_RECIPE_SHA256", "b" * 64)
    monkeypatch.setenv("WNN_E540_SUBSET_SHA256", "c" * 64)
    monkeypatch.setenv("WNN_E540_AUTHORIZATION_PATH", receipt_path.as_posix())
    monkeypatch.setenv("WNN_E540_KERNEL_WALL_SECONDS", "6000")
    monkeypatch.setenv("WNN_E540_WORKER_WALL_SECONDS", "5200")

    def reject_data_hash() -> None:
        raise AssertionError("pending authorization must stop before mounted-input hashing")

    monkeypatch.setattr(e540_v8_worker, "verify_mounted_data", reject_data_hash)
    with pytest.raises(ValueError, match="missing valid status"):
        e540_v8_worker.run()

    preflight = json.loads((tmp_path / "runtime/startup_preflight.json").read_text())
    assert preflight["status"] == "failed_before_mounted_data_hashing"
    assert len(preflight["cli_checks"]) == 6
