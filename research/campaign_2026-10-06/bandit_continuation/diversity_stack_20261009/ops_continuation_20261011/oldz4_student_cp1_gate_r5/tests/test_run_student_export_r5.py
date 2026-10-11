from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PACKET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKET_ROOT))
import run_student_export_r5 as runner  # noqa: E402


def _write_json(path: Path, value: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_root_admission_binds_new_r5_attempt_and_both_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scientific_contract = tmp_path / "scientific.json"
    runner_contract = tmp_path / "runner.json"
    gate_path = tmp_path / "gate.json"
    admission_path = tmp_path / "admission.json"
    scientific_hash = _write_json(
        scientific_contract,
        {"run": "model-run", "export": {"prediction_relative_path": "predictions.npy"}},
    )
    runner_hash = _write_json(
        runner_contract,
        {
            "schema": "wnn.local-student-export-runner-contract.r5",
            "status": "prepared_code_only_not_admitted",
            "base_contract_sha256": scientific_hash,
            "base_contract_relative_path": "scientific.json",
            "output_relative_path_from_base_contract": "predictions.npy",
            "attempt_id": "oldz4-r5-f0-s3-student-export",
            "run": "model-run",
            "gate_schema": "wnn.oldz4.student-cp1-terminal-gate.r5",
            "root_admission_schema": "wnn.oldz4.student-export-root-admission.r5",
            "authorized_action": "fixed-checkpoint-export-only",
        },
    )
    gate_hash = _write_json(
        gate_path,
        {
            "schema": "wnn.oldz4.student-cp1-terminal-gate.r5",
            "status": "pass",
            "checkpoint_sha256": "cp1-sha",
        },
    )
    _write_json(
        admission_path,
        {
            "schema": "wnn.oldz4.student-export-root-admission.r5",
            "status": "admitted",
            "attempt_id": "oldz4-r5-f0-s3-student-export",
            "run": "model-run",
            "contract_sha256": scientific_hash,
            "runner_contract_sha256": runner_hash,
            "checkpoint_gate_sha256": gate_hash,
            "checkpoint_sha256": "cp1-sha",
            "authorized_action": "fixed-checkpoint-export-only",
            "admitted_utc": "2026-10-12T00:00:00Z",
            "resource_profile_receipt": "profile-receipt-sha",
        },
    )
    monkeypatch.setattr(runner, "CONTRACT_PATH", scientific_contract)
    monkeypatch.setattr(runner, "RUNNER_CONTRACT_PATH", runner_contract)
    monkeypatch.setattr(runner, "EXPECTED_EXPORT_ATTEMPT_ID", "oldz4-r5-f0-s3-student-export")

    gate = runner._verify_root_export_admission(tmp_path, gate_path, admission_path)

    assert gate["checkpoint_sha256"] == "cp1-sha"


def test_r4_admission_cannot_authorize_r5_export_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scientific_contract = tmp_path / "scientific.json"
    runner_contract = tmp_path / "runner.json"
    gate_path = tmp_path / "gate.json"
    admission_path = tmp_path / "admission.json"
    scientific_hash = _write_json(
        scientific_contract,
        {"run": "model-run", "export": {"prediction_relative_path": "predictions.npy"}},
    )
    runner_hash = _write_json(
        runner_contract,
        {
            "schema": "wnn.local-student-export-runner-contract.r5",
            "status": "prepared_code_only_not_admitted",
            "base_contract_sha256": scientific_hash,
            "base_contract_relative_path": "scientific.json",
            "output_relative_path_from_base_contract": "predictions.npy",
            "attempt_id": "oldz4-r5-f0-s3-student-export",
            "run": "model-run",
            "gate_schema": "wnn.oldz4.student-cp1-terminal-gate.r5",
            "root_admission_schema": "wnn.oldz4.student-export-root-admission.r5",
            "authorized_action": "fixed-checkpoint-export-only",
        },
    )
    gate_hash = _write_json(
        gate_path,
        {
            "schema": "wnn.oldz4.student-cp1-terminal-gate.r5",
            "status": "pass",
            "checkpoint_sha256": "cp1-sha",
        },
    )
    _write_json(
        admission_path,
        {
            "schema": "wnn.oldz4.student-export-root-admission.r4",
            "status": "admitted",
            "attempt_id": "oldz4-r4-f0-s3-student-export",
            "run": "model-run",
            "contract_sha256": scientific_hash,
            "runner_contract_sha256": runner_hash,
            "checkpoint_gate_sha256": gate_hash,
            "checkpoint_sha256": "cp1-sha",
            "authorized_action": "fixed-checkpoint-export-only",
            "admitted_utc": "2026-10-12T00:00:00Z",
            "resource_profile_receipt": "profile-receipt-sha",
        },
    )
    monkeypatch.setattr(runner, "CONTRACT_PATH", scientific_contract)
    monkeypatch.setattr(runner, "RUNNER_CONTRACT_PATH", runner_contract)
    monkeypatch.setattr(runner, "EXPECTED_EXPORT_ATTEMPT_ID", "oldz4-r5-f0-s3-student-export")

    with pytest.raises(ValueError, match="root export admission schema differs"):
        runner._verify_root_export_admission(tmp_path, gate_path, admission_path)
