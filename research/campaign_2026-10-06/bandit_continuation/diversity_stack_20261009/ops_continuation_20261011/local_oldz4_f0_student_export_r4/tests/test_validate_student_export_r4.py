from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import validate_student_export_r4 as validator
from validate_student_export_r4 import (
    _assert_model_finite,
    _check_update_diagnostics,
    validate_export,
    validate_prediction_payload,
    write_once,
)


def _epoch(attempts: int, completed: int, skipped: int, batches: int) -> dict[str, int]:
    return {
        "optimizer_attempts": attempts,
        "optimizer_completed_updates": completed,
        "amp_skipped_updates": skipped,
        "completed_batches": batches,
    }


def _diagnostic_pair() -> tuple[list[dict], list[dict]]:
    diagnostics = []
    history = []
    cumulative = {key: 0 for key in _epoch(0, 0, 0, 0)}
    for index in range(2):
        counters = _epoch(4, 3, 1, 2)
        for key, value in counters.items():
            cumulative[key] += value
        diagnostic = {"epoch_index": index, "epoch": counters, "cumulative": dict(cumulative)}
        diagnostics.append(diagnostic)
        history.append({"epoch": index, "update_diagnostics": diagnostic})
    return diagnostics, history


def _sidecar(
    prediction_sha256: str,
    row_groups: list[int],
    checkpoint_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "run": "oldz4_h230_student_f0_s3_r3",
        "split": "valid",
        "epoch": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_sha256": prediction_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "row_groups": row_groups,
        "inference_precision": "fp16",
        "quantization_preset": None,
        "recurrent_rank": None,
        "evaluation_membership_sha256": None,
        "fit_subset_manifest_sha256": None,
    }


def test_update_diagnostics_allow_reported_amp_skips_when_counts_reconcile() -> None:
    diagnostics, history = _diagnostic_pair()

    totals = _check_update_diagnostics(diagnostics, history)

    assert totals == {
        "optimizer_attempts": 8,
        "optimizer_completed_updates": 6,
        "amp_skipped_updates": 2,
        "completed_batches": 4,
    }


def test_update_diagnostics_reject_attempt_count_mismatch() -> None:
    diagnostics, history = _diagnostic_pair()
    diagnostics[1]["epoch"]["optimizer_attempts"] = 7
    history[1]["update_diagnostics"]["epoch"]["optimizer_attempts"] = 7

    with pytest.raises(ValueError, match="optimizer counters do not reconcile"):
        _check_update_diagnostics(diagnostics, history)


def test_update_diagnostics_reject_missing_epoch() -> None:
    diagnostics, history = _diagnostic_pair()

    with pytest.raises(ValueError, match="both fit epochs"):
        _check_update_diagnostics(diagnostics[:1], history[:1])


def test_model_finite_gate_accepts_finite_weights_and_readout() -> None:
    checkpoint = {"model": {"weight": torch.ones(2, 3)}, "readout": [torch.zeros(2)]}

    assert _assert_model_finite(checkpoint) == 1


def test_model_finite_gate_rejects_nonfinite_weight() -> None:
    checkpoint = {"model": {"weight": torch.tensor([1.0, float("nan")])}}

    with pytest.raises(ValueError, match="nonfinite model tensor"):
        _assert_model_finite(checkpoint)


def test_prediction_gate_accepts_exact_shape_sidecar_and_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "prediction.npy"
    prediction = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.save(path, prediction)
    prediction_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    row_groups = [7, 19]
    metadata = _sidecar(prediction_hash, row_groups, "checkpoint", "split")

    actual = validate_prediction_payload(
        path,
        metadata,
        run="oldz4_h230_student_f0_s3_r3",
        epoch=1,
        checkpoint_sha256="checkpoint",
        split_manifest_sha256="split",
        expected_row_groups=row_groups,
        expected_shape=(2, 3, 4),
        expected_bytes=path.stat().st_size,
    )

    assert actual == prediction_hash


def test_prediction_gate_rejects_reordered_sidecar_membership(tmp_path: Path) -> None:
    path = tmp_path / "prediction.npy"
    np.save(path, np.zeros((2, 3, 4), dtype=np.float32))
    prediction_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = _sidecar(prediction_hash, [19, 7], "checkpoint", "split")

    with pytest.raises(ValueError, match="sidecar differs for row_groups"):
        validate_prediction_payload(
            path,
            metadata,
            run="oldz4_h230_student_f0_s3_r3",
            epoch=1,
            checkpoint_sha256="checkpoint",
            split_manifest_sha256="split",
            expected_row_groups=[7, 19],
            expected_shape=(2, 3, 4),
            expected_bytes=path.stat().st_size,
        )


def test_prediction_gate_rejects_nonfinite_payload(tmp_path: Path) -> None:
    path = tmp_path / "prediction.npy"
    np.save(path, np.array([[[float("inf")]]], dtype=np.float32))
    prediction_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = _sidecar(prediction_hash, [7], "checkpoint", "split")

    with pytest.raises(ValueError, match="nonfinite prediction"):
        validate_prediction_payload(
            path,
            metadata,
            run="oldz4_h230_student_f0_s3_r3",
            epoch=1,
            checkpoint_sha256="checkpoint",
            split_manifest_sha256="split",
            expected_row_groups=[7],
            expected_shape=(1, 1, 1),
            expected_bytes=path.stat().st_size,
        )


def test_receipt_write_never_overwrites_prior_evidence(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    original = {"status": "first"}
    write_once(path, original)
    before = path.read_bytes()

    with pytest.raises(FileExistsError):
        write_once(path, {"status": "retry"})

    assert path.read_bytes() == before
    assert json.loads(before) == original


def test_export_gate_uses_nested_contract_prediction_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    run = "oldz4_h230_student_f0_s3_r3"
    prediction = project / "cache" / f"pred_{run}_valid.npy"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"fixture-npy")
    prediction.with_suffix(".json").write_text("{}", encoding="utf-8")
    checkpoint = project / "artifacts" / "runs" / run / "model_ep1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture-checkpoint")
    checkpoint_hash = validator.sha256_file(checkpoint)
    target_index = project / "cache" / "valid_targets.npz"
    np.savez(target_index, row_group=np.arange(1873, dtype=np.int64))

    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "run": run,
                "split_manifest_sha256": "split-hash",
                "exposure": "fixture exposure",
                "export": {
                    "prediction_relative_path": "cache/pred_oldz4_h230_student_f0_s3_r3_valid.npy"
                },
            }
        ),
        encoding="utf-8",
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "schema": "wnn.oldz4.student-cp1-terminal-gate.r4",
                "status": "pass",
                "run": run,
                "checkpoint_epoch": 1,
                "checkpoint_sha256": checkpoint_hash,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "CONTRACT_PATH", contract_path)

    def accept_payload(
        actual_path: Path,
        metadata: dict[str, object],
        *,
        expected_shape: tuple[int, int, int],
        expected_bytes: int,
        **_kwargs: object,
    ) -> str:
        assert actual_path == prediction
        assert expected_shape == (1873, 20000, 2)
        assert expected_bytes == 299_680_128
        return "prediction-hash"

    monkeypatch.setattr(validator, "validate_prediction_payload", accept_payload)

    receipt = validate_export(project, gate_path)

    assert receipt["prediction_path"] == str(prediction)
    assert receipt["prediction_sha256"] == "prediction-hash"
