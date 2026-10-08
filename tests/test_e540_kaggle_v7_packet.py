from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

import scripts.e540_kaggle_v7_worker as e540_v7_worker
from scripts.e540_kaggle_v7_worker import (
    AUTHORIZATION_PATH,
    EXPECTED_PROFILE_ATTEMPTS,
    PACKET_ID,
    RUN_NAMES,
    SUBSET_MANIFEST,
    checkpoint_named_tensors,
    expected_fit_updates,
    parse_trainer_output,
    predict_argv,
    trainer_argv,
    validate_conditional_authorization,
    validate_finite_prediction_array,
    validate_finite_tensors,
    validate_fit_update_contract,
    validate_profile,
    validate_recipe_cli,
)


def _fp32_profile_output(
    *,
    skipped: int = 0,
    nonfinite: int = 0,
    gradient_stats_steps: int = EXPECTED_PROFILE_ATTEMPTS,
    scaler_enabled: bool = False,
) -> str:
    completed = EXPECTED_PROFILE_ATTEMPTS - skipped
    evidence = {
        "optimizer_attempts": EXPECTED_PROFILE_ATTEMPTS,
        "optimizer_completed_updates": completed,
        "amp_skipped_updates": skipped,
        "grad_scaler_enabled": scaler_enabled,
        "loss_checks": EXPECTED_PROFILE_ATTEMPTS,
        "loss_finite": True,
        "gradient_finite": nonfinite == 0,
        "gradient_check_basis": "complete pre-clip GradStats gradient norms",
        "gradient_stats_steps": gradient_stats_steps,
        "gradient_stats_nonfinite_steps": nonfinite,
        "valid_for_throughput": (
            skipped == 0 and nonfinite == 0 and gradient_stats_steps == EXPECTED_PROFILE_ATTEMPTS
        ),
    }
    return (
        'PROFILE {"step_wall": 12.5, "gpu_total": 8.0}\n'
        f"PROFILE_UPDATE_EVIDENCE {json.dumps(evidence)}\n"
    )


def test_actual_fp32_recipe_parsers_accept_both_profiles_fits_and_predictions() -> None:
    checks = validate_recipe_cli()

    assert len(checks) == 6
    assert all(check["status"] == "parsed" for check in checks)
    assert all(check["script_token_removed"] is True for check in checks)
    for mode in RUN_NAMES:
        for profile in (False, True):
            argv = trainer_argv(mode, profile=profile)
            assert argv[argv.index("--precision") + 1] == "fp32"
            assert argv[argv.index("--amp-init-scale") + 1] == "1"
            assert "--grad-stats" in argv
        assert predict_argv(mode)[1] == RUN_NAMES[mode]


def test_disabled_scaler_profile_requires_exact_finite_grad_stats() -> None:
    step_wall, evidence = validate_profile(parse_trainer_output(_fp32_profile_output()))

    assert step_wall == 12.5
    assert evidence["grad_scaler_enabled"] is False
    assert evidence["optimizer_completed_updates"] == EXPECTED_PROFILE_ATTEMPTS
    assert evidence["gradient_stats_steps"] == EXPECTED_PROFILE_ATTEMPTS
    assert evidence["gradient_stats_nonfinite_steps"] == 0
    assert evidence["gradient_finite"] is True


@pytest.mark.parametrize(
    "output",
    [
        _fp32_profile_output(skipped=1),
        _fp32_profile_output(nonfinite=1),
        _fp32_profile_output(gradient_stats_steps=EXPECTED_PROFILE_ATTEMPTS - 1),
        _fp32_profile_output(scaler_enabled=True),
    ],
)
def test_disabled_scaler_profile_rejects_skips_nonfinite_or_incomplete_evidence(
    output: str,
) -> None:
    with pytest.raises(ValueError, match="finite FP32 updates"):
        validate_profile(parse_trainer_output(output))


def test_finite_tensor_validator_checks_each_named_parameter() -> None:
    tensors = {"weight": 1.25, "bias": -0.5}

    assert validate_finite_tensors(tensors, lambda value: math.isfinite(value)) == 2
    with pytest.raises(ValueError, match="weight"):
        validate_finite_tensors({"weight": float("nan")}, lambda value: math.isfinite(value))
    with pytest.raises(ValueError, match="no model tensors"):
        validate_finite_tensors({}, lambda value: True)


def test_checkpoint_finiteness_covers_model_and_fixed_readout_tensors() -> None:
    checkpoint = {
        "model": {"weight": 1.25},
        "readout": [2.5, -0.5],
    }

    assert checkpoint_named_tensors(checkpoint, snapshot_name="model_ep0") == {
        "model_ep0.model.weight": 1.25,
        "model_ep0.readout.0": 2.5,
        "model_ep0.readout.1": -0.5,
    }
    with pytest.raises(ValueError, match="no model state mapping"):
        checkpoint_named_tensors({"model": {}}, snapshot_name="model")
    with pytest.raises(ValueError, match="readout is malformed"):
        checkpoint_named_tensors({"model": {"weight": 1.0}, "readout": 1.0}, snapshot_name="model")


def test_prediction_finiteness_requires_exact_fp32_shape_and_values() -> None:
    np = pytest.importorskip("numpy")
    predictions = np.zeros((2, 20_000, 2), dtype=np.float32)

    assert validate_finite_prediction_array(predictions, (2, 20_000, 2)) == {
        "shape": [2, 20_000, 2],
        "dtype": "float32",
        "finite": True,
    }
    with pytest.raises(ValueError, match="wrong shape or is not FP32"):
        validate_finite_prediction_array(predictions.astype(np.float64), (2, 20_000, 2))
    with pytest.raises(ValueError, match="wrong shape or is not FP32"):
        validate_finite_prediction_array(predictions, (2, 20_000, 1))
    predictions[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_finite_prediction_array(predictions, (2, 20_000, 2))


def test_full_fit_requires_exact_observed_gradient_count_and_no_nonfinite_steps() -> None:
    update = {
        "optimizer_attempts": 72_996,
        "optimizer_completed_updates": 72_996,
        "amp_skipped_updates": 0,
        "grad_stats": {"steps": 72_996, "nonfinite_steps": 0},
    }

    validate_fit_update_contract(update, 72_996)
    for field, value in (
        ("optimizer_completed_updates", 72_995),
        ("amp_skipped_updates", 1),
    ):
        failed = {**update, field: value}
        with pytest.raises(ValueError, match="exact finite FP32"):
            validate_fit_update_contract(failed, 72_996)
    with pytest.raises(ValueError, match="exact finite FP32"):
        validate_fit_update_contract(
            {**update, "grad_stats": {"steps": 72_995, "nonfinite_steps": 0}},
            72_996,
        )
    with pytest.raises(ValueError, match="exact finite FP32"):
        validate_fit_update_contract(
            {**update, "grad_stats": {"steps": 72_996, "nonfinite_steps": 1}},
            72_996,
        )


def test_root_conditional_authorization_binds_fp32_packet_recipe_and_subset(
    tmp_path: Path,
) -> None:
    expected = {
        "packet_id": PACKET_ID,
        "code_manifest_sha256": "a" * 64,
        "recipe_sha256": "b" * 64,
        "subset_manifest_sha256": "c" * 64,
        "kernel_process_wall_seconds_max": 6000,
        "worker_wall_seconds_max": 5200,
    }
    receipt = {
        "schema": "wnn.e540-conditional-fit-authorization.v1",
        "owner": "root",
        "status": "approved",
        "authorized": True,
        "conditional_fit_after_profile_gate": True,
        **expected,
        "authorization_id": "root-reviewed-v7-fp32",
    }
    path = tmp_path / AUTHORIZATION_PATH.name
    path.write_text(json.dumps(receipt), encoding="utf-8")

    assert validate_conditional_authorization(path, expected) == receipt
    receipt["recipe_sha256"] = "d" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind recipe_sha256"):
        validate_conditional_authorization(path, expected)


def test_pending_authorization_stops_startup_before_mounted_input_hashing(
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
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(e540_v7_worker, "RUNTIME", runtime)
    monkeypatch.setattr(e540_v7_worker, "AUTHORIZATION_PATH", receipt_path)
    monkeypatch.setenv("WNN_KERNEL_DEADLINE_MONOTONIC", str(time.monotonic() + 6000))
    monkeypatch.setenv("WNN_E540_PACKET_ID", PACKET_ID)
    monkeypatch.setenv("WNN_E540_CODE_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("WNN_E540_RECIPE_SHA256", "b" * 64)
    monkeypatch.setenv("WNN_E540_SUBSET_SHA256", "c" * 64)
    monkeypatch.setenv("WNN_E540_AUTHORIZATION_PATH", receipt_path.as_posix())
    monkeypatch.setenv("WNN_E540_KERNEL_WALL_SECONDS", "6000")
    monkeypatch.setenv("WNN_E540_WORKER_WALL_SECONDS", "5200")

    def reject_data_hash() -> None:
        raise AssertionError("pending authorization must stop before mounted-data hashing")

    monkeypatch.setattr(e540_v7_worker, "verify_mounted_data", reject_data_hash)
    with pytest.raises(ValueError, match="missing valid status"):
        e540_v7_worker.run()

    preflight = json.loads((runtime / "startup_preflight.json").read_text())
    assert preflight["status"] == "failed_before_mounted_data_hashing"
    assert len(preflight["cli_checks"]) == 6


def test_fp32_fit_update_projection_matches_frozen_membership() -> None:
    assert SUBSET_MANIFEST.name == "E540_FIT_SUBSET_V5.json"
    assert expected_fit_updates(1597, 252) == 72_996
