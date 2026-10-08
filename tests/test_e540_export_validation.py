from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.e540_export_validation import (
    CHECKPOINT_WRITE_ALLOWANCE_SECONDS,
    validate_fit_artifact_metadata,
    validate_history_provenance_duration,
    validate_pair_metadata,
)
from scripts.e540_kaggle_v10_export_worker import validate_authorization, validate_prediction_clis

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/e540_v8_v9_fit_metadata"
SUBSET_PATH = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/E540_FIT_SUBSET_V5.json"
)


def read_fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def subset_with_identity() -> dict[str, object]:
    subset = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    import hashlib

    subset["_sha256"] = hashlib.sha256(SUBSET_PATH.read_bytes()).hexdigest()
    return subset


def validate_arm(mode: str) -> tuple[dict[str, object], dict[str, object]]:
    history = read_fixture(f"{mode}_history.json")
    provenance = read_fixture(f"{mode}_provenance.json")
    assert isinstance(provenance, dict)
    run_name = "e540v8-linear-s540-fp32" if mode == "linear" else "e540v9-ple-s540-fp32"
    hashes = provenance["checkpoint_sha256"]
    validate_fit_artifact_metadata(
        mode,
        history,
        provenance,
        subset_with_identity(),
        model_sha256=hashes["model.pt"],
        epoch0_sha256=hashes["model_ep0.pt"],
        expected_run_name=run_name,
    )
    return history[0], provenance


def test_actual_v8_and_v9_fit_metadata_are_a_matched_pair() -> None:
    _, linear = validate_arm("linear")
    _, ple = validate_arm("ple")

    assert linear["fitting_seconds"] == 421.4
    assert ple["fitting_seconds"] == 559.8
    assert validate_pair_metadata(linear, ple)


def test_actual_v9_pre_save_history_and_post_save_provenance_are_accepted() -> None:
    history = read_fixture("ple_history.json")
    provenance = read_fixture("ple_provenance.json")
    assert history[0]["train_seconds"] == 559.7
    assert provenance["fitting_seconds"] == 559.8

    result = validate_history_provenance_duration(
        history[0]["train_seconds"], provenance["fitting_seconds"]
    )

    assert result["observed_post_history_seconds"] == pytest.approx(0.1)
    assert result["checkpoint_write_allowance_seconds"] == CHECKPOINT_WRITE_ALLOWANCE_SECONDS


@pytest.mark.parametrize(
    ("history_seconds", "provenance_seconds"),
    [
        (100.0, 131.0),  # checkpoint save interval exceeds the declared 30s bound
        (100.0, 99.0),  # post-save provenance cannot precede history by a full second
        (float("nan"), 100.0),
        (100.0, float("inf")),
        (0.0, 0.1),
        (True, 1.0),
    ],
)
def test_invalid_or_unbounded_durations_are_rejected(
    history_seconds: object, provenance_seconds: object
) -> None:
    with pytest.raises(ValueError):
        validate_history_provenance_duration(history_seconds, provenance_seconds)


def test_fit_metadata_rejects_history_provenance_update_mismatch() -> None:
    history, provenance = read_fixture("ple_history.json"), read_fixture("ple_provenance.json")
    assert isinstance(provenance, dict)
    bad_history = copy.deepcopy(history)
    bad_history[0]["update_diagnostics"]["epoch"]["optimizer_completed_updates"] -= 1
    hashes = provenance["checkpoint_sha256"]

    with pytest.raises(ValueError, match="history and sibling provenance differ in epoch"):
        validate_fit_artifact_metadata(
            "ple",
            bad_history,
            provenance,
            subset_with_identity(),
            model_sha256=hashes["model.pt"],
            epoch0_sha256=hashes["model_ep0.pt"],
            expected_run_name="e540v9-ple-s540-fp32",
        )


def test_pair_metadata_rejects_changed_sampler_identity() -> None:
    _, linear = validate_arm("linear")
    _, ple = validate_arm("ple")
    changed = copy.deepcopy(ple)
    changed["update_diagnostics"][0]["sample_order_identity"]["seeded_order"]["seed"] += 1

    with pytest.raises(ValueError, match="different observed sampler identities"):
        validate_pair_metadata(linear, changed)


def test_both_prediction_commands_parse_against_real_cli_without_torch() -> None:
    checks = validate_prediction_clis()
    assert [check["mode"] for check in checks] == ["linear", "ple"]
    assert all(check["status"] == "parsed" for check in checks)
    assert all(check["feature_only_subset"] for check in checks)
    assert all(check["reader_requests_no_targets_or_mask"] for check in checks)


def test_export_authorization_is_fail_closed_and_hash_bound(tmp_path: Path) -> None:
    expected = {
        "packet_id": "E540_FP32_V10_FIXED_CHECKPOINT_EXPORT_ONLY",
        "artifact_binding_sha256": "a" * 64,
        "recipe_sha256": "b" * 64,
    }
    pending = {
        "schema": "wnn.e540-export-only-authorization.v1",
        "owner": "root",
        "status": "pending_root_review",
        "authorized": False,
        "export_only": True,
        "fit_allowed": False,
        "provider_launch_authorized": False,
        **expected,
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(pending))
    with pytest.raises(ValueError, match="missing valid status"):
        validate_authorization(path, expected)

    approved = {
        **pending,
        "status": "approved",
        "authorized": True,
        "provider_launch_authorized": True,
        "authorization_id": "root-export-v1",
    }
    path.write_text(json.dumps(approved))
    assert validate_authorization(path, expected)["authorization_id"] == "root-export-v1"
    approved["artifact_binding_sha256"] = "c" * 64
    path.write_text(json.dumps(approved))
    with pytest.raises(ValueError, match="does not bind artifact_binding_sha256"):
        validate_authorization(path, expected)
