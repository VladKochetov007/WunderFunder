from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
H218_DIR = ROOT / "research/campaign_2026-10-06/bandit_continuation/attention_training_H218"
EXPORTER_PATH = H218_DIR / "export_fold0_eval.py"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(H218_DIR))

import grade_fold0_reference_blends as grader  # noqa: E402

from wnn.gmetric import global_wp  # noqa: E402


def test_alignment_joins_reordered_rows_by_valid_source_and_group() -> None:
    aligned = grader.align_evaluation_rows(
        [3, 8], candidate_groups=[8, 3], reference_groups=[9, 8, 3], target_groups=[3, 8, 9]
    )

    assert aligned == [(("valid", 3), 1, 2, 0), (("valid", 8), 0, 1, 1)]


def test_source_qualified_ids_allow_equal_train_and_valid_integers() -> None:
    split = {
        "fit": {"train": [4], "valid": [6]},
        "evaluation": {"valid": [4]},
        "sequences": {
            "train": [{"row_group": 4, "component": 1}],
            "valid": [
                {"row_group": 4, "component": 2},
                {"row_group": 6, "component": 3},
            ],
        },
    }
    provenance = {
        "fit_row_groups": {"train": [4], "valid": [6]},
        "evaluation_row_groups": [4],
    }

    result = grader.validate_candidate_fit_membership(split, provenance, [4])

    assert result["fit_evaluation_source_qualified_overlap"] == 0
    assert result["fit_evaluation_component_overlap"] == 0
    assert grader.source_group_index("train", [4]) != grader.source_group_index("valid", [4])


def test_candidate_valid_fit_overlap_is_rejected() -> None:
    split = {
        "fit": {"train": [4], "valid": [4]},
        "evaluation": {"valid": [4]},
        "sequences": {
            "train": [{"row_group": 4, "component": 1}],
            "valid": [{"row_group": 4, "component": 2}],
        },
    }
    provenance = {
        "fit_row_groups": {"train": [4], "valid": [4]},
        "evaluation_row_groups": [4],
    }

    with pytest.raises(ValueError, match="source-qualified evaluation group"):
        grader.validate_candidate_fit_membership(split, provenance, [4])


def test_candidate_component_overlap_is_rejected_across_sources() -> None:
    split = {
        "fit": {"train": [4], "valid": []},
        "evaluation": {"valid": [9]},
        "sequences": {
            "train": [{"row_group": 4, "component": 12}],
            "valid": [{"row_group": 9, "component": 12}],
        },
    }
    provenance = {
        "fit_row_groups": {"train": [4], "valid": []},
        "evaluation_row_groups": [9],
    }

    with pytest.raises(ValueError, match="component overlap"):
        grader.validate_candidate_fit_membership(split, provenance, [9])


def test_prediction_validation_rejects_nonfinite_rows() -> None:
    prediction = np.zeros((2, 8, 2), dtype=np.float32)
    prediction[1, 7, 1] = np.nan

    with pytest.raises(ValueError, match="nonfinite"):
        grader.validate_prediction_array(prediction, expected_rows=2, expected_timesteps=8)


def test_missing_root_admission_blocks_all_scoring() -> None:
    with pytest.raises(FileNotFoundError, match="not authorized"):
        grader.validate_root_admission(
            ROOT,
            ROOT / "missing_h218_grade_admission.json",
            Path(__file__),
            grader.PREREG,
        )


def test_frozen_preregistration_uses_top_level_fit_attempt_id() -> None:
    prereg = json.loads(grader.PREREG.read_text())

    assert grader.sha256_file(grader.PREREG) == grader.EXPECTED_PREREG_SHA256
    assert "fit_attempt_id" not in prereg["candidate"]
    assert grader.preregistered_fit_attempt_id(prereg) == prereg["fit_attempt_id"]


def test_actual_bandit_terminal_record_pairs_with_root_normalization() -> None:
    attempt_id = "h218-fit-attempt-fixture"
    root_attestation = {
        "fit_attempt_id": attempt_id,
        "exit_code": 0,
        "terminal_status": "success",
        "checkpoint_sha256": "a" * 64,
    }
    actual_attempt_record = {
        "id": attempt_id,
        "status": "finished",
        "exit_status": 0,
        "phase": "terminal",
    }

    assert "terminal_status" not in actual_attempt_record
    grader.validate_root_terminal_attestation(
        root_attestation,
        actual_attempt_record,
        attempt_id,
        attempt_id,
        "fit_terminal_receipt",
    )
    failed_attempt = {**actual_attempt_record, "status": "running"}
    with pytest.raises(ValueError, match="successful terminal bandit attempt"):
        grader.validate_root_terminal_attestation(
            root_attestation,
            failed_attempt,
            attempt_id,
            attempt_id,
            "fit_terminal_receipt",
        )


def test_export_attempt_identity_is_admitted_separately_from_fit() -> None:
    fit_attempt_id = grader.preregistered_fit_attempt_id(json.loads(grader.PREREG.read_text()))
    export_id = "h218-export-attempt-fixture"
    root_attestation = {
        "fit_attempt_id": fit_attempt_id,
        "attempt_id": export_id,
        "exit_code": 0,
        "terminal_status": "success",
        "checkpoint_sha256": "a" * 64,
        "prediction_sha256": "b" * 64,
        "sidecar_sha256": "c" * 64,
        "source_lock_sha256": "d" * 64,
    }
    actual_attempt_record = {
        "id": export_id,
        "status": "finished",
        "exit_status": 0,
        "phase": "terminal",
    }

    assert export_id != fit_attempt_id
    grader.validate_root_terminal_attestation(
        root_attestation,
        actual_attempt_record,
        fit_attempt_id,
        export_id,
        "export_terminal_receipt",
    )


def test_h218_sidecar_fixture_matches_exporter_metadata_layout() -> None:
    exporter_tree = ast.parse(EXPORTER_PATH.read_text())
    metadata_nodes = [
        node.value
        for node in ast.walk(exporter_tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "metadata" for target in node.targets)
        and isinstance(node.value, ast.Dict)
    ]
    assert len(metadata_nodes) == 1
    producer_keys = {
        key.value
        for key in metadata_nodes[0].keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    groups = [4, 8, 12]
    sidecar_fixture = {
        "schema": "h218-fold0-feature-only-prediction-v1",
        "run": "h218_fixture",
        "fit_attempt_id": "h218-fit-attempt-fixture",
        "epoch": 3,
        "checkpoint_sha256": "a" * 64,
        "prediction_sha256": "b" * 64,
        "prediction_path": "cache/pred_h218_fixture.npy",
        "prediction_shape": [len(groups), 20_000, 2],
        "prediction_dtype": "float32",
        "prediction_bytes": 128,
        "split_manifest_path": "research/fold0.json",
        "split_manifest_sha256": "c" * 64,
        "evaluation_group_count": len(groups),
        "evaluation_row_groups": groups,
        "evaluation_group_ids_sha256": "d" * 64,
        "labels_read": False,
        "targets_read": False,
        "inference_precision": "fp32",
        "sequence_state_reset": "batch_size=1 starts state=None for every sequence; state carries only across that sequence's chunks",
        "independent_grading_only": True,
        "official_score_computed": False,
        "source_sha256": {"exporter.py": "e" * 64, "source_lock": "f" * 64},
        "timing": {"prediction_and_feature_io_seconds": 1.0},
    }
    assert set(sidecar_fixture) == producer_keys
    grader.validate_export_sidecar(
        sidecar_fixture,
        {
            "run": "h218_fixture",
            "fit_attempt_id": "h218-fit-attempt-fixture",
            "epoch": 3,
            "prediction_shape": [3, 20_000, 2],
            "prediction_dtype": "float32",
        },
        groups,
    )

    invalid_fixture = {**sidecar_fixture, "targets_read": True}
    with pytest.raises(ValueError, match="read scoring labels"):
        grader.validate_export_sidecar(invalid_fixture, {}, groups)


def test_raw_blend_occurs_before_metric_clipping() -> None:
    reference = np.asarray([[3.0]], dtype=np.float32)
    candidate = np.asarray([[-1.0]], dtype=np.float32)

    blend = grader.raw_blend(reference, candidate, alpha=0.5)

    assert blend.item() == pytest.approx(1.0)
    assert np.clip(blend, -2.0, 2.0).item() == pytest.approx(1.0)
    assert (
        0.5 * np.clip(reference, -2.0, 2.0) + 0.5 * np.clip(candidate, -2.0, 2.0)
    ).item() == pytest.approx(0.5)


def test_pooled_scores_match_official_metric_and_bootstrap_is_paired() -> None:
    rng = np.random.default_rng(7)
    targets = rng.normal(0.0, 0.8, size=(5, 128, 2)).astype(np.float32)
    reference = (0.6 * targets + rng.normal(0.0, 0.7, targets.shape)).astype(np.float32)
    candidate = (0.7 * targets + rng.normal(0.0, 0.9, targets.shape)).astype(np.float32)
    targets[0, 0, 0] = np.nan  # Unscored warmup rows are outside the official metric mask.
    masks = rng.random((5, 128)) > 0.17
    aligned = [(("valid", index), index, index, index) for index in range(5)]
    components = np.asarray([3, 3, 5, 6, 6], dtype=np.int64)
    scorer = grader._load_module("h218_test_campaign_score", grader.SCORER_PATH)
    pair_helpers = grader._load_module("h218_test_pair_helpers", grader.H217_REFERENCE_HELPERS)

    result = grader.score_predeclared_blends(
        targets,
        masks,
        candidate,
        reference,
        aligned,
        components,
        scorer,
        pair_helpers,
        n_boot=80,
        bootstrap_seed=271828,
    )
    blends = cast(list[dict[str, object]], result["predeclared_blends_all_reported"])
    assert [entry["alpha_on_h218"] for entry in blends] == [0.0, 0.1, 0.25, 0.5, 1.0]
    need_prediction = np.arange(targets.shape[1]) >= 99
    selected = masks & need_prediction[None, :]
    for entry in blends:
        alpha = cast(float, entry["alpha_on_h218"])
        raw = grader.raw_blend(reference, candidate, alpha)
        direct = global_wp(targets, raw, selected)
        score = cast(dict[str, float], entry["score"])
        assert score["wp"] == pytest.approx(cast(float, direct["wp"]), abs=1e-12)
        bootstrap = cast(dict[str, object], entry["paired_component_bootstrap_vs_reference"])
        assert bootstrap["n_components"] == 3
        assert bootstrap["n_boot"] == 80
        assert bootstrap["metric_id"] == "global_wp_clip2_v1"

    baseline_bootstrap = cast(
        dict[str, object], blends[0]["paired_component_bootstrap_vs_reference"]
    )
    assert baseline_bootstrap["mean_diff"] == pytest.approx(0.0, abs=1e-12)
    assert len(cast(list[float], baseline_bootstrap["ci95"])) == 2
    repeated = grader.score_predeclared_blends(
        targets,
        masks,
        candidate,
        reference,
        aligned,
        components,
        scorer,
        pair_helpers,
        n_boot=80,
        bootstrap_seed=271828,
    )
    repeated_blends = cast(list[dict[str, object]], repeated["predeclared_blends_all_reported"])
    for first, second in zip(blends, repeated_blends, strict=True):
        first_bootstrap = cast(dict[str, object], first["paired_component_bootstrap_vs_reference"])
        second_bootstrap = cast(
            dict[str, object], second["paired_component_bootstrap_vs_reference"]
        )
        assert first_bootstrap == second_bootstrap
