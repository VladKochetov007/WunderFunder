"""Synthetic integrity and scoring checks for the fixed continuation grader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_bandit_continuation import (
    _blend_equal_pair,
    _expected_c0_runs,
    _official_mask,
    _validate_prediction_artifact,
    _validate_target_index,
    _verify_command_arguments,
    _verify_provenance_arguments,
    compare_arrays,
)
from wnn.gmetric import global_wp
from wnn.splits import file_sha256


def test_e502_preregistered_run_names_are_seed_bound() -> None:
    preregistration = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "research/campaign_2026-10-06/bandit_continuation/local/E502_C0_PAIRED_v2.json"
        ).read_text()
    )
    assert _expected_c0_runs(preregistration) == {
        "bd26_c0_s3_f0",
        "bd26_c0_s4_f0",
    }


def test_checkpoint_cli_arguments_are_bound_to_registered_command() -> None:
    command = [
        ".venv/bin/python",
        "scripts/train_rnn.py",
        "--name",
        "bd26_c0_s3_f0",
        "--seed",
        "3",
        "--lr",
        "0.0001",
        "--no-eval-during-fit",
        "--split-manifest",
        "research/campaign_2026-09-29/artifacts/splits/fold0.json",
    ]
    args = {
        "name": "bd26_c0_s3_f0",
        "seed": 3,
        "lr": 0.0001,
        "no_eval_during_fit": True,
        "split_manifest": "/absolute/fold0.json",
    }
    _verify_command_arguments(args, command, "bd26_c0_s3_f0")
    with pytest.raises(ValueError, match="argument lr differs"):
        _verify_command_arguments(args | {"lr": 0.0002}, command, "bd26_c0_s3_f0")


def test_recursive_lineage_uses_checkpoint_bound_arguments() -> None:
    args = {"init": "parent@ep3", "seed": 3}
    _verify_provenance_arguments(args, {"args": args}, {"args": args}, "candidate_s3_f0")
    with pytest.raises(ValueError, match="arguments disagree"):
        _verify_provenance_arguments(
            args,
            {"args": args},
            {"args": {"init": "different_parent@ep3", "seed": 3}},
            "candidate_s3_f0",
        )


def test_official_mask_requires_prediction_and_scored_flags() -> None:
    is_scored = np.ones((2, 20_000), dtype=bool)
    is_scored[0, 150] = False
    mask = _official_mask(is_scored)
    assert not mask[:, :99].any()
    assert mask[0, 99]
    assert not mask[0, 150]
    assert mask[1, 150]


def test_fixed_pair_blends_raw_members_before_official_clipping() -> None:
    first = np.zeros((2, 20_000, 2), dtype=np.float32)
    second = np.zeros_like(first)
    first[0, :2, 0] = [3.0, -3.0]
    second[0, :2, 0] = [1.0, -1.0]
    pair = _blend_equal_pair(first, second)
    assert pair.dtype == np.float32
    assert pair[0, :2, 0].tolist() == [2.0, -2.0]
    clipped_first = np.clip(first, -2.0, 2.0)
    clipped_second = np.clip(second, -2.0, 2.0)
    assert not np.array_equal(pair, (clipped_first + clipped_second) * 0.5)


def test_target_cache_index_must_match_full_frozen_valid_split() -> None:
    row_groups = [17, 41]
    targets = np.zeros((2, 20_000, 2), dtype=np.float32)
    scored = np.ones((2, 20_000), dtype=bool)
    market_class = np.array([2, 3], dtype=np.int64)
    _validate_target_index(row_groups, targets, scored, market_class, (41, 17))
    with pytest.raises(ValueError, match="frozen valid groups"):
        _validate_target_index(row_groups, targets, scored, market_class, (41, 52))
    with pytest.raises(ValueError, match="frozen valid groups"):
        _validate_target_index([17, 17], targets, scored, market_class, (17, 41))


def test_paired_stream_score_matches_official_global_metric() -> None:
    rng = np.random.default_rng(8012)
    targets = rng.normal(size=(4, 20_000, 2)).astype(np.float32)
    candidate = (0.7 * targets + rng.normal(size=targets.shape)).astype(np.float32)
    reference = (0.5 * targets + rng.normal(size=targets.shape)).astype(np.float32)
    candidate[0, 100, 0] = 4.0
    reference[1, 101, 1] = -4.0
    is_scored = rng.random((4, 20_000)) > 0.03
    mask = _official_mask(is_scored)
    components = np.array([0, 0, 1, 1], dtype=np.int64)

    result = compare_arrays(
        targets,
        candidate,
        reference,
        is_scored,
        components,
        n_boot=16,
        seed=123,
    )
    expected_candidate = global_wp(targets, candidate, mask)
    expected_reference = global_wp(targets, reference, mask)
    assert result["metric_id"] == "global_wp_clip2_v1"
    assert result["selected_rows"] == int(mask.sum())
    assert result["candidate_wp"] == pytest.approx(expected_candidate["wp"], abs=1e-6)
    assert result["reference_wp"] == pytest.approx(expected_reference["wp"], abs=1e-6)
    assert result["n_components"] == 2
    assert result["n_boot"] == 16


def test_prediction_artifact_binds_bytes_checkpoint_and_row_order(tmp_path: Path) -> None:
    prediction_path = tmp_path / "pred.npy"
    rows = [17, 41]
    expected = np.zeros((2, 20_000, 2), dtype=np.float32)
    np.save(prediction_path, expected)
    metadata = {
        "run": "candidate_s3_f0",
        "split": "valid",
        "epoch": 1,
        "split_manifest_sha256": "split-sha",
        "checkpoint_sha256": "checkpoint-sha",
        "prediction_sha256": file_sha256(prediction_path),
        "row_groups": rows,
        "inference_precision": "fp16",
    }
    memory_map, binding = _validate_prediction_artifact(
        prediction_path,
        metadata,
        run="candidate_s3_f0",
        epoch=1,
        split_sha256="split-sha",
        checkpoint_sha256="checkpoint-sha",
        row_groups=rows,
        shape=expected.shape,
    )
    assert isinstance(memory_map, np.memmap)
    assert binding["prediction_sha256"] == metadata["prediction_sha256"]

    bad_rows = metadata | {"row_groups": list(reversed(rows))}
    with pytest.raises(ValueError, match="row order"):
        _validate_prediction_artifact(
            prediction_path,
            bad_rows,
            run="candidate_s3_f0",
            epoch=1,
            split_sha256="split-sha",
            checkpoint_sha256="checkpoint-sha",
            row_groups=rows,
            shape=expected.shape,
        )

    bad_checkpoint = metadata | {"checkpoint_sha256": "wrong-checkpoint"}
    with pytest.raises(ValueError, match="fixed checkpoint"):
        _validate_prediction_artifact(
            prediction_path,
            bad_checkpoint,
            run="candidate_s3_f0",
            epoch=1,
            split_sha256="split-sha",
            checkpoint_sha256="checkpoint-sha",
            row_groups=rows,
            shape=expected.shape,
        )

    with prediction_path.open("r+b") as stream:
        stream.seek(-4, 2)
        stream.write(np.array([np.nan], dtype=np.float32).tobytes())
    changed = metadata | {"prediction_sha256": file_sha256(prediction_path)}
    with pytest.raises(ValueError, match="nonfinite"):
        _validate_prediction_artifact(
            prediction_path,
            changed,
            run="candidate_s3_f0",
            epoch=1,
            split_sha256="split-sha",
            checkpoint_sha256="checkpoint-sha",
            row_groups=rows,
            shape=expected.shape,
        )
