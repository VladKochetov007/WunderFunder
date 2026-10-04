from __future__ import annotations

import copy
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from tests.test_trainer_resume_integration import _repository, training
from wnn import data
from wnn.privileged import (
    inner_manifest,
    privileged_inputs,
    teacher_prediction_groups,
    validate_crossfit_cache,
    validate_inner_manifest,
    validate_teacher_checkpoint,
)
from wnn.splits import CampaignSplit

pipeline = importlib.import_module("privileged_crossfit")


def _outer(root: Path) -> CampaignSplit:
    components = tuple((split, group, group) for split in ("train", "valid") for group in range(5))
    return CampaignSplit(
        root / "outer.json", "a" * 64, 0, 4, tuple(range(4)), tuple(range(4)), (4,), components
    )


def test_origin_alignment_and_causal_equal_width_control() -> None:
    raw = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    current = privileged_inputs(raw, "current", 2)
    future = privileged_inputs(raw, "future", 2)
    assert current.shape == future.shape == raw.shape
    assert torch.equal(current, raw)
    assert torch.equal(future, raw[:, [2, 3, 4, 5, 5, 5]])
    changed = raw.clone()
    changed[:, 2] += 100
    assert not torch.equal(privileged_inputs(changed, "future", 2)[:, 0], future[:, 0])
    assert torch.equal(privileged_inputs(changed, "current", 2)[:, 0], current[:, 0])


def test_component_assignment_is_joint_across_train_and_valid(tmp_path: Path) -> None:
    outer = _outer(tmp_path)
    values = [inner_manifest(outer, fold, 2, 98) for fold in range(2)]
    for value in values:
        assert value["fit"]["train"] == value["fit"]["valid"]
        assert set(value["fit"]["train"]).isdisjoint(value["predict"]["valid"])
        assert 4 not in value["fit"]["train"] + value["predict"]["valid"]
        corrupt = copy.deepcopy(value)
        corrupt["fit"]["train"].append(value["predict"]["train"][0])
        with pytest.raises(ValueError, match="component-disjoint"):
            validate_inner_manifest(corrupt, outer)
    assert set(values[0]["predict"]["train"]) | set(values[1]["predict"]["train"]) == set(range(4))


@pytest.mark.parametrize("mode", ["none", "current", "future"])
def test_actual_teacher_fit_and_crossfit_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _repository(tmp_path)
    device = os.environ.get("WNN_INTEGRATION_DEVICE", "cpu")
    table = pq.read_table(tmp_path / "datasets/valid.parquet")
    pq.write_table(table.drop(["is_scored"]), tmp_path / "datasets/train.parquet", row_group_size=8)
    np.savez(tmp_path / "cache/train_targets.npz", row_group=np.arange(5), seq_ix=np.arange(5))
    outer = _outer(tmp_path)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(training, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(data, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(training, "WARMUP", 1)
    monkeypatch.setattr(training, "load_campaign_split", lambda path, root: outer)
    monkeypatch.setenv("WNN_LOADER_WORKERS", "0")
    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)
    directory = tmp_path / "inner"
    pipeline.prepare(outer, directory, 2, 98)
    checkpoints = []
    for fold in range(2):
        name = f"teacher_{fold}"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_rnn.py",
                "--name",
                name,
                "--device",
                device,
                "--train-source",
                "train+valid",
                "--split-manifest",
                str(outer.path),
                "--fit-groups-manifest",
                str(directory / f"inner{fold}.json"),
                "--privileged-mode",
                mode,
                "--privileged-horizon",
                "2",
                "--batch",
                "2",
                "--hidden",
                "4",
                "--layers",
                "1",
                "--chunk",
                "4",
                "--epochs",
                "1",
                "--checkpoint-policy",
                "fixed",
                "--no-eval-during-fit",
                "--loss",
                "wmse",
                "--seed",
                "3",
            ],
        )
        training.main()
        path = tmp_path / "artifacts/runs" / name / "model.pt"
        checkpoint = torch.load(path, weights_only=False)
        inner = validate_teacher_checkpoint(checkpoint, outer)
        assert (
            checkpoint["n_in"]
            == data.N_FEATURES * (1 if mode == "none" else 2) + training.N_EVENT_FEATURES
        )
        assert checkpoint["provenance"]["deployment_safe"] == (mode == "none")
        assert inner["predict"]["train"]
        corrupted = copy.deepcopy(checkpoint)
        corrupted["provenance"]["fit_row_groups"]["train"].append(inner["predict"]["train"][0])
        with pytest.raises(ValueError, match="membership"):
            validate_teacher_checkpoint(corrupted, outer)
        checkpoints.append(path)
    pipeline.extract(outer, checkpoints, "crossfit", torch.device(device), 1, 4)
    if mode == "none":
        pipeline.extract(
            outer, checkpoints, "fitted", torch.device(device), 1, 4, exposure="fitted"
        )
    else:
        with pytest.raises(ValueError, match="ordinary causal"):
            pipeline.extract(
                outer, checkpoints, "fitted", torch.device(device), 1, 4, exposure="fitted"
            )
    for split in ("train", "valid"):
        path = tmp_path / "cache" / f"pred_crossfit_{split}.npy"
        prediction = np.load(path)
        assert np.isfinite(prediction[:4]).all()
        assert np.isnan(prediction[4]).all()
        sidecar = json.loads(path.with_suffix(".json").read_text())
        validate_crossfit_cache(sidecar, outer, tmp_path, split)
        assert sidecar["prediction_sha256"] == training.file_sha256(path)
        corrupt = copy.deepcopy(sidecar)
        corrupt["teachers"][0]["predicted_groups"][split].append(4)
        with pytest.raises(ValueError, match="misassigned"):
            validate_crossfit_cache(corrupt, outer, tmp_path, split)
        if mode == "none":
            fitted_path = tmp_path / "cache" / f"pred_fitted_{split}.npy"
            fitted_side = json.loads(fitted_path.with_suffix(".json").read_text())
            validate_crossfit_cache(fitted_side, outer, tmp_path, split)
            assert np.isfinite(np.load(fitted_path)[:4]).all()
            assert np.isnan(np.load(fitted_path)[4]).all()
            assert [entry["checkpoint_sha256"] for entry in fitted_side["teachers"]] == [
                entry["checkpoint_sha256"] for entry in sidecar["teachers"]
            ]
            for fit_teacher, excluding_teacher in zip(
                fitted_side["teachers"], sidecar["teachers"], strict=True
            ):
                fitted_groups = set(fit_teacher["predicted_groups"][split])
                excluded_groups = set(excluding_teacher["predicted_groups"][split])
                assert fitted_groups.isdisjoint(excluded_groups)
                assert fitted_groups | excluded_groups == set(range(4))
            corrupt = copy.deepcopy(fitted_side)
            corrupt["target_exposure"] = "excluded"
            with pytest.raises(ValueError, match="misassigned"):
                validate_crossfit_cache(corrupt, outer, tmp_path, split)
            corrupt["crossfit_format"] = "wnn.origin-teacher-crossfit.v1"
            corrupt["target_exposure"] = "fitted"
            with pytest.raises(ValueError, match="legacy"):
                validate_crossfit_cache(corrupt, outer, tmp_path, split)
    bound = training.bind_fold_teachers([("crossfit", 1.0)], outer, ("train", "valid"))
    assert len(bound) == 2
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rnn.py",
            "--name",
            "student",
            "--device",
            device,
            "--train-source",
            "train+valid",
            "--split-manifest",
            str(outer.path),
            "--distill",
            "crossfit",
            "--batch",
            "2",
            "--hidden",
            "4",
            "--layers",
            "1",
            "--chunk",
            "4",
            "--epochs",
            "1",
            "--checkpoint-policy",
            "fixed",
            "--no-eval-during-fit",
            "--loss",
            "wmse",
            "--seed",
            "3",
        ],
    )
    training.main()
    if mode == "none":
        student_argv = list(sys.argv)
        student_argv[student_argv.index("--name") + 1] = "student_fitted"
        student_argv[student_argv.index("--distill") + 1] = "fitted"
        monkeypatch.setattr(sys, "argv", student_argv)
        training.main()
    student = torch.load(tmp_path / "artifacts/runs/student/model.pt", weights_only=False)
    assert student["args"]["privileged_mode"] == "none"
    assert student["n_in"] == data.N_FEATURES + training.N_EVENT_FEATURES
    raw = torch.randn(1, 8, data.N_FEATURES)
    changed = raw.clone()
    changed[:, 4:] += 10
    assert torch.equal(training.build_inputs(raw)[:, :4], training.build_inputs(changed)[:, :4])


def test_fitted_exposure_refuses_nonpaired_fold_count(tmp_path: Path) -> None:
    outer = _outer(tmp_path)
    inner = inner_manifest(outer, 0, 2, 98)
    inner["folds"] = 3
    with pytest.raises(ValueError, match="exactly two"):
        teacher_prediction_groups({"args": {"privileged_mode": "none"}}, inner, "fitted")
