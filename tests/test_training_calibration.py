from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from tests.test_trainer_resume_integration import _repository, training
from wnn import data
from wnn.propensity import PropensityNet
from wnn.training_calibration import mass_observation_artifact, summarize_masses


def test_weight_path_uses_joint_mask_scale_before_since_weight() -> None:
    labels = torch.ones(3, 4, 2)
    valid = torch.tensor([[False, True, True, True]]).expand(3, -1)
    scored = torch.tensor(
        [[False, False, False, False], [False, True, False, True], [False, True, False, False]]
    )
    probability = torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9], [0.3, 0.4, 0.5, 0.6]])
    inputs = torch.zeros(3, 4, data.N_FEATURES + training.N_EVENT_FEATURES)
    inputs[..., data.N_FEATURES + training.FEATURE_NAMES.index("since_i0px")] = torch.log1p(
        torch.arange(4, dtype=torch.float32)
    )
    args = argparse.Namespace(
        mask_weight=False,
        prop_cache=True,
        prop_power=2.0,
        prop_eps=0.03,
        class_weight="",
        valid_true_mask=0.03,
        sync_free=True,
        since_weight=2.0,
        since_floor=0.25,
        episode_weight=0.0,
    )
    result = training.main_row_weights(inputs, labels, valid, scored, object(), probability, args)
    expected = valid.float() * (probability.square() + 0.03)
    scale = expected[1:].sum() / (valid[1:] * (scored[1:].float() + 0.03)).sum()
    expected[1:] = valid[1:] * (scored[1:].float() + 0.03) * scale
    expected *= (1 / (1 + torch.arange(4) / 2.0)).clamp_min(0.25)
    torch.testing.assert_close(result, expected, rtol=3e-7, atol=0)


def test_actual_main_calibrates_mixed_pool_and_writes_no_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository(tmp_path)
    valid_table = pq.read_table(tmp_path / "datasets/valid.parquet")
    pq.write_table(
        valid_table.drop(["is_scored"]), tmp_path / "datasets/train.parquet", row_group_size=8
    )
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(data, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(training, "WARMUP", 1)
    monkeypatch.setattr(training, "CORRUPT_TRAIN_ROW_GROUPS", set(range(5, 10607)))
    monkeypatch.setattr(training, "valid_row_groups", lambda path: (np.arange(4), np.array([4])))
    monkeypatch.setattr(training, "dev_halves", lambda path: (np.arange(2), np.arange(2, 4)))
    monkeypatch.setenv("WNN_LOADER_WORKERS", "0")
    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)
    model = PropensityNet(data.N_FEATURES + training.N_EVENT_FEATURES, hidden=2)
    parent = tmp_path / "artifacts/runs/prop/model.pt"
    parent.parent.mkdir(parents=True)
    torch.save({"n_in": model.mean.numel(), "hidden": 2, "model": model.state_dict()}, parent)
    for split, probability in (("train", 0.2), ("valid", 0.8)):
        path = tmp_path / "cache" / f"propcache_prop_{split}.npy"
        np.save(path, np.full((5, 8), probability, dtype=np.float32))
        path.with_suffix(".json").write_text(
            json.dumps({"model_sha256": training.file_sha256(parent)})
        )
    report_path = tmp_path / "reference.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rnn.py",
            "--name",
            "calibrate",
            "--device",
            "cpu",
            "--train-source",
            "train+valid",
            "--train-seqs",
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
            "--propensity",
            "prop",
            "--prop-cache",
            "--sync-free",
            "--prop-power",
            "2",
            "--prop-eps",
            ".03",
            "--valid-true-mask",
            ".03",
            "--since-weight",
            "32",
            "--since-floor",
            ".25",
            "--calibrate-distill-reference",
            str(report_path),
        ],
    )
    training.main()
    report = json.loads(report_path.read_text())
    # Batch true-mask scaling preserves each batch's total propensity mass here:
    # all target magnitudes are equal and every raw price changes (since=0).
    expected = 7 * (2 * (0.2**2 + 0.03) + 4 * (0.8**2 + 0.03)) * (0.12543 + 0.25086) / 6
    assert report["reference_mass"] == pytest.approx(expected, rel=2e-7)
    assert report["updates"] == 6
    observation = report["update_masses"]
    values = np.asarray(observation["values"], dtype="<f8")
    assert observation["count"] == 6
    assert observation["sha256"] == hashlib.sha256(values.tobytes()).hexdigest()
    assert values.mean() == pytest.approx(report["reference_mass"], rel=1e-12)
    assert report["sample_order"][0]["unique_retained_count"] == 6
    assert report["contract"]["fit_membership"]["train"]["count"] == 2
    assert report["contract"]["fit_membership"]["valid"]["count"] == 4
    assert "propcache_prop_train.npy" in report["contract"]["consumed_caches"]
    assert not (tmp_path / "artifacts/runs/calibrate/model.pt").exists()


@pytest.mark.parametrize("values", [[], [float("nan")], [-1.0], [0.0]])
def test_invalid_calibration_rejected(values: list[float]) -> None:
    with pytest.raises(ValueError):
        summarize_masses(values)


def test_update_mass_artifact_binds_float64_le_observations() -> None:
    artifact = mass_observation_artifact([1.0, 0.0, 2.5])
    values = np.asarray(artifact["values"], dtype="<f8")
    assert artifact["format"] == "wnn.float64-le-update-masses"
    assert artifact["dtype"] == "<f8"
    assert artifact["count"] == 3
    assert artifact["sha256"] == hashlib.sha256(values.tobytes()).hexdigest()
    assert values.tolist() == [1.0, 0.0, 2.5]
