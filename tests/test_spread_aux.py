from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn.functional as functional

from tests.test_privileged_crossfit import _outer
from tests.test_trainer_resume_integration import _repository, training
from tests.test_training_state import _assert_equal
from wnn import data
from wnn.spread_aux import (
    file_sha256,
    structural_loss,
    validate_spread_cache,
    validate_spread_parent,
)
from wnn.training_state import load_training_state


def _cache(root: Path, split_hash: str) -> Path:
    directory = root / "cache/spread"
    directory.mkdir()
    (directory / "selection.json").write_text(json.dumps({"map_fit_train_groups": [0, 1]}))
    (directory / "interior_code_table.json").write_text("[]")
    metadata = {
        "status": "complete",
        "supervision_only": True,
        "N_SP_AUX": 12,
        "split_manifest_sha256": split_hash,
        "fit_train_groups": list(range(4)),
        "fit_valid_groups": list(range(4)),
        "evaluation_valid_groups": [4],
        "unknown": 255,
        "dtype": "uint8",
        "selection_sha256": file_sha256(directory / "selection.json"),
        "map_sha256": file_sha256(directory / "interior_code_table.json"),
        "files": {},
    }
    for split in ("train", "valid"):
        values = np.full((4, 8), 3, np.uint8)
        values[:, ::2] = 255
        path = directory / f"spread_{split}.npy"
        np.save(path, values)
        metadata["files"][split] = {
            "file": path.name,
            "shape": list(values.shape),
            "row_groups": list(range(4)),
            "sha256": file_sha256(path),
        }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return directory


def test_known_mask_preserves_branch_class_and_denominator() -> None:
    logits = torch.randn(1, 4, 19, generator=torch.Generator().manual_seed(101), requires_grad=True)
    weights = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    spread = torch.tensor([[2, 3, 4, 5]])
    branch, classes = torch.zeros_like(spread), torch.ones_like(spread)
    encoded = torch.tensor([[255, 3, 255, 5]], dtype=torch.uint8)
    actual = structural_loss(logits, spread, branch, classes, weights, 0, 12, encoded)
    expected = 0
    for lo, hi, labels, mask in (
        (0, 12, spread, encoded != 255),
        (12, 15, branch, torch.ones_like(spread)),
        (15, 19, classes, torch.ones_like(spread)),
    ):
        term = functional.cross_entropy(
            logits[..., lo:hi].reshape(-1, hi - lo), labels.ravel(), reduction="none"
        )
        expected = expected + (term * weights.ravel() * mask.ravel()).sum() / weights.sum()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert not logits.grad[0, [0, 2], :12].count_nonzero()
    assert logits.grad[0, [0, 2], 12:].count_nonzero()


@pytest.mark.parametrize("mode", ["proxy_known", "current_known"])
def test_modern_spread_main_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A nonhistorical parent/seed/fold and two-epoch continuation exercise real cache losses."""
    _repository(tmp_path)
    table = pq.read_table(tmp_path / "datasets/valid.parquet")
    pq.write_table(table.drop(["is_scored"]), tmp_path / "datasets/train.parquet", row_group_size=8)
    outer = _outer(tmp_path)._replace(fold=2)
    cache = _cache(tmp_path, outer.sha256)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(data, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(training, "WARMUP", 1)
    monkeypatch.setattr(training, "load_campaign_split", lambda path, root: outer)
    monkeypatch.setenv("WNN_LOADER_WORKERS", "0")
    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)
    device = os.environ.get("WNN_INTEGRATION_DEVICE", "cpu")
    base = [
        "train_rnn.py",
        "--device",
        device,
        "--train-source",
        "train+valid",
        "--split-manifest",
        str(outer.path),
        "--batch",
        "2",
        "--hidden",
        "4",
        "--layers",
        "1",
        "--chunk",
        "4",
        "--epochs",
        "2",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "1",
        "--no-eval-during-fit",
        "--loss",
        "wmse",
        "--aux-struct",
        ".3",
        "--aux-next",
        ".3",
        "--ema-teacher",
        "1",
        "--eval-ema",
        "--seed",
        "11",
        "--save-training-state",
    ]

    def run(name: str, extra: list[str]) -> dict:
        monkeypatch.setattr(sys, "argv", [*base, "--name", name, *extra])
        training.main()
        return load_training_state(tmp_path / "artifacts/runs" / name / "training_state.pt")

    run("modern_combo_parent", ["--seed", "7"])
    parent_path = tmp_path / "artifacts/runs/modern_combo_parent/model.pt"
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    base += ["--init", "modern_combo_parent", "--inherit-init-preprocessing"]
    run("original_continuation", [])
    base += ["--spread-aux-mode", mode, "--spread-cache-root", str(cache)]
    reference = run("reference", [])
    run("interrupted", ["--stop-after-epoch", "0"])
    resumed = run(
        "resumed",
        ["--resume-state", str(tmp_path / "artifacts/runs/interrupted/training_state.pt")],
    )
    for key in (
        "contract",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "ema",
        "loader",
        "rng",
        "auxiliary",
    ):
        _assert_equal(reference[key], resumed[key])
    assert reference["progress"]["step_count"] == resumed["progress"]["step_count"] == 16
    _assert_equal(parent["model"]["mean"], reference["model"]["mean"])
    _assert_equal(parent["model"]["std"], reference["model"]["std"])
    corrupt = copy.deepcopy(parent)
    corrupt["provenance"]["fit_row_groups"]["valid"].append(4)
    with pytest.raises(ValueError, match="membership"):
        validate_spread_parent(corrupt, outer.sha256, parent["provenance"]["fit_row_groups"])
    with pytest.raises(ValueError, match="matching clean"):
        validate_spread_parent(parent, "b" * 64, parent["provenance"]["fit_row_groups"])
    metadata_path = cache / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    (cache / "selection.json").write_text(json.dumps({"map_fit_train_groups": [4]}))
    metadata["selection_sha256"] = file_sha256(cache / "selection.json")
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="outside the fitting"):
        validate_spread_cache(cache, outer.sha256, range(4), range(4), [4], sequence_rows=8)
