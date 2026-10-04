from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tests.test_training_state import _assert_equal
from wnn import data
from wnn.training_state import load_training_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_rnn as training  # noqa: E402


def _repository(root: Path) -> None:
    (root / "src").symlink_to(Path(__file__).resolve().parents[1] / "src", target_is_directory=True)
    (root / "datasets").mkdir()
    (root / "cache").mkdir()
    rng = np.random.default_rng(71)
    values = rng.normal(size=(5, 8, data.N_FEATURES)).astype("float32")
    targets = np.tile(np.array([[-0.12543, 0.25086]], dtype="float32"), (5, 8, 1))
    targets[:, 4:, :] *= -1
    tables = []
    for group in range(5):
        columns = {name: values[group, :, index] for index, name in enumerate(data.FEATURE_COLUMNS)}
        columns.update(
            seq_ix=np.full(8, group),
            t0=targets[group, :, 0],
            t1=targets[group, :, 1],
            is_scored=np.arange(8) % 2 == 1,
        )
        tables.append(pa.table(columns))
    pq.write_table(pa.concat_tables(tables), root / "datasets/valid.parquet", row_group_size=8)
    np.savez(root / "cache/valid_targets.npz", row_group=np.arange(5), seq_ix=np.arange(5))
    np.save(root / "cache/pred_teacher_valid.npy", targets * 0.7)


@pytest.mark.parametrize("controlled", [False, True])
def test_real_main_epoch_resume_and_completed_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, controlled: bool
) -> None:
    """Execute actual main, parquet reads, normalization, all P0 losses and exports."""
    _repository(tmp_path)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(data, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(training, "WARMUP", 1)
    monkeypatch.setattr(training, "valid_row_groups", lambda path: (np.arange(4), np.array([4])))
    monkeypatch.setattr(training, "dev_halves", lambda path: (np.arange(2), np.arange(2, 4)))
    monkeypatch.setenv("WNN_LOADER_WORKERS", "0")
    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)
    base = [
        "train_rnn.py",
        "--device",
        "cpu",
        "--train-source",
        "valid",
        "--full-valid",
        "--batch",
        "2",
        "--hidden",
        "4",
        "--layers",
        "1",
        "--chunk",
        "4",
        "--epochs",
        "3",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "2",
        "--no-eval-during-fit",
        "--loss",
        "wmse",
        "--distill",
        "teacher",
        "--aux-struct",
        ".3",
        "--aux-next",
        ".3",
        "--ema-teacher",
        "1",
        "--eval-ema",
        "--sync-free",
        "--seed",
        "3",
        "--save-training-state",
    ]
    if controlled:
        base += [
            "--chunk",
            "2",
            "--update-rows",
            "4",
            "--detach-rows",
            "2",
            "--distill-boot-h",
            "1",
            "--ema-reference-rows",
            "4",
            "--optim",
            "muon",
            "--optim-switch",
            "0.5:adamw",
            "--grad-noise",
            "1e-6",
            "--batch-order",
            "seeded",
            "--antithetic",
        ]

    def run(name: str, extra: list[str]) -> Path:
        monkeypatch.setattr(sys, "argv", [*base, "--name", name, *extra])
        training.main()
        return tmp_path / "artifacts/runs" / name

    reference_dir = run("reference", [])
    interrupted_dir = run("interrupted", ["--stop-after-epoch", "0"])
    resumed_dir = run("resumed", ["--resume-state", str(interrupted_dir / "training_state.pt")])
    reference = load_training_state(reference_dir / "training_state.pt")
    resumed = load_training_state(resumed_dir / "training_state.pt")
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
    assert reference["progress"]["step_count"] == resumed["progress"]["step_count"] == 12
    for first, second in zip(
        reference["progress"]["update_diagnostics"],
        resumed["progress"]["update_diagnostics"],
        strict=True,
    ):
        _assert_equal(first, second)
    recovered_dir = run("recovered", ["--resume-state", str(resumed_dir / "training_state.pt")])
    recovered_model = torch.load(recovered_dir / "model.pt", map_location="cpu", weights_only=False)
    expected_model = torch.load(reference_dir / "model.pt", map_location="cpu", weights_only=False)
    _assert_equal(recovered_model["model"], expected_model["model"])
    assert recovered_model["epoch"] == 2
    assert len(json.loads((recovered_dir / "history.json").read_text())) == 3
    assert (
        json.loads((recovered_dir / "provenance.json").read_text())["cumulative_updates"][
            "optimizer_attempts"
        ]
        == 12
    )
