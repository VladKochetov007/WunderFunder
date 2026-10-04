"""Tiny actual-trainer integration for E425 soft-route diagnostics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[5]
STAGING = REPO_ROOT / "research/campaign_2026-10-04/staging/e425"
FIXTURE_SEQUENCE_LENGTH = 200
FEATURES = 112


@dataclass(frozen=True)
class _FixtureSplit:
    """Minimal fit membership consumed by the actual staged trainer."""

    path: Path
    sha256: str
    fold: int = 0
    n_folds: int = 1
    train_groups: tuple[int, ...] = (0, 1)
    valid_groups: tuple[int, ...] = (0, 1)
    evaluation_groups: tuple[int, ...] = (2,)
    components: tuple[tuple[str, int, int], ...] = (
        ("train", 0, 0),
        ("train", 1, 1),
        ("valid", 0, 2),
        ("valid", 1, 3),
        ("valid", 2, 4),
    )

    def assert_fit_groups(self, train: tuple[int, ...], valid: tuple[int, ...]) -> None:
        """Reject fixture groups outside fit membership or overlapping evaluation."""
        if not set(map(int, train)).issubset(self.train_groups):
            raise ValueError("TRAIN groups fall outside fixture fit membership")
        if not set(map(int, valid)).issubset(self.valid_groups):
            raise ValueError("VALID groups fall outside fixture fit membership")
        components = {(source, group): value for source, group, value in self.components}
        fit_components = {components[("train", int(group))] for group in train}
        fit_components.update(components[("valid", int(group))] for group in valid)
        evaluation_components = {components[("valid", group)] for group in self.evaluation_groups}
        if fit_components & evaluation_components:
            raise ValueError("fixture fit and evaluation components overlap")


def _load_staged_trainer() -> ModuleType:
    """Load staged Python modules before the actual trainer module is imported."""
    staged_source = STAGING / "src"
    root_source = REPO_ROOT / "src"
    staged_path, root_path = str(staged_source), str(root_source)
    sys.path[:] = [path for path in sys.path if path not in (staged_path, root_path)]
    sys.path[0:0] = [staged_path, root_path]
    package = sys.modules.get("wnn")
    if package is not None:
        overlay = str(staged_source / "wnn")
        package.__path__ = [overlay] + [path for path in package.__path__ if path != overlay]
    for name in ("wnn.specialist", "wnn.specialist_diagnostics"):
        loaded = sys.modules.get(name)
        loaded_path = getattr(loaded, "__file__", "")
        if loaded is not None and not Path(loaded_path).is_relative_to(STAGING):
            del sys.modules[name]
    trainer_path = STAGING / "scripts/train_rnn.py"
    spec = importlib.util.spec_from_file_location("e425_actual_staged_train_rnn", trainer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load staged trainer at {trainer_path}")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    return trainer


def _install_fixture(monkeypatch, trainer: ModuleType, root: Path) -> _FixtureSplit:
    """Install 200-row data while retaining the real optimizer/training loop."""
    for source_file in (REPO_ROOT / "src/wnn").rglob("*.py"):
        target = root / "src/wnn" / source_file.relative_to(REPO_ROOT / "src/wnn")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
    for directory in ("datasets", "cache", "artifacts/runs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    manifest = root / "split.json"
    manifest.write_text("{}\n")
    split = _FixtureSplit(manifest, hashlib.sha256(b"e425-synthetic-split").hexdigest())
    monkeypatch.setattr(trainer, "ROOT", root)
    monkeypatch.setattr(trainer, "SEQUENCE_LENGTH", FIXTURE_SEQUENCE_LENGTH)
    monkeypatch.setattr(trainer, "load_campaign_split", lambda _path, _root: split)
    monkeypatch.setattr(trainer, "bind_fold_teachers", lambda *_args: [{"run": "tiny_teacher"}])
    monkeypatch.setattr(trainer, "validate_campaign_checkpoint", lambda *_args: None)

    class _SyntheticSequences(Dataset):
        def __init__(self, source, indices, _teachers=None, **_kwargs) -> None:
            self.source = source
            self.indices = list(map(int, indices))

        def __len__(self) -> int:
            return len(self.indices)

        def batch_tensor_bytes(self, batch_size: int, *, heuristics=None) -> int:
            del heuristics
            return batch_size * 64_000

        def __getitem__(self, index: int):
            del index
            steps = np.arange(FIXTURE_SEQUENCE_LENGTH, dtype=np.float32)
            features = np.zeros((FIXTURE_SEQUENCE_LENGTH, FEATURES), dtype=np.float32)
            features[:, 0] = np.sin(steps / 11.0)
            features[:, 1] = np.cos(steps / 7.0)
            target = np.zeros((FIXTURE_SEQUENCE_LENGTH, 4), dtype=np.float32)
            target[:, 0] = ((steps.astype(np.int64) // 10) % 2) * 0.12543
            target[:, 1] = -target[:, 0]
            target[:, 2] = 0.03
            target[:, 3] = -0.02
            scored = np.zeros(FIXTURE_SEQUENCE_LENGTH, dtype=bool)
            if self.source == "valid":
                scored[99::100] = True
            return torch.from_numpy(features), torch.from_numpy(target), torch.from_numpy(scored)

    monkeypatch.setattr(trainer, "Sequences", _SyntheticSequences)
    return split


def test_actual_staged_trainer_records_soft_expert_gradients_and_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train four tiny optimizer windows through staged `main`, then audit provenance."""
    trainer = _load_staged_trainer()
    _install_fixture(monkeypatch, trainer, tmp_path)
    run_name = "e425_tiny_soft"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rnn.py",
            "--name",
            run_name,
            "--device",
            "cpu",
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--train-source",
            "train+valid",
            "--train-seqs",
            "2",
            "--valid-repeat",
            "1",
            "--valid-train-fraction",
            "1",
            "--full-valid",
            "--no-eval-during-fit",
            "--checkpoint-policy",
            "fixed",
            "--checkpoint-epoch",
            "0",
            "--batch-order",
            "seeded",
            "--batch",
            "2",
            "--chunk",
            "100",
            "--epochs",
            "1",
            "--hidden",
            "4",
            "--expert-hidden",
            "2",
            "--num-experts",
            "4",
            "--layers",
            "1",
            "--dropout",
            "0",
            "--arch",
            "specialist",
            "--specialist-route",
            "soft",
            "--prefix-rows",
            "99",
            "--seed",
            "3",
            "--optim",
            "muon",
            "--muon-scope",
            "rnn",
            "--lr",
            "0.001",
            "--wd",
            "0.01",
            "--loss",
            "wmse",
            "--aux-struct",
            "0.3",
            "--aux-next",
            "0.3",
            "--ema-teacher",
            "1",
            "--ema-decay",
            "0.999",
            "--eval-ema",
            "--precision",
            "fp32",
        ],
    )

    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer.main()
    finally:
        torch.set_num_threads(prior_threads)

    output = tmp_path / "artifacts/runs" / run_name
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["args"]["specialist_route"] == "soft"
    assert (
        provenance["model_source_sha256"]
        == hashlib.sha256((STAGING / "src/wnn/specialist.py").read_bytes()).hexdigest()
    )
    epoch = provenance["soft_route_diagnostics"][0]
    assert epoch["optimizer_attempts"] == 4
    assert epoch["route_rows"] == 800
    assert len(epoch["expert_prediction_gradient_l2_mean"]) == 4
    assert len(epoch["expert_parameter_update_l2_mean"]) == 4
    assert epoch["expert_prediction_gradient_nonzero_steps"] == [4] * 4
    assert epoch["expert_parameter_nonzero_update_steps"] == [4] * 4
    assert all(value > 0 for value in epoch["expert_prediction_gradient_l2_mean"])
    assert all(value > 0 for value in epoch["expert_parameter_update_l2_mean"])
    assert math.isfinite(epoch["route_entropy_mean_nats"])
    assert (output / "model_ep0.pt").is_file()
