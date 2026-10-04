"""Tiny staged-trainer CPU integration for E426 clock separation and resume."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[5]
STAGING = REPO_ROOT / "research/campaign_2026-10-04/staging/e426"
SEQUENCE_LENGTH = 200
N_FEATURES = 112
STOP_AFTER_FIRST_EPOCH = "stop after persisted epoch-zero state"


@dataclass(frozen=True)
class _FixtureSplit:
    """Small component-safe stand-in for the parsed campaign manifest."""

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

    def assert_fit_groups(self, train, valid) -> None:
        """Require TRAIN and fit-VALID membership; exclude the evaluation group."""
        if not set(map(int, train)).issubset(self.train_groups):
            raise ValueError("TRAIN rows fall outside the fixture fit manifest")
        if not set(map(int, valid)).issubset(self.valid_groups):
            raise ValueError("VALID rows fall outside the fixture fit manifest")
        components = {(split, group): component for split, group, component in self.components}
        fit_components = {components[("train", int(group))] for group in train}
        fit_components.update(components[("valid", int(group))] for group in valid)
        eval_components = {components[("valid", group)] for group in self.evaluation_groups}
        if fit_components & eval_components:
            raise ValueError("fit/evaluation components overlap")


def _load_staged_trainer() -> ModuleType:
    """Load the isolated trainer with staged WNN overrides ahead of root modules."""
    staged_src = str(STAGING / "src")
    root_src = str(REPO_ROOT / "src")
    sys.path[:] = [path for path in sys.path if path not in (staged_src, root_src)]
    sys.path[0:0] = [staged_src, root_src]
    package = sys.modules.get("wnn")
    if package is not None:
        overlay = str(STAGING / "src" / "wnn")
        package.__path__ = [overlay] + [path for path in package.__path__ if path != overlay]
    for name in ("wnn.event_ema", "wnn.training_state"):
        loaded = sys.modules.get(name)
        loaded_path = getattr(loaded, "__file__", "")
        if loaded is not None and not Path(loaded_path).is_relative_to(STAGING):
            del sys.modules[name]
    module_path = STAGING / "scripts" / "train_rnn.py"
    spec = importlib.util.spec_from_file_location("e426_staged_train_rnn", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load staged trainer at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_event_parquet(path: Path, counts: tuple[int, ...]) -> None:
    """Write one raw input-only a6 endpoint stream per synthetic row group."""
    endpoint = np.float32(-5.199337482452393)
    values = np.zeros(len(counts) * SEQUENCE_LENGTH, dtype=np.float32)
    for group, count in enumerate(counts):
        start = group * SEQUENCE_LENGTH
        values[start : start + count] = endpoint
    pq.write_table(pa.table({"a6": values}), path, row_group_size=SEQUENCE_LENGTH)


def _install_cpu_fixture(monkeypatch, trainer: ModuleType, root: Path) -> _FixtureSplit:
    """Install synthetic inputs and exact root source closure for the real trainer loop."""
    for directory in ("datasets", "cache", "artifacts/runs/tiny_prop"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for source_path in (REPO_ROOT / "src/wnn").rglob("*.py"):
        relative_path = source_path.relative_to(REPO_ROOT / "src/wnn")
        target_path = root / "src/wnn" / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)
    (root / "split.json").write_text("{}\n")
    _write_event_parquet(root / "datasets/train.parquet", (0, 1))
    _write_event_parquet(root / "datasets/valid.parquet", (10, 100, 200))

    split = _FixtureSplit(root / "split.json", hashlib.sha256(b"fixture-split").hexdigest())
    monkeypatch.setattr(trainer, "ROOT", root)
    # Bound only the fixture's synthetic sequence length; production remains 20k rows.
    monkeypatch.setattr(trainer, "SEQUENCE_LENGTH", SEQUENCE_LENGTH)
    endpoint_counter = trainer.endpoint_events_by_update
    monkeypatch.setattr(
        trainer,
        "endpoint_events_by_update",
        lambda path, groups, update_rows: endpoint_counter(
            path, groups, update_rows, sequence_length=SEQUENCE_LENGTH
        ),
    )
    monkeypatch.setattr(trainer, "load_campaign_split", lambda _path, _root: split)
    monkeypatch.setattr(trainer, "bind_fold_teachers", lambda *_args: [{"run": "tiny_teacher"}])
    monkeypatch.setattr(trainer, "validate_campaign_checkpoint", lambda *_args: None)
    monkeypatch.setattr(trainer, "load_propensity", lambda *_args: object())
    monkeypatch.setattr(
        trainer,
        "prop_cache_path",
        lambda name, source: root / "cache" / f"propcache_{name}_{source}.npy",
    )

    class _SyntheticSequences(Dataset):
        def __init__(self, source, indices, _teachers=None, prop_cache=None, **_kwargs) -> None:
            self.split = source
            self.indices = list(map(int, indices))
            self.prop_cache = prop_cache

        def __len__(self) -> int:
            return len(self.indices)

        def batch_tensor_bytes(self, batch_size: int, *, heuristics=None) -> int:
            del heuristics
            return batch_size * 1_000_000

        def __getitem__(self, index: int):
            rows = np.arange(SEQUENCE_LENGTH, dtype=np.int32)
            features = np.zeros((SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
            target = np.zeros((SEQUENCE_LENGTH, 4), dtype=np.float32)
            target[:, 0] = ((rows // 10) % 2).astype(np.float32) * np.float32(0.12543)
            target[:, 2] = np.float32(0.03)
            target[:, 3] = np.float32(-0.02)
            scored = np.zeros(SEQUENCE_LENGTH, dtype=bool)
            if self.split == "valid":
                scored[99::100] = True
            result = (
                torch.from_numpy(features),
                torch.from_numpy(target),
                torch.from_numpy(scored),
            )
            if self.prop_cache is not None:
                probability = torch.full((SEQUENCE_LENGTH,), 0.5, dtype=torch.float32)
                result += (probability,)
            return result

    monkeypatch.setattr(trainer, "Sequences", _SyntheticSequences)

    teacher_checkpoint = torch.nn.Linear(1, 1)
    torch.save(
        {"model": teacher_checkpoint.state_dict()}, root / "artifacts/runs/tiny_prop/model.pt"
    )
    for source, row_groups in (("train", (0, 1)), ("valid", (0, 1, 2))):
        np.savez(root / "cache" / f"{source}_targets.npz", row_group=np.asarray(row_groups))
        prediction_rows = len(row_groups)
        np.save(
            root / "cache" / f"pred_tiny_teacher_{source}.npy",
            np.zeros((prediction_rows, 1), dtype=np.float32),
        )
        propensity_path = root / "cache" / f"propcache_tiny_prop_{source}.npy"
        np.save(propensity_path, np.full((prediction_rows, 1), 0.5, dtype=np.float32))
        propensity_path.with_suffix(".json").write_text("{}\n")
    return split


def _argv(root: Path, run_name: str, *, resume_path: Path | None = None) -> list[str]:
    """Return a two-epoch, fixed-final-export E426 combo-objective fixture."""
    values = [
        "train_rnn.py",
        "--name",
        run_name,
        "--device",
        "cpu",
        "--split-manifest",
        str(root / "split.json"),
        "--train-source",
        "train+valid",
        "--train-seqs",
        "2",
        "--valid-repeat",
        "1",
        "--full-valid",
        "--no-eval-during-fit",
        "--eval-source",
        "holdout",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "1",
        "--save-training-state",
        "--batch-order",
        "seeded",
        "--batch",
        "2",
        "--chunk",
        "100",
        "--epochs",
        "2",
        "--hidden",
        "4",
        "--layers",
        "1",
        "--dropout",
        "0",
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
        "--distill",
        "tiny_teacher",
        "--distill-alpha",
        "0.5",
        "--distill-boot-h",
        "8",
        "--propensity",
        "tiny_prop",
        "--prop-cache",
        "--prop-power",
        "2",
        "--prop-eps",
        "0.03",
        "--valid-true-mask",
        "0.03",
        "--since-weight",
        "32",
        "--since-floor",
        "0.25",
        "--ema-teacher",
        "1",
        "--ema-decay",
        "0.8",
        "--eval-ema",
        "--teacher-ema-clock",
        "aligned",
        "--teacher-ema-shuffle-seed",
        "86173",
    ]
    if resume_path is not None:
        values += ["--resume-state", str(resume_path)]
    return values


def _assert_export_is_evaluation_ema(root: Path, run_name: str) -> tuple[dict, dict]:
    """Check distinct saved EMA roles and exact fixed-epoch inference identity."""
    state = torch.load(
        root / "artifacts" / "runs" / run_name / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint = torch.load(
        root / "artifacts" / "runs" / run_name / "model_ep1.pt",
        map_location="cpu",
        weights_only=False,
    )
    teacher = state["teacher_ema"]["state"]
    evaluation = state["evaluation_ema"]["state"]
    assert teacher["n_averaged"] == evaluation["n_averaged"]
    assert any(
        not torch.equal(teacher[key], evaluation[key])
        for key in teacher
        if key.startswith("module.")
    )
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(value, evaluation[f"module.{key}"], rtol=0, atol=0)
    assert checkpoint["epoch"] == 1
    assert checkpoint["args"]["checkpoint_policy"] == "fixed"
    assert checkpoint["args"]["checkpoint_epoch"] == 1
    return state, checkpoint


def test_staged_two_epoch_resume_matches_uninterrupted_fixed_export(tmp_path, monkeypatch) -> None:
    """Run real trainer loops around synthetic data and compare epoch-boundary resume."""
    monkeypatch.setenv("WNN_LOADER_WORKERS", "0")
    monkeypatch.setenv("WNN_CPU_THREADS", "1")
    trainer = _load_staged_trainer()
    _install_cpu_fixture(monkeypatch, trainer, tmp_path)

    original_argv = sys.argv
    original_save = trainer.atomic_save_training_state
    try:
        sys.argv = _argv(tmp_path, "uninterrupted")
        trainer.main()
        _assert_export_is_evaluation_ema(tmp_path, "uninterrupted")

        def save_then_interrupt(state, path):
            original_save(state, path)
            if state["next_epoch"] == 1:
                raise RuntimeError(STOP_AFTER_FIRST_EPOCH)

        monkeypatch.setattr(trainer, "atomic_save_training_state", save_then_interrupt)
        sys.argv = _argv(tmp_path, "resumed")
        with pytest.raises(RuntimeError, match=STOP_AFTER_FIRST_EPOCH):
            trainer.main()

        resume_path = tmp_path / "artifacts/runs/resumed/training_state.pt"
        monkeypatch.setattr(trainer, "atomic_save_training_state", original_save)
        sys.argv = _argv(tmp_path, "resumed", resume_path=resume_path)
        trainer.main()
        resumed_state, resumed_checkpoint = _assert_export_is_evaluation_ema(tmp_path, "resumed")

        baseline = torch.load(
            tmp_path / "artifacts/runs/uninterrupted/model_ep1.pt",
            map_location="cpu",
            weights_only=False,
        )
        for key, value in baseline["model"].items():
            torch.testing.assert_close(value, resumed_checkpoint["model"][key], rtol=0, atol=0)
        provenance = json.loads((tmp_path / "artifacts/runs/resumed/provenance.json").read_text())
        source = provenance["teacher_ema_source"]
        payload = json.dumps(source["ordered_fit_rows"], separators=(",", ":")).encode("ascii")
        assert hashlib.sha256(payload).hexdigest() == source["ordered_fit_rows_sha256"]
        assert source["excluded_group_holes"] == [
            {
                "split": "train",
                "row_group": 528,
                "reason": "corrupt TRAIN row group; excluded from the declared fit manifest",
            }
        ]
        assert source["split_manifest_sha256"] == provenance["split_manifest_sha256"]
        assert [entry["epoch"] for entry in provenance["teacher_ema_clock"]["epochs"]] == [0, 1]
        assert resumed_state["next_epoch"] == 2
    finally:
        sys.argv = original_argv
