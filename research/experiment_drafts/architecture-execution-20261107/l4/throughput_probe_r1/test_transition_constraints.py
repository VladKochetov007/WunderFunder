from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from wnn.training.sequence_checkpoint import load_training_state
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_epoch_index import publish_epoch_index, validate_epoch_index_state
from wnn.training.sequence_epoch_prediction import epoch_admission, validate_epoch_payload
from wnn.training.sequence_provenance import canonical_sha256, file_sha256


@pytest.fixture(autouse=True)
def windows_fixture_directory_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass POSIX-only fixture durability on Windows, retaining all validation code."""
    if os.name == "nt":
        monkeypatch.setattr("wnn.training.sequence_epoch_index_io.fsync_directory", lambda _: None)


def identities() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Describe a diagnostic-only transition without constructing or fitting a model."""
    config = {"checkpoint_retention": "all_epochs", "epochs": 12, "seed": 101001}
    files = {
        "scripts/train_sequence.py": "a" * 64,
        "src/wnn/training/sequence_monitor_gates.py": "b" * 64,
        "src/wnn/training/sequence_monitor_diagnostics.py": "c" * 64,
    }
    data = {"monitoring_plan": {"enabled": True}}
    old = {
        "config_sha256": canonical_sha256(config),
        "source_files": files,
        "source_sha256": canonical_sha256(files),
        "data": data,
        "data_sha256": canonical_sha256(data),
        "runtime": {"device_uuid": "fixture-only-no-CUDA"},
        "world_size": 1,
    }
    new = copy.deepcopy(old)
    new["source_files"]["src/wnn/training/sequence_monitor_gates.py"] = "d" * 64
    new["source_files"]["src/wnn/training/sequence_monitor_diagnostics.py"] = "e" * 64
    new["source_sha256"] = canonical_sha256(new["source_files"])
    return old, new, config


def safe_state(identity: dict[str, object]) -> dict[str, object]:
    """Create tensor/RNG/offset fixtures; never instantiate a trainable architecture."""
    return {
        "checkpoint_schema": "wnn.sequence-resume.v1",
        "contract": identity,
        "world_size": 1,
        "next_epoch": 1,
        "global_step": 2,
        "model": {"fixture": torch.tensor([1.0, 2.0])},
        "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.25, 0.5])}}},
        "scheduler": {"last_epoch": 2, "_last_lr": [0.0007]},
        "history": [{"epoch": 0, "complete": True, "updates": 2}],
        "rank_rng_states": [
            {
                "python": random.Random(101001).getstate(),
                "numpy": np.random.RandomState(101001).get_state(),  # noqa: NPY002
                "torch_cpu": torch.Generator().manual_seed(101001).get_state(),
                "torch_cuda": torch.tensor([0, 1, 2], dtype=torch.uint8),
            }
        ],
        "rank_log_offsets": [81],
        "metrics_offset": 41,
        "monitor_offsets": {"events": 121, "evaluations": 161},
    }


def assert_identical(left: object, right: object) -> None:
    """Compare heterogeneous checkpoint values without tensor truth-value shortcuts."""
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_identical(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            assert_identical(lhs, rhs)
    else:
        assert left == right


def test_source_only_resume_clone_preserves_tensors_rng_offsets(tmp_path: Path) -> None:
    """The resume envelope itself is compatible; retained-epoch contracts are the blocker."""
    old, new, _ = identities()
    original = safe_state(old)
    old_path, new_path = tmp_path / "old.pt", tmp_path / "new.pt"
    torch.save(original, old_path)
    old_hash = file_sha256(old_path)
    cloned = copy.deepcopy(original)
    cloned["contract"] = new
    torch.save(cloned, new_path)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    loaded = load_training_state(new_path, new, context)
    for key in original.keys() - {"contract"}:
        assert_identical(original[key], loaded[key])
    assert file_sha256(old_path) == old_hash
    with pytest.raises(RuntimeError, match="identity does not match"):
        load_training_state(old_path, new, context)


@pytest.mark.parametrize("offset", [-1, True, None])
def test_canonical_loader_refuses_unsafe_monitor_offsets(tmp_path: Path, offset: object) -> None:
    old, _, _ = identities()
    state = safe_state(old)
    state["monitor_offsets"]["events"] = offset
    path = tmp_path / "invalid-offset.pt"
    torch.save(state, path)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    with pytest.raises(RuntimeError, match="safe monitor log offsets"):
        load_training_state(path, old, context)


def epoch_fixture(
    root: Path, identity: dict[str, object], config: dict[str, object]
) -> dict[str, object]:
    """Publish a canonical retained-epoch artifact with an honest historical producer."""
    folder = root / "epoch_checkpoints"
    folder.mkdir(parents=True)
    temporary = folder / "temporary.pt"
    payload = {
        "checkpoint_schema": "wnn.sequence-epoch.v1",
        "model": {"fixture": torch.tensor([1.0, 2.0])},
        "config": config,
        "epoch": 0,
        "completed_epochs": 1,
        "global_step": 2,
        "identity": identity,
        "data": identity["data"],
        "provenance": {"identity": identity},
    }
    torch.save(payload, temporary)
    sha = file_sha256(temporary)
    path = folder / f"epoch-000001-{sha}.pt"
    temporary.rename(path)
    return {
        "epoch": 0,
        "completed_epochs": 1,
        "global_step": 2,
        "checkpoint_path": f"epoch_checkpoints/{path.name}",
        "checkpoint_sha256": sha,
    }


def test_historical_index_cannot_resume_under_new_source(tmp_path: Path) -> None:
    old, new, config = identities()
    entry = epoch_fixture(tmp_path, old, config)
    publish_epoch_index(tmp_path, old, 0, 1, 2, entry, None)
    history = [{"epoch": 0, "complete": True, "updates": 2}]
    assert validate_epoch_index_state(tmp_path, old, 1, 2, history)
    with pytest.raises(ValueError, match="config/data/source/runtime identity does not match"):
        validate_epoch_index_state(tmp_path, new, 1, 2, history)


def test_reindexing_old_payload_as_new_source_breaks_inference_admission(tmp_path: Path) -> None:
    old, new, config = identities()
    entry = epoch_fixture(tmp_path, old, config)
    index = publish_epoch_index(tmp_path, new, 0, 1, 2, entry, None)
    checkpoint = tmp_path / entry["checkpoint_path"]
    admission = epoch_admission(checkpoint, tmp_path / "epoch_index.json", index["index_sha256"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    with pytest.raises(ValueError, match="identity differs from its index"):
        validate_epoch_payload(payload, admission)
