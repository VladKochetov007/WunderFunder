from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wnn.training.sequence_checkpoint import load_training_state
from wnn.training.sequence_checkpoint_policy import (
    FINAL_ENDPOINT_PURPOSE,
    RESUME_STATE_PURPOSE,
    RETAINED_EPOCH_PURPOSE,
)
from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_epoch_checkpoints import (
    preflight_epoch_storage,
    save_epoch_checkpoint,
)
from wnn.training.sequence_epoch_fit import reconcile_epoch_snapshots
from wnn.training.sequence_fit_artifacts import write_final_outputs
from wnn.training.sequence_fit_state import training_state


def _snapshot(
    output: Path, model: torch.nn.Module, epoch: int, config: dict[str, object]
) -> dict[str, object]:
    return save_epoch_checkpoint(
        output,
        model,
        config,
        {"config_sha256": "a" * 64},
        {"mean": [0.0], "std": [1.0]},
        {"split": "training-only"},
        output / "profile.json",
        "b" * 64,
        [{"epoch": index, "updates": 1} for index in range(epoch + 1)],
        epoch,
        epoch + 1,
    )


def _patch_directory_fsync_for_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map unsupported directory fsyncs to a disposable regular file in tests."""
    import os

    original_open = os.open

    def portable_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        target = Path(path) if isinstance(path, (str, Path)) else None
        if target is not None and target.is_dir():
            target = target / ".test-directory-fsync"
            return original_open(target, os.O_CREAT | os.O_RDWR)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", portable_open)
    import wnn.training.sequence_epoch_checkpoints as checkpoints

    monkeypatch.setattr(checkpoints, "fsync_directory", lambda _: None)


def test_epoch_snapshot_is_cloned_and_never_auto_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_directory_fsync_for_windows(monkeypatch)
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    config = {"epochs": 2, "model": {"attention_backend": "math"}}
    first = _snapshot(tmp_path, model, 0, config)
    first_path = tmp_path / str(first["checkpoint_path"])
    first_payload = torch.load(first_path, map_location="cpu", weights_only=False)
    assert first_payload["provenance"]["checkpoint_purpose"] == RETAINED_EPOCH_PURPOSE
    assert first_payload["provenance"]["checkpoint_policy"] == "all-safe-epochs-inference-only"

    with torch.no_grad():
        model.weight.fill_(7.0)
    second = _snapshot(tmp_path, model, 1, config)
    second_payload = torch.load(
        tmp_path / str(second["checkpoint_path"]), map_location="cpu", weights_only=False
    )
    first_after_mutation = torch.load(first_path, map_location="cpu", weights_only=False)
    assert torch.equal(first_after_mutation["model"]["weight"], torch.ones_like(model.weight))
    assert torch.equal(second_payload["model"]["weight"], torch.full_like(model.weight, 7.0))


def test_epoch_storage_preflight_accounts_for_every_snapshot_and_safety_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wnn.training.sequence_epoch_checkpoints as checkpoints

    model = torch.nn.Linear(3, 2)
    state_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
    )
    snapshots_only = int(state_bytes * 3 * 1.1) + (64 << 20)
    required = snapshots_only + 6 * state_bytes
    monkeypatch.setattr(
        checkpoints.shutil, "disk_usage", lambda _: SimpleNamespace(free=snapshots_only)
    )
    assert preflight_epoch_storage(tmp_path, model, 3) == snapshots_only
    with pytest.raises(OSError, match="needs at least"):
        preflight_epoch_storage(tmp_path, model, 3, reserve_atomic_checkpoint_state=True)

    final_only_reserve = (64 << 20) + state_bytes
    monkeypatch.setattr(
        checkpoints.shutil, "disk_usage", lambda _: SimpleNamespace(free=final_only_reserve)
    )
    assert preflight_epoch_storage(tmp_path, model, 0, reserve_atomic_checkpoint_state=True) == (
        final_only_reserve
    )
    monkeypatch.setattr(
        checkpoints.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=final_only_reserve - 1),
    )
    with pytest.raises(OSError, match="needs at least"):
        preflight_epoch_storage(tmp_path, model, 0, reserve_atomic_checkpoint_state=True)
    assert preflight_epoch_storage(tmp_path, model, 0) == 0
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda _: SimpleNamespace(free=required))
    assert (
        preflight_epoch_storage(tmp_path, model, 3, reserve_atomic_checkpoint_state=True)
        == required
    )
    monkeypatch.setattr(
        checkpoints.shutil, "disk_usage", lambda _: SimpleNamespace(free=required - 1)
    )
    with pytest.raises(OSError, match="needs at least"):
        preflight_epoch_storage(tmp_path, model, 3, reserve_atomic_checkpoint_state=True)


def test_resume_rejects_wrong_fit_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "training_state.pt"
    torch.save(
        {"checkpoint_schema": "wnn.sequence-resume.v1", "contract": {"fit": "other"}},
        checkpoint,
    )
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    with pytest.raises(
        RuntimeError, match="resume config/data/source/runtime identity does not match"
    ):
        load_training_state(checkpoint, {"fit": "expected"}, context)


def test_new_endpoint_policy_preflights_atomic_resume_space_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wnn.training.sequence_epoch_checkpoints as checkpoints

    model = torch.nn.Linear(2, 1)
    state_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
    )
    snapshot_only = int(state_bytes * 2 * 1.1) + (64 << 20)
    monkeypatch.setattr(
        checkpoints.shutil, "disk_usage", lambda _: SimpleNamespace(free=snapshot_only)
    )
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    config: dict[str, object] = {
        "epochs": 2,
        "checkpoint_retention": "all_epochs",
        "checkpoint_endpoint_policy": "configured_final_epoch",
    }
    with pytest.raises(RuntimeError, match="all-epoch checkpoint storage preflight failed"):
        reconcile_epoch_snapshots(
            tmp_path,
            model,
            config,
            {"config_sha256": "a" * 64},
            {"mean": [0.0], "std": [1.0]},
            {},
            tmp_path / "profile.json",
            "b" * 64,
            0,
            0,
            [],
            context,
        )
    assert not (tmp_path / "epoch_checkpoints").exists()


def test_resume_state_is_labeled_as_full_safe_state() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    state = training_state({"fit": "identity"}, 1, 0, 0, model, optimizer, scheduler, [], [0], 0)
    assert state["checkpoint_purpose"] == RESUME_STATE_PURPOSE
    assert {"model", "optimizer", "scheduler"} <= set(state)


def test_fixed_final_endpoint_ignores_higher_earlier_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_directory_fsync_for_windows(monkeypatch)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(7.0)
    history = [
        {"epoch": 0, "valid_metric": 0.99},
        {"epoch": 1, "valid_metric": 0.10},
    ]
    config: dict[str, object] = {
        "epochs": 2,
        "checkpoint_retention": "all_epochs",
        "checkpoint_endpoint_policy": "configured_final_epoch",
        "model": {"attention_backend": "math"},
    }
    write_final_outputs(
        tmp_path,
        model,
        config,
        history,
        {"config_sha256": "a" * 64},
        {"mean": [0.0], "std": [1.0]},
        {"split": "training-only"},
        tmp_path / "profile.json",
        "b" * 64,
        2,
        "c" * 64,
    )
    payload = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    assert payload["epoch"] == 1 and payload["completed_epochs"] == 2
    assert torch.equal(payload["model"]["weight"], torch.full((1, 1), 7.0))
    assert payload["provenance"]["checkpoint_purpose"] == FINAL_ENDPOINT_PURPOSE
    assert payload["provenance"]["checkpoint_endpoint_policy"] == "configured_final_epoch"
    assert payload["provenance"]["history"] == history


def test_generic_labels_overlay_and_historical_v1_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    historical = load_config(path)
    assert "checkpoint_endpoint_policy" not in historical

    source = (
        Path(__file__).parents[1]
        / "research/experiment_drafts/ensemble-objective-fix-20261108/configs/labels_s101101.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    assert raw["checkpoint_retention"] == "final_only"
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    fixed_v1 = tmp_path / "fixed-v1.json"
    fixed_v1.write_text(json.dumps(raw), encoding="utf-8")
    assert load_config(fixed_v1)["checkpoint_retention"] == "final_only"

    generic_labels = raw.copy()
    generic_labels.pop("objective_mode")
    generic_labels["checkpoint_retention"] = "all_epochs"
    generic_labels["checkpoint_endpoint_policy"] = "configured_final_epoch"
    path.write_text(json.dumps(generic_labels), encoding="utf-8")
    overlay = load_config(path)
    assert overlay["checkpoint_retention"] == "all_epochs"
    assert overlay["checkpoint_endpoint_policy"] == "configured_final_epoch"

    raw["checkpoint_retention"] = "all_epochs"
    raw["checkpoint_endpoint_policy"] = "configured_final_epoch"
    fixed_v1.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="ensemble objective requires"):
        load_config(fixed_v1)
