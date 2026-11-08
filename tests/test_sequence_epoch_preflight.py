from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wnn.training.sequence_epoch_checkpoints import preflight_epoch_storage


def _state_bytes(model: torch.nn.Module) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())


def test_atomic_checkpoint_reserve_adds_six_state_copies_and_preserves_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Linear(2, 2)
    state_bytes = _state_bytes(model)
    snapshots = 3
    snapshot_only = int(state_bytes * snapshots * 1.10) + (64 << 20)
    monkeypatch.setattr(
        "wnn.training.sequence_epoch_checkpoints.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=snapshot_only + state_bytes * 6 - 1),
    )

    assert preflight_epoch_storage(tmp_path, model, snapshots) == snapshot_only
    with pytest.raises(OSError, match="all-epoch checkpoint retention needs"):
        preflight_epoch_storage(tmp_path, model, snapshots, reserve_atomic_checkpoint_state=True)


def test_atomic_checkpoint_reserve_formula_with_remaining_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Linear(2, 2)
    state_bytes = _state_bytes(model)
    snapshots = 3
    expected = int(state_bytes * snapshots * 1.10) + (64 << 20) + state_bytes * 6
    monkeypatch.setattr(
        "wnn.training.sequence_epoch_checkpoints.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=expected),
    )

    assert (
        preflight_epoch_storage(tmp_path, model, snapshots, reserve_atomic_checkpoint_state=True)
        == expected
    )


def test_atomic_checkpoint_reserve_covers_final_model_at_zero_remaining_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Linear(2, 2)
    expected = _state_bytes(model) + (64 << 20)
    monkeypatch.setattr(
        "wnn.training.sequence_epoch_checkpoints.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=expected),
    )

    assert (
        preflight_epoch_storage(tmp_path, model, 0, reserve_atomic_checkpoint_state=True)
        == expected
    )
    assert preflight_epoch_storage(tmp_path, model, 0) == 0
