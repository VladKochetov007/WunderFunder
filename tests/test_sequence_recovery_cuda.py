"""CUDA acceptance for deterministic epoch-boundary Transformer recovery."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import wnn.training.sequence_fit_modes as sequence_fit_modes
from wnn.training.sequence_checkpoint import load_training_state
from wnn.training.sequence_distributed import DistributedContext, configure_strict_fp32
from wnn.training.sequence_fit_modes import run_full_fit
from wnn.training.sequence_fit_utils import (
    SEQUENCE_ROWS,
    RankJsonlLog,
    SequenceDataset,
    SequenceItem,
    make_optimizer,
    seed_worker,
    total_updates,
)
from wnn.training.sequence_models import build_sequence_model

pytestmark = pytest.mark.gpu


class _SyntheticSequenceDataset:
    """Return deterministic, full-length records through the production dataset protocol."""

    def __init__(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(1729)
        self._record: SequenceItem = {
            "x": torch.randn(SEQUENCE_ROWS, 112, generator=generator) * 0.05,
            "y": torch.randn(SEQUENCE_ROWS, 2, generator=generator) * 0.1,
            "mask": torch.arange(SEQUENCE_ROWS) >= 99,
            "teacher": torch.zeros(SEQUENCE_ROWS, 2),
            "hints": torch.empty(SEQUENCE_ROWS, 0),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> SequenceItem:
        if index == 0:
            return self._record
        if index != -1:
            raise IndexError(index)
        return cast(
            SequenceItem, {key: torch.zeros_like(value) for key, value in self._record.items()}
        )


class _StopAfterSavedEpoch(RuntimeError):
    """Model a process loss immediately after the epoch-one safe checkpoint is durable."""


def _fit_config(optimizer_name: str) -> dict[str, Any]:
    # The runtime config intentionally crosses the same heterogeneous mapping boundary as JSON.
    config: dict[str, Any] = {
        "epochs": 2,
        "batch_size": 1,
        "accumulate_chunks": 1,
        "chunk_rows": 10_000,
        "seed": 1234,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "optimizer": optimizer_name,
        "grad_clip": 1.0,
        "warmup_fraction": 0.25,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
        "model": {
            "architecture": "transformer",
            "hidden": 16,
            "layers": 2,
            "heads": 2,
            "ff_multiplier": 2,
            "context_rows": 8,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "hint_dim": 0,
            "hint_projection": "learned",
        },
    }
    if optimizer_name == "muon":
        config.update(
            {
                "lr": 3e-4,
                "muon_momentum": 0.95,
                "muon_ns_steps": 5,
                "muon_ns_eps": 1e-7,
                "adamw_beta1": 0.9,
                "adamw_beta2": 0.95,
                "adamw_eps": 1e-8,
                "schedule": "warmup_cosine_v2",
                "min_lr_ratio": 0.1,
            }
        )
    elif optimizer_name != "adamw":
        raise ValueError(f"unsupported acceptance optimizer: {optimizer_name}")
    return config


def _new_training_objects(
    config: dict[str, Any], dataset: SequenceDataset, device: torch.device, seed: int
) -> tuple[nn.Module, Optimizer, LambdaLR]:
    seed_worker(seed, 0)
    model = build_sequence_model(config["model"]).to(device=device, dtype=torch.float32)
    updates = total_updates(config, len(dataset), world_size=1)
    optimizer, scheduler = make_optimizer(model, config, updates)
    return model, optimizer, scheduler


def _assert_nested_close(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_close(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_close(left_value, right_value)
    else:
        assert left == right


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA receiver")
@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_transformer_epoch_checkpoint_resume_matches_uninterrupted_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, optimizer_name: str
) -> None:
    configure_strict_fp32()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    context = DistributedContext(
        rank=0, local_rank=0, world_size=1, device=device, process_group=False
    )
    config = _fit_config(optimizer_name)
    dataset = _SyntheticSequenceDataset()
    profile_receipt = tmp_path / "synthetic-profile.json"
    profile_receipt.write_text('{"fixture":"cuda-recovery"}\n', encoding="utf-8")
    profile_hash = hashlib.sha256(profile_receipt.read_bytes()).hexdigest()
    identity = {
        "schema": "synthetic-recovery-test-v1",
        "world_size": 1,
        "optimizer": optimizer_name,
    }
    normalization = {"mean": [0.0] * 112, "std": [1.0] * 112}
    checkpoint_path = tmp_path / "interrupted" / "training_state.pt"

    uninterrupted_dir = tmp_path / "uninterrupted"
    interrupted_dir = tmp_path / "interrupted"
    uninterrupted_dir.mkdir()
    interrupted_dir.mkdir()

    def execute(
        output: Path, seed: int, resume_path: Path | None = None
    ) -> tuple[nn.Module, Optimizer, LambdaLR]:
        model, optimizer, scheduler = _new_training_objects(config, dataset, device, seed)
        log = RankJsonlLog(output / "rank0.jsonl", resume=resume_path is not None)
        try:
            run_full_fit(
                dataset,
                model,
                model,
                optimizer,
                scheduler,
                config,
                context,
                log,
                None,
                identity,
                normalization,
                {"fixture": "deterministic-full-sequence-v1"},
                output,
                resume_path,
                profile_receipt,
                profile_hash,
            )
        finally:
            log.close()
        return model, optimizer, scheduler

    original_save = sequence_fit_modes.save_training_state
    saved_epoch: list[int] = []

    def save_then_interrupt(
        path: Path, state: dict[str, object], worker: DistributedContext
    ) -> None:
        original_save(path, state, worker)
        next_epoch = int(state["next_epoch"])
        saved_epoch.append(next_epoch)
        if next_epoch == 1:
            raise _StopAfterSavedEpoch

    monkeypatch.setattr(sequence_fit_modes, "save_training_state", save_then_interrupt)
    with pytest.raises(_StopAfterSavedEpoch):
        execute(interrupted_dir, seed=321)
    monkeypatch.setattr(sequence_fit_modes, "save_training_state", original_save)

    assert checkpoint_path.is_file()
    assert saved_epoch == [0, 1]
    checkpoint = load_training_state(checkpoint_path, identity, context)
    assert checkpoint["next_epoch"] == 1
    assert checkpoint["global_step"] == 2
    with pytest.raises(RuntimeError, match="identity does not match"):
        load_training_state(checkpoint_path, identity | {"wrong_identity": True}, context)

    uninterrupted_model, uninterrupted_optimizer, uninterrupted_scheduler = execute(
        uninterrupted_dir, seed=321
    )
    resumed_model, resumed_optimizer, resumed_scheduler = execute(
        interrupted_dir, seed=999, resume_path=checkpoint_path
    )

    _assert_nested_close(uninterrupted_model.state_dict(), resumed_model.state_dict())
    _assert_nested_close(uninterrupted_optimizer.state_dict(), resumed_optimizer.state_dict())
    _assert_nested_close(uninterrupted_scheduler.state_dict(), resumed_scheduler.state_dict())
    assert (uninterrupted_dir / "model.pt").is_file()
    assert (interrupted_dir / "model.pt").is_file()

    uninterrupted_model.eval()
    resumed_model.eval()
    sample = dataset[0]["x"][:128].unsqueeze(0).to(device)
    with torch.no_grad():
        expected, _, _ = uninterrupted_model(sample)
        actual, _, _ = resumed_model(sample)
    torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)
