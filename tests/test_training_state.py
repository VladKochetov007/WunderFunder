from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from wnn.optim import MuonAdamW
from wnn.training_state import (
    atomic_save_training_state,
    capture_rng,
    capture_training_state,
    load_training_state,
    restore_rng,
    restore_training_state,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_rnn import FastEMA  # noqa: E402


def _objects(seed: int, fast_ema: bool, weighted: bool, muon: bool = False) -> tuple:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Dropout(0.25), torch.nn.Linear(4, 2)
    )
    model.register_buffer("mean", torch.tensor([0.5, -0.25, 0.1]))
    model.register_buffer("std", torch.tensor([1.5, 0.5, 2.0]))
    optimizer = (
        MuonAdamW(
            [
                {"params": [model[0].weight], "muon": True},
                {"params": [model[0].bias, *model[2].parameters()]},
            ],
            lr=0.01,
        )
        if muon
        else torch.optim.AdamW(model.parameters(), lr=0.01)
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, total_steps=12, pct_start=0.25
    )
    scaler = torch.amp.GradScaler("cpu")
    ema = (
        FastEMA(model, 0.9)
        if fast_ema
        else torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9)
        )
    )
    swa = torch.optim.swa_utils.AveragedModel(model)
    values = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 20
    dataset = TensorDataset(values, torch.arange(8))
    sampler = (
        WeightedRandomSampler(
            torch.arange(1, 9, dtype=torch.float32),
            num_samples=8,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        if weighted
        else None
    )
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=sampler is None)
    return model, optimizer, scheduler, scaler, loader, ema, swa


def _epoch(objects: tuple, overflow_batch: int = -1) -> list[tuple]:
    model, optimizer, scheduler, scaler, loader, ema, swa = objects
    rows = []
    for batch, (values, indices) in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        prediction = model((values - model.mean) / model.std)
        target = values[:, :2] + 0.01 * (random.random() + np.random.rand())
        loss = (prediction - target).square().mean()
        if batch == overflow_batch:
            loss = loss * float("inf")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update_parameters(model)
        swa.update_parameters(model)
        rows.append((indices.tolist(), loss.item(), scheduler.get_last_lr()))
    return rows


def _capture(objects: tuple, epoch: int) -> dict:
    model, optimizer, scheduler, scaler, loader, ema, swa = objects
    return capture_training_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loader=loader,
        contract={"epochs": 3, "total_steps": 12, "fixed_epoch": 2},
        next_epoch=epoch + 1,
        progress={"step_count": (epoch + 1) * 4},
        ema=ema,
        swa=swa,
        auxiliary={"norm_ema": torch.tensor(13.0)},
    )


def _assert_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, list | tuple):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_equal(first, second)
    else:
        assert left == right


@pytest.mark.parametrize("fast_ema", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("muon", [False, True])
def test_epoch_resume_is_bitwise_identical(
    tmp_path: Path, fast_ema: bool, weighted: bool, muon: bool
) -> None:
    uninterrupted = _objects(7, fast_ema, weighted, muon)
    reference = [
        _epoch(uninterrupted, overflow_batch=3),
        _epoch(uninterrupted),
        _epoch(uninterrupted),
    ]
    final_reference = _capture(uninterrupted, 2)

    interrupted = _objects(7, fast_ema, weighted, muon)
    assert _epoch(interrupted, overflow_batch=3) == reference[0]
    path = tmp_path / "training_state.pt"
    before_save = capture_rng()
    atomic_save_training_state(_capture(interrupted, 0), path)
    _assert_equal(before_save, capture_rng())
    saved = load_training_state(path, final_reference["contract"])
    resumed = _objects(1234, fast_ema, weighted, muon)
    model, optimizer, scheduler, scaler, loader, ema, swa = resumed
    restore_training_state(
        saved,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loader=loader,
        ema=ema,
        swa=swa,
    )
    restore_rng(saved["rng"])
    assert [_epoch(resumed), _epoch(resumed)] == reference[1:]
    _assert_equal(final_reference, _capture(resumed, 2))


def test_resume_rejects_inference_and_changed_horizon(tmp_path: Path) -> None:
    path = tmp_path / "model_ep0.pt"
    torch.save({"model": {}, "epoch": 0}, path)
    with pytest.raises(ValueError, match="full epoch training state"):
        load_training_state(path)
    objects = _objects(3, True, False)
    _epoch(objects)
    state = _capture(objects, 0)
    atomic_save_training_state(state, path)
    with pytest.raises(ValueError, match="contract fields.*epochs"):
        load_training_state(path, {**state["contract"], "epochs": 4})


def test_atomic_failure_preserves_previous_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "training_state.pt"
    path.write_bytes(b"previous committed state")

    def broken_save(state: object, stream: object) -> None:
        stream.write(b"incomplete next state")
        raise OSError("simulated disk exhaustion")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk exhaustion"):
        atomic_save_training_state({}, path)
    assert path.read_bytes() == b"previous committed state"
    assert list(tmp_path.iterdir()) == [path]


def test_capture_owns_tensor_storage() -> None:
    objects = _objects(3, True, False)
    _epoch(objects)
    state = _capture(objects, 0)
    before = state["model"]["0.weight"].clone()
    _epoch(objects)
    assert torch.equal(state["model"]["0.weight"], before)
