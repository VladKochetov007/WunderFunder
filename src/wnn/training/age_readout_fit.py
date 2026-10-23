"""Matched three-arm cached residual fitting and target-free evaluation."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from wnn.conditional_readout import primary_pair_wmse
from wnn.training.checkpoints import checkpoint_selected

from .age_readout_checkpoint import restore_panel_checkpoint, save_panel_checkpoint
from .age_readout_cohort import FitBatch, iter_eval_sequences, iter_fit_batches
from .age_readout_features import normalized_age, normalized_base
from .age_readout_model import AgeResidualHead, zero_head_parity

ARMS = ("A", "G", "F")
EPOCHS = 3
FIXED_EPOCH = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
BATCH_ROWS = 8192
ProgressCallback = Callable[[dict[str, object]], None]
PredictionCallback = Callable[[str, int, np.ndarray], None]


def fit_panel_fold(
    fit_cache: np.ndarray,
    eval_cache: np.ndarray,
    parent_mean_array: np.ndarray,
    parent_std_array: np.ndarray,
    *,
    fold: int,
    seed: int,
    device_name: str,
    identity: dict[str, object],
    checkpoint_dir: Path | None,
    resume_from: Path | None = None,
    max_fit_sequences: int | None = None,
    max_eval_sequences: int | None = None,
    progress: ProgressCallback | None = None,
    prediction: PredictionCallback | None = None,
    release_fit_sequence: Callable[[int], None] | None = None,
    release_eval_sequence: Callable[[int], None] | None = None,
) -> dict[str, object]:
    """Fit A/G/F together so each cache sequence and age block is prepared once."""
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    if torch.get_default_dtype() != torch.float32 or seed < 0:
        raise RuntimeError("age readout requires a nonnegative seed and FP32 default dtype")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fit_count = _sequence_limit(max_fit_sequences, len(fit_cache), "fit")
    eval_count = _sequence_limit(max_eval_sequences, len(eval_cache), "evaluation")
    mean = _normalization_tensor(parent_mean_array, "parent mean", device)
    std = _normalization_tensor(parent_std_array, "parent standard deviation", device)
    if mean.shape != (211,) or std.shape != mean.shape or bool(torch.any(std <= 0)):
        raise ValueError("parent normalization must be positive-scale width211")
    if not zero_head_parity():
        raise AssertionError("fresh residual head failed parent parity")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    template = AgeResidualHead()
    models = {arm: AgeResidualHead().to(device) for arm in ARMS}
    for model in models.values():
        model.load_state_dict(template.state_dict())
    optimizers = {
        arm: torch.optim.AdamW(
            models[arm].parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        for arm in ARMS
    }
    records: list[dict[str, object]] = []
    start_epoch = 0
    if resume_from is not None:
        start_epoch, records = restore_panel_checkpoint(
            resume_from, identity, models, optimizers, device, ARMS, EPOCHS
        )

    _synchronize(device)
    fit_started = time.perf_counter()
    for epoch in range(start_epoch, EPOCHS):
        totals = _new_totals(device)
        batches = 0
        first_batch = True
        for batch in iter_fit_batches(
            fit_cache,
            fit_count,
            mean,
            std,
            device,
            batch_rows=BATCH_ROWS,
            release_sequence=release_fit_sequence,
        ):
            if first_batch and epoch == 0:
                _check_batch_zero_parity(batch, models)
                first_batch = False
            _train_batch(batch, models, optimizers, totals)
            batches += 1
            if progress is not None and batches % 100 == 0:
                progress({"fold": fold, "epoch": epoch, "batches": batches})
        if batches == 0:
            raise ValueError("fit cache yielded no batches")
        record = _epoch_record(epoch, fold, seed, batches, totals)
        records.append(record)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_panel_checkpoint(
                checkpoint_dir / f"epoch{epoch}.pt", identity, epoch, models, optimizers, records
            )
        if progress is not None:
            progress(record)

    if not checkpoint_selected(FIXED_EPOCH, "fixed", FIXED_EPOCH, None, float("-inf")):
        raise RuntimeError("fixed epoch 2 was not selected")
    _synchronize(device)
    fit_seconds = time.perf_counter() - fit_started
    eval_started = time.perf_counter()
    _predict_eval(
        eval_cache,
        eval_count,
        mean,
        std,
        models,
        device,
        prediction,
        release_eval_sequence,
    )
    _synchronize(device)
    eval_seconds = time.perf_counter() - eval_started
    return {
        "fold": fold,
        "seed": seed,
        "arms": list(ARMS),
        "fit_sequences": fit_count,
        "eval_sequences": eval_count,
        "fit_seconds": fit_seconds,
        "eval_seconds": eval_seconds,
        "records": records,
        "models": models,
        "fixed_checkpoint_epoch": FIXED_EPOCH,
        "identity": identity,
    }


def _train_batch(
    batch: FitBatch,
    models: dict[str, AgeResidualHead],
    optimizers: dict[str, torch.optim.Optimizer],
    totals: dict[str, dict[str, Tensor]],
) -> None:
    zero = torch.zeros_like(batch.group_age)
    age_by_arm = {"A": zero, "G": batch.group_age, "F": batch.field_age}
    with torch.autocast(device_type=batch.base.device.type, enabled=False):
        for arm in ARMS:
            features = torch.cat((batch.base, age_by_arm[arm]), dim=-1)
            prediction = models[arm](features, batch.parent)
            loss = primary_pair_wmse(
                prediction.unsqueeze(0),
                batch.target.unsqueeze(0),
                batch.row_weight.unsqueeze(0),
                batch.target.unsqueeze(0),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"arm {arm} loss became nonfinite")
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = clip_grad_norm_(models[arm].parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
            optimizer.step()
            if not bool(
                torch.stack([torch.isfinite(p).all() for p in models[arm].parameters()]).all()
            ):
                raise FloatingPointError(f"arm {arm} parameters became nonfinite")
            totals[arm]["loss"] += loss.detach()
            totals[arm]["grad"] += norm.detach()
            totals[arm]["max_grad"] = torch.maximum(totals[arm]["max_grad"], norm.detach())


def _predict_eval(
    cache: np.ndarray,
    sequence_count: int,
    mean: Tensor,
    std: Tensor,
    models: dict[str, AgeResidualHead],
    device: torch.device,
    sink: PredictionCallback | None,
    release_sequence: Callable[[int], None] | None = None,
) -> None:
    """Predict without reading labels, row weights, scoring masks or evaluation flags."""
    for model in models.values():
        model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for sequence_index, sequence in enumerate(
            iter_eval_sequences(cache, sequence_count, device, release_sequence=release_sequence)
        ):
            base = normalized_base(sequence.raw, mean, std)
            inputs = {
                "A": torch.cat((base, torch.zeros((len(base), 112), device=device)), dim=-1),
                "G": torch.cat((base, normalized_age(sequence.raw, "group_age")), dim=-1),
                "F": torch.cat((base, normalized_age(sequence.raw, "field_age")), dim=-1),
            }
            for arm in ARMS:
                output = models[arm](inputs[arm], sequence.parent)
                if not bool(torch.isfinite(output).all()):
                    raise FloatingPointError(f"arm {arm} evaluation became nonfinite")
                if sink is not None:
                    sink(arm, sequence_index, output.cpu().numpy().copy())


def _new_totals(device: torch.device) -> dict[str, dict[str, Tensor]]:
    return {
        arm: {
            "loss": torch.zeros((), dtype=torch.float32, device=device),
            "grad": torch.zeros((), dtype=torch.float32, device=device),
            "max_grad": torch.zeros((), dtype=torch.float32, device=device),
        }
        for arm in ARMS
    }


def _epoch_record(
    epoch: int, fold: int, seed: int, batches: int, totals: dict[str, dict[str, Tensor]]
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "fold": fold,
        "seed": seed,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "clip": GRADIENT_CLIP,
        "batch_rows": BATCH_ROWS,
        "updates_per_arm": batches,
        "arms": {
            arm: {
                "mean_batch_wmse": float(totals[arm]["loss"].item() / batches),
                "mean_preclip_gradient_norm": float(totals[arm]["grad"].item() / batches),
                "max_preclip_gradient_norm": float(totals[arm]["max_grad"].item()),
            }
            for arm in ARMS
        },
    }


def _check_batch_zero_parity(batch: FitBatch, models: dict[str, AgeResidualHead]) -> None:
    zero = torch.zeros_like(batch.group_age)
    for arm, age in (("A", zero), ("G", batch.group_age), ("F", batch.field_age)):
        features = torch.cat((batch.base, age), dim=-1)
        if not torch.equal(models[arm](features, batch.parent), batch.parent):
            raise AssertionError(f"arm {arm} did not start at its cached parent output")


def _normalization_tensor(values: np.ndarray, name: str, device: torch.device) -> Tensor:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector")
    return torch.tensor(array, dtype=torch.float32, device=device)


def _sequence_limit(value: int | None, available: int, name: str) -> int:
    limit = available if value is None else value
    if limit < 1 or limit > available:
        raise ValueError(f"{name} sequence limit is outside the cache membership")
    return limit


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
