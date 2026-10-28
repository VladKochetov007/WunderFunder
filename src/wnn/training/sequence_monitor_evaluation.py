from __future__ import annotations

import random
import time
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from wnn.gmetric import GlobalAccumulator
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_utils import SequenceDataset, collate_sequences
from wnn.training.sequence_models import detach_state
from wnn.training.sequence_precision import forward_context


def evaluate_monitoring_sets(
    model: nn.Module,
    train_eval: SequenceDataset,
    valid: SequenceDataset,
    chunk_rows: int,
    precision_mode: str,
    context: DistributedContext,
    evaluation_batch_size: int = 1,
) -> tuple[dict[str, int | float], dict[str, int | float]]:
    """Evaluate both report-only cohorts while restoring mode, buffers, and RNG."""
    modes = [(module, module.training) for module in model.modules()]
    bn_buffers = _batch_norm_buffers(model)
    rng = _rng_state(context)
    started = time.perf_counter()
    try:
        model.eval()
        with torch.inference_mode():
            train_metrics = evaluate_dataset(
                model,
                train_eval,
                "train_eval",
                chunk_rows,
                precision_mode,
                context,
                evaluation_batch_size,
            )
            valid_metrics = evaluate_dataset(
                model,
                valid,
                "valid",
                chunk_rows,
                precision_mode,
                context,
                evaluation_batch_size,
            )
    finally:
        for module, training in modes:
            module.training = training
        _restore_rng(rng, context)
    _assert_batch_norm_unchanged(bn_buffers)
    elapsed = time.perf_counter() - started
    train_metrics["timing/evaluation_seconds"] = elapsed
    valid_metrics["timing/evaluation_seconds"] = elapsed
    return train_metrics, valid_metrics


@torch.inference_mode()
def evaluate_dataset(
    model: nn.Module,
    dataset: SequenceDataset,
    prefix: str,
    chunk_rows: int,
    precision_mode: str,
    context: DistributedContext,
    evaluation_batch_size: int = 1,
) -> dict[str, int | float]:
    """Compute official-mask weighted MSE and globally pooled official WP."""
    if not len(dataset) or chunk_rows <= 0 or evaluation_batch_size <= 0:
        raise ValueError("monitoring evaluation requires sequences and positive batch/chunk sizes")
    accumulator = GlobalAccumulator()
    squared_error = 0.0
    target_weight = 0.0
    prediction_sum = 0.0
    prediction_square_sum = 0.0
    clipped_count = 0
    selected_values = 0
    selected_rows = 0
    for batch_start in range(0, len(dataset), evaluation_batch_size):
        batch_indices = list(
            range(batch_start, min(batch_start + evaluation_batch_size, len(dataset)))
        )
        batch = collate_sequences(dataset, batch_indices)
        inputs, targets, mask = batch["x"], batch["y"], batch["mask"]
        if inputs.ndim != 3 or targets.shape != (*inputs.shape[:2], 2):
            raise ValueError("monitoring dataset does not contain canonical full sequences")
        state = None
        for start in range(0, inputs.shape[1], chunk_rows):
            stop = min(start + chunk_rows, inputs.shape[1])
            values = inputs[:, start:stop].to(context.device, non_blocking=True)
            with forward_context(precision_mode, context.device.type):
                prediction, _, next_state = model(values, state)
            state = detach_state(next_state)
            predicted = prediction.float().cpu().numpy()
            target = targets[:, start:stop].numpy()
            scored = mask[:, start:stop].numpy()
            accumulator.add(target, predicted, scored)
            clipped_target = np.clip(target, -2.0, 2.0)
            weight = np.abs(clipped_target) * scored[..., None]
            difference = predicted.astype(np.float64) - clipped_target.astype(np.float64)
            squared_error += float((weight * np.square(difference)).sum())
            target_weight += float(weight.sum())
            selected = np.broadcast_to(scored[..., None], predicted.shape)
            chosen = predicted[selected].astype(np.float64)
            prediction_sum += float(chosen.sum())
            prediction_square_sum += float(np.square(chosen).sum())
            clipped_count += int((np.abs(chosen) > 2.0).sum())
            selected_values += len(chosen)
            selected_rows += int(scored.sum())
    if selected_values <= 0:
        raise ValueError("monitoring cohort has no official scored rows")
    pooled = accumulator.result()
    mean = prediction_sum / selected_values
    variance = max(0.0, prediction_square_sum / selected_values - mean * mean)
    return {
        f"{prefix}/loss": squared_error / target_weight if target_weight > 0 else 0.0,
        f"{prefix}/wp": float(pooled["wp"]),
        f"{prefix}/wp/t0": float(pooled["t0"]),
        f"{prefix}/wp/t1": float(pooled["t1"]),
        f"{prefix}/selected_rows": selected_rows,
        f"{prefix}/sequence_count": len(dataset),
        f"{prefix}/prediction_mean": mean,
        f"{prefix}/prediction_std": variance**0.5,
        f"{prefix}/prediction_clipped_fraction": clipped_count / selected_values,
    }


def _batch_norm_buffers(model: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Snapshot only BatchNorm running buffers to prove evaluation never updates them."""
    names = ("running_mean", "running_var", "num_batches_tracked")
    return [
        (value, value.detach().clone())
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        for name in names
        if isinstance((value := getattr(module, name, None)), torch.Tensor)
    ]


def _assert_batch_norm_unchanged(buffers: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Fail if monitored evaluation changed any captured BatchNorm running buffer."""
    if any(not torch.equal(value, before) for value, before in buffers):
        raise RuntimeError("report-only monitoring changed BatchNorm running state")


def _rng_state(context: DistributedContext) -> dict[str, object]:
    """Capture host, NumPy, CPU, and available CUDA RNG state for exact restoration."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002 - preserve callers' global RNG exactly
        "torch": torch.get_rng_state(),
    }
    if context.device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state(context.device)
    return state


def _restore_rng(state: Mapping[str, object], context: DistributedContext) -> None:
    """Restore every RNG stream captured before report-only model evaluation."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002 - restore the captured global RNG
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device=context.device)
