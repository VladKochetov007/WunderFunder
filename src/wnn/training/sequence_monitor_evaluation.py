from __future__ import annotations

import random
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from wnn.gmetric import GlobalAccumulator
from wnn.training.sequence_blend import compose_prediction
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_utils import SequenceDataset, collate_sequences
from wnn.training.sequence_models import detach_state
from wnn.training.sequence_monitor_events import MonitorEventWriter
from wnn.training.sequence_monitor_identity import validate_dataset_members
from wnn.training.sequence_precision import forward_context


@dataclass(frozen=True)
class EvaluationEventContext:
    """Immutable identities needed to validate and record one paired report."""

    writer: MonitorEventWriter
    train_eval_members: list[tuple[str, int]]
    evaluation_members: list[tuple[str, int]]
    train_eval_mask: str
    evaluation_role: str
    evaluation_mask: str
    metric_context: dict[str, object]
    train_eval_context: dict[str, object]
    evaluation_context: dict[str, object]


def evaluate_and_record_epoch(
    model: nn.Module,
    train_eval: SequenceDataset,
    evaluation_set: SequenceDataset,
    chunk_rows: int,
    evaluation_batch_size: int,
    precision_mode: str,
    context: DistributedContext,
    epoch: int,
    step: int,
    event_context: EvaluationEventContext,
) -> dict[str, int | float]:
    """Validate both memberships, score them, and append a role-labelled event."""
    role = event_context.evaluation_role
    validate_dataset_members(
        train_eval, event_context.train_eval_members, "train_eval", event_context.train_eval_mask
    )
    validate_dataset_members(
        evaluation_set,
        event_context.evaluation_members,
        role,
        event_context.evaluation_mask,
    )
    train_metrics, role_metrics = evaluate_monitoring_sets(
        model,
        train_eval,
        evaluation_set,
        chunk_rows,
        precision_mode,
        context,
        evaluation_batch_size,
        evaluation_prefix=role,
    )
    if train_metrics["train_eval/sequence_count"] != len(event_context.train_eval_members):
        raise ValueError("train-eval evaluation did not cover its complete fit cohort")
    if role_metrics[f"{role}/sequence_count"] != len(event_context.evaluation_members):
        raise ValueError("evaluation did not cover its complete source-qualified membership")
    _add_score_aliases(train_metrics, role_metrics, role)
    event_context.writer.append(
        "evaluation",
        step,
        epoch + 1,
        train_metrics | role_metrics,
        event_context.metric_context
        | {
            "train_eval": event_context.train_eval_context
            | {
                "coverage_complete": True,
                "batch_size": evaluation_batch_size,
            },
            role: event_context.evaluation_context
            | {
                "coverage_complete": True,
                "selected_rows": role_metrics[f"{role}/selected_rows"],
                "batch_size": evaluation_batch_size,
            },
        },
    )
    return train_metrics | role_metrics


def _add_score_aliases(
    train_metrics: dict[str, int | float],
    role_metrics: dict[str, int | float],
    role: str,
) -> None:
    """Add plotting-friendly score aliases while retaining the raw WP fields."""
    train_metrics["train_eval/score"] = train_metrics["train_eval/wp"]
    train_metrics["train_eval/score/t0"] = train_metrics["train_eval/wp/t0"]
    train_metrics["train_eval/score/t1"] = train_metrics["train_eval/wp/t1"]
    role_metrics[f"{role}/score"] = role_metrics[f"{role}/wp"]
    role_metrics[f"{role}/score/t0"] = role_metrics[f"{role}/wp/t0"]
    role_metrics[f"{role}/score/t1"] = role_metrics[f"{role}/wp/t1"]


def evaluate_monitoring_sets(
    model: nn.Module,
    train_eval: SequenceDataset,
    valid: SequenceDataset,
    chunk_rows: int,
    precision_mode: str,
    context: DistributedContext,
    evaluation_batch_size: int = 1,
    *,
    evaluation_prefix: str = "valid",
    blend_alpha: float | None = None,
) -> tuple[dict[str, int | float], dict[str, int | float]]:
    """Evaluate both declared cohorts while restoring model mode, buffers, and RNG."""
    modes = [(module, module.training) for module in model.modules()]
    bn_buffers = _batch_norm_buffers(model)
    rng = _rng_state(context)
    started = time.perf_counter()
    try:
        model.eval()
        with torch.inference_mode():
            train_started = time.perf_counter()
            train_metrics = evaluate_dataset(
                model,
                train_eval,
                "train_eval",
                chunk_rows,
                precision_mode,
                context,
                evaluation_batch_size,
                blend_alpha=blend_alpha,
            )
            train_metrics["timing/train_eval_seconds"] = time.perf_counter() - train_started
            valid_started = time.perf_counter()
            valid_metrics = evaluate_dataset(
                model,
                valid,
                evaluation_prefix,
                chunk_rows,
                precision_mode,
                context,
                evaluation_batch_size,
                blend_alpha=blend_alpha,
            )
            valid_metrics[f"timing/{evaluation_prefix}_seconds"] = (
                time.perf_counter() - valid_started
            )
    finally:
        for module, training in modes:
            module.training = training
        _restore_rng(rng, context)
    _assert_batch_norm_unchanged(bn_buffers)
    elapsed = time.perf_counter() - started
    train_metrics["timing/paired_evaluation_seconds"] = elapsed
    valid_metrics["timing/paired_evaluation_seconds"] = elapsed
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
    *,
    blend_alpha: float | None = None,
) -> dict[str, int | float]:
    """Compute the cohort-mask weighted MSE and globally pooled WP."""
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
        prediction_chunks: list[torch.Tensor] = []
        for start in range(0, inputs.shape[1], chunk_rows):
            stop = min(start + chunk_rows, inputs.shape[1])
            values = inputs[:, start:stop].to(context.device, non_blocking=True)
            with forward_context(precision_mode, context.device.type):
                prediction, _, next_state = model(values, state)
            state = detach_state(next_state)
            prediction_chunks.append(prediction.float())
        # One device-to-host copy per sequence batch avoids synchronizing for every
        # short recurrent chunk while preserving the full pooled scoring inputs.
        predicted_tensor = torch.cat(prediction_chunks, dim=1).cpu()
        if blend_alpha is not None:
            predicted_tensor = compose_prediction(predicted_tensor, batch["teacher"], blend_alpha)
        predicted = predicted_tensor.numpy()
        target = targets.numpy()
        scored = mask.numpy()
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
        raise ValueError(f"{prefix} monitoring cohort has no selected rows")
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
