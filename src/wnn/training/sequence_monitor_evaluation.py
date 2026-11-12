from __future__ import annotations

import random
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_utils import SequenceDataset
from wnn.training.sequence_monitor_dataset import evaluate_dataset
from wnn.training.sequence_monitor_events import MonitorEventWriter
from wnn.training.sequence_monitor_head_metrics import add_score_aliases as _add_score_aliases
from wnn.training.sequence_monitor_identity import validate_dataset_members


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
    report_standalone: bool = False,
    since_diagnostics: bool = False,
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
                report_standalone=report_standalone,
                since_diagnostics=since_diagnostics,
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
                report_standalone=report_standalone,
                since_diagnostics=since_diagnostics,
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
