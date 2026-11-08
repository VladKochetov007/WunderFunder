"""Merge activation and optimizer probes into replayable diagnostics events."""

from __future__ import annotations

from collections.abc import Mapping

from wnn.optimizer_update_view import (
    flatten_optimizer_update_diagnostics,
    optimizer_update_group_context,
)
from wnn.training.sequence_monitor_events import MonitorEventWriter


def record_diagnostics_event(
    writer: MonitorEventWriter,
    step: int,
    epoch_index: int,
    epoch_position: float,
    hook_interval: int,
    hook_metrics: Mapping[str, int | float],
    optimizer_update: object = None,
) -> None:
    """Write at most one merged diagnostic event for each sampled update step."""
    metrics = dict(hook_metrics)
    metrics.update(flatten_optimizer_update_diagnostics(optimizer_update))
    context: dict[str, object] = {
        "sampling": f"every {hook_interval} optimizer steps",
        "scope": "rank-zero sampled forward/backward optimizer window",
        "missing_metrics": "unavailable model branches are omitted",
        "gradient_absence_semantics": (
            "named missing_output_gradient counts are output tensors without backward gradients; "
            "detached recurrent returned states may be expected. missing_parameter_gradient_count "
            "separately counts trainable parameters whose grad is None after backward."
        ),
        "hook_semantics": "tensor-hook measurements are detached; callbacks return None",
        "parameter_norm_sampling": "first 4096 trainable scalars in model parameter order",
        "epoch_index": epoch_index + 1,
    }
    if optimizer_update is not None:
        if not isinstance(optimizer_update, Mapping):
            raise ValueError("optimizer update diagnostics must be a mapping")
        context["optimizer_update_sampling"] = (
            f"one-based update 1 and each {optimizer_update.get('sampling_interval')} optimizer steps"
        )
        context["optimizer_update_groups"] = optimizer_update_group_context(optimizer_update)
    writer.append("diagnostics", step, epoch_position, metrics, context)
