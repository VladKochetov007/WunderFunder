from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel

from wnn.training.sequence_auxiliary import (
    SequenceAuxiliaryTargets,
    auxiliary_denominators,
    auxiliary_loss_sums,
)
from wnn.training.sequence_distributed import (
    DistributedContext,
    all_ranks_true,
    global_sum,
)
from wnn.training.sequence_models import detach_state
from wnn.training.sequence_objectives import (
    detached_sums,
    objective_numerators,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_precision import backward_context, forward_context


@dataclass(frozen=True)
class EpochResult:
    """Completed or bounded sequence-pass metrics, including aggregate sufficient sums."""

    updates: int
    complete: bool
    totals: torch.Tensor
    elapsed_seconds: float
    objective_totals: torch.Tensor | None = None
    auxiliary_totals: torch.Tensor | None = None


def _check_finite_loss(loss: torch.Tensor, context: DistributedContext) -> None:
    local_finite = bool(torch.isfinite(loss.detach()).all())
    if not all_ranks_true(local_finite, context):
        raise FloatingPointError("nonfinite objective detected consistently across torchrun ranks")


def _backward_window(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    target: torch.Tensor,
    teacher: torch.Tensor,
    hint_target: torch.Tensor,
    mask: torch.Tensor,
    x: torch.Tensor,
    label_row_weights: torch.Tensor | None,
    intervals: list[tuple[int, int]],
    real_sequence_count: int,
    denominators: torch.Tensor,
    auxiliary_targets: SequenceAuxiliaryTargets | None,
    state: Any,  # Model-owned recurrent state has architecture-specific container structure.
    config: dict[str, object],
    context: DistributedContext,
    objective_stats: torch.Tensor | None = None,
) -> tuple[Any, torch.Tensor]:
    """Forward and backpropagate one pre-normalized TBPTT accumulation window."""
    local_stats = torch.zeros(
        8 if auxiliary_targets is not None else 6,
        dtype=torch.float64,
        device=context.device,
    )
    for index, (start, stop) in enumerate(intervals):
        final_microchunk = index == len(intervals) - 1
        sync_context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not final_microchunk
            else nullcontext()
        )
        attention_backend = str(config["model"].get("attention_backend", "math"))
        attention_context = (
            sdpa_kernel(SDPBackend.MATH) if attention_backend == "math" else nullcontext()
        )
        mode = str(config.get("precision_mode", "fp32"))
        with sync_context, attention_context:
            with forward_context(mode, context.device.type):
                prediction, hint_prediction, next_state = model(x[:, start:stop], state)
            state = detach_state(next_state)
            prediction = prediction.float()
            hint_prediction = hint_prediction.float()
            numerators = objective_numerators(
                prediction[..., :2],
                hint_prediction,
                target[:, start:stop],
                teacher[:, start:stop],
                hint_target[:, start:stop],
                mask[:, start:stop],
                blend_objective_alpha=float(config.get("blend_objective_alpha", 0.0)),
                objective_mode=config.get("objective_mode"),
                context=context,
                correlation_prediction_std_floor=float(
                    config.get("correlation_prediction_std_floor", 0.01)
                ),
                supervised_target=config.get("supervised_target"),
                label_row_weights=(
                    label_row_weights[:, start:stop] if label_row_weights is not None else None
                ),
            )
            loss = scaled_objective(
                numerators,
                denominators,
                context.world_size,
                float(config["kd_weight"]),
                float(config["hint_weight"]),
                microchunks=len(intervals),
            )
            auxiliary_sums = auxiliary_loss_sums(prediction, auxiliary_targets, start, stop)
            if auxiliary_targets is not None:
                structure_denominator = float(denominators[2].item())
                next_denominator = float(denominators[3].item())
                structure_scale = (
                    context.world_size / structure_denominator if structure_denominator > 0 else 0.0
                )
                next_scale = context.world_size / next_denominator if next_denominator > 0 else 0.0
                loss = loss + auxiliary_sums[0] * (
                    float(config["aux_structure_weight"]) * structure_scale
                )
                loss = loss + auxiliary_sums[1] * (
                    float(config["aux_next_change_weight"]) * next_scale
                )
            _check_finite_loss(loss, context)
            with backward_context(mode, context.device.type):
                loss.backward()
            if objective_stats is not None and numerators.correlation is not None:
                objective_stats[:3] += torch.cat(
                    (
                        numerators.correlation.detach(),
                        numerators.correlation_loss.detach().reshape(1),
                    )
                ).double() / len(intervals)
        local_stats[:5] += detached_sums(numerators, target[:, start:stop], mask[:, start:stop])
        local_stats[5] += real_sequence_count * (stop - start)
        if auxiliary_targets is not None:
            local_stats[6:8] += auxiliary_sums.detach().double()
    return state, local_stats


def _window_denominators(
    target: torch.Tensor,
    mask: torch.Tensor,
    hints: torch.Tensor,
    intervals: list[tuple[int, int]],
    auxiliary_targets: SequenceAuxiliaryTargets | None,
    context: DistributedContext,
) -> torch.Tensor:
    """Reduce all objective denominators across every rank before one backward window."""
    local = torch.zeros(4, dtype=torch.float32, device=context.device)
    for start, stop in intervals:
        local[:2] += window_denominators(
            target[:, start:stop], mask[:, start:stop], hints[:, start:stop]
        )
    local[2:] += auxiliary_denominators(auxiliary_targets, intervals, context.device)
    global_denominators = global_sum(local, context)
    if not all_ranks_true(bool(torch.isfinite(global_denominators).all()), context):
        raise FloatingPointError("nonfinite objective denominator detected across torchrun ranks")
    return global_denominators
