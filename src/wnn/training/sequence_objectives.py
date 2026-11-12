from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from wnn.features import causal_since_i0px_torch
from wnn.training.sequence_blend import compose_prediction
from wnn.training.sequence_distributed import (
    DistributedContext,
    all_ranks_true,
    global_sum,
)
from wnn.training.sequence_ensemble_config import (
    MIXTURE_MODE,
    STANDALONE_LABELS_KD_MODE,
    STANDALONE_LABELS_MODE,
)
from wnn.training.sequence_ensemble_correlation import weighted_microbatch_correlation


@dataclass(frozen=True)
class ObjectiveNumerators:
    """Unnormalized local objective sums for one recurrent chunk."""

    label: torch.Tensor
    distillation: torch.Tensor
    hints: torch.Tensor
    correlation: torch.Tensor | None = None
    correlation_loss: torch.Tensor | None = None


def window_denominators(
    target: torch.Tensor, mask: torch.Tensor, hint_target: torch.Tensor
) -> torch.Tensor:
    """Compute label and hint denominators before a distributed accumulation window."""
    clipped = torch.where(mask[..., None], target.float(), 0).clamp(-2.0, 2.0)
    row_mask = mask.to(dtype=torch.float32)
    label_mass = (clipped.abs() * row_mask[..., None]).sum()
    hint_mass = row_mask.sum() * hint_target.shape[-1]
    return torch.stack((label_mass, hint_mass))


def global_window_denominators(
    target: torch.Tensor,
    mask: torch.Tensor,
    hint_target: torch.Tensor,
    intervals: list[tuple[int, int]],
    context: DistributedContext,
    *,
    window_denominator: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    | None = None,
) -> torch.Tensor:
    """Reduce shared loss denominators over every chunk and rank before backward."""
    local = torch.zeros(2, dtype=torch.float32, device=context.device)
    denominator_fn = window_denominator or window_denominators
    for start, stop in intervals:
        local += denominator_fn(
            target[:, start:stop], mask[:, start:stop], hint_target[:, start:stop]
        )
    denominators = global_sum(local, context)
    if not all_ranks_true(bool(torch.isfinite(denominators).all()), context):
        raise FloatingPointError("nonfinite objective denominator detected across torchrun ranks")
    return denominators


def since_event_row_weights(inputs: torch.Tensor, tau: float, floor: float) -> torch.Tensor:
    """Compute causal, input-only main-label weights from the current i0 price-event age."""
    if (
        isinstance(tau, bool)
        or isinstance(floor, bool)
        or not math.isfinite(tau)
        or not math.isfinite(floor)
        or tau <= 0
        or not 0 < floor <= 1
    ):
        raise ValueError("since-event weighting requires positive tau and floor in (0, 1]")
    age = causal_since_i0px_torch(inputs).float()
    return torch.clamp(1.0 / (1.0 + age / tau), min=floor)


def objective_numerators(
    prediction: torch.Tensor,
    hint_prediction: torch.Tensor,
    target: torch.Tensor,
    teacher: torch.Tensor,
    hint_target: torch.Tensor,
    mask: torch.Tensor,
    *,
    blend_objective_alpha: float = 0.0,
    objective_mode: str | None = None,
    context: DistributedContext | None = None,
    correlation_prediction_std_floor: float = 0.01,
    supervised_target: int | None = None,
    label_row_weights: torch.Tensor | None = None,
) -> ObjectiveNumerators:
    """Return clipped weighted label/KD sums and a masked elementwise hint sum."""
    if objective_mode == STANDALONE_LABELS_KD_MODE:
        if blend_objective_alpha != 0 or hint_prediction.shape[-1] or hint_target.shape[-1]:
            raise ValueError("standalone label+KD objective forbids blends and hidden hints")
        if any(value.dtype != torch.float32 for value in (prediction, target, teacher)):
            raise TypeError("standalone label+KD objective requires strict FP32 tensors")
    if supervised_target is not None:
        if type(supervised_target) is not int or supervised_target not in (0, 1):
            raise ValueError("supervised_target must be null, 0, or 1")
        if objective_mode != STANDALONE_LABELS_MODE or blend_objective_alpha != 0:
            raise ValueError("single-target supervision requires the standalone label objective")
    raw_target = target
    if objective_mode is not None:
        selected = mask[..., None]
        finite = bool(torch.isfinite(prediction[mask]).all() and torch.isfinite(target[mask]).all())
        if objective_mode == STANDALONE_LABELS_KD_MODE:
            finite = finite and bool(torch.isfinite(teacher[mask]).all())
        if context is None or not all_ranks_true(finite, context):
            raise FloatingPointError(
                "new objective requires finite selected raw values on every rank"
            )
        prediction = torch.where(selected, prediction, 0)
        target = torch.where(selected, target, 0)
        teacher = torch.where(selected, teacher, 0)
        hint_prediction = torch.where(selected, hint_prediction, 0)
        hint_target = torch.where(selected, hint_target, 0)
    clipped = target.float().clamp(-2.0, 2.0)
    weight = clipped.abs() * mask.to(dtype=torch.float32)[..., None]
    label_weight = weight
    if label_row_weights is not None:
        if (
            label_row_weights.shape != mask.shape
            or label_row_weights.device != mask.device
            or not torch.is_floating_point(label_row_weights)
            or not bool(torch.isfinite(label_row_weights).all())
            or bool((label_row_weights < 0).any())
            or bool((label_row_weights > 1).any())
        ):
            raise ValueError("label row weights must be finite [B,T] values in [0, 1]")
        label_weight = weight * label_row_weights.float()[..., None]
    if supervised_target is not None:
        target_selector = torch.zeros_like(weight)
        target_selector[..., supervised_target] = 1
        weight = weight * target_selector
        label_weight = label_weight * target_selector
    label_error = prediction.float() - clipped
    correlation, correlation_loss = None, None
    if objective_mode == MIXTURE_MODE:
        if context is None or blend_objective_alpha != 0.1:
            raise ValueError("mixture Pearson requires explicit distributed context and alpha0.1")
        mixture = compose_prediction(prediction, teacher, blend_objective_alpha)
        label_error = mixture - clipped
        correlation, correlation_loss = weighted_microbatch_correlation(
            mixture, raw_target, mask, context, correlation_prediction_std_floor
        )
    elif blend_objective_alpha > 0:
        # Preserve original-label weights; alpha^-2 matches curvature with respect to the head.
        label_error = (
            compose_prediction(prediction, teacher, blend_objective_alpha) - clipped
        ) / blend_objective_alpha
    teacher_error = prediction.float() - teacher.detach().float().clamp(-2.0, 2.0)
    hint_error = hint_prediction.float() - hint_target.float()
    hint_weight = mask.to(dtype=torch.float32)[..., None]
    return ObjectiveNumerators(
        (label_error.square() * label_weight).sum(),
        (teacher_error.square() * weight).sum(),
        (hint_error.square() * hint_weight).sum(),
        correlation,
        correlation_loss,
    )


def scaled_objective(
    numerators: ObjectiveNumerators,
    denominators: torch.Tensor,
    world_size: int,
    kd_weight: float,
    hint_weight: float,
    *,
    microchunks: int = 1,
) -> torch.Tensor:
    """Scale local sums for DDP's gradient averaging across the full window."""
    label_denom = float(denominators[0].item())
    hint_denom = float(denominators[1].item())
    label_scale = world_size / label_denom if label_denom > 0 else 0.0
    hint_scale = world_size / hint_denom if hint_denom > 0 else 0.0
    loss = (
        numerators.label * label_scale
        + numerators.distillation * (kd_weight * label_scale)
        + numerators.hints * (hint_weight * hint_scale)
    )
    if numerators.correlation_loss is not None:
        if kd_weight or hint_weight or microchunks < 1:
            raise ValueError(
                "mixture Pearson permits no auxiliary losses and requires actual chunk count"
            )
        # Differentiable global SUM backward already sums identical losses across
        # ranks. DDP averages parameter gradients, so NO extra world-size factor.
        return 0.5 * loss + 0.5 * numerators.correlation_loss / microchunks
    return loss


def detached_sums(
    numerators: ObjectiveNumerators, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Pack chunk numerators and valid-row count for epoch telemetry reductions."""
    rows = mask.sum(dtype=torch.float64)
    return torch.stack(
        (
            numerators.label.detach().double(),
            numerators.distillation.detach().double(),
            numerators.hints.detach().double(),
            rows,
            torch.tensor(float(target.shape[0] * target.shape[1]), device=target.device),
        )
    )
