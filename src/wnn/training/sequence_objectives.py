from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObjectiveNumerators:
    """Unnormalized local objective sums for one recurrent chunk."""

    label: torch.Tensor
    distillation: torch.Tensor
    hints: torch.Tensor


def window_denominators(
    target: torch.Tensor, mask: torch.Tensor, hint_target: torch.Tensor
) -> torch.Tensor:
    """Compute label and hint denominators before a distributed accumulation window."""
    clipped = target.float().clamp(-2.0, 2.0)
    row_mask = mask.to(dtype=torch.float32)
    label_mass = (clipped.abs() * row_mask[..., None]).sum()
    hint_mass = row_mask.sum() * hint_target.shape[-1]
    return torch.stack((label_mass, hint_mass))


def objective_numerators(
    prediction: torch.Tensor,
    hint_prediction: torch.Tensor,
    target: torch.Tensor,
    teacher: torch.Tensor,
    hint_target: torch.Tensor,
    mask: torch.Tensor,
) -> ObjectiveNumerators:
    """Return clipped weighted label/KD sums and a masked elementwise hint sum."""
    clipped = target.float().clamp(-2.0, 2.0)
    weight = clipped.abs() * mask.to(dtype=torch.float32)[..., None]
    label_error = prediction.float() - clipped
    teacher_error = prediction.float() - teacher.float().clamp(-2.0, 2.0)
    hint_error = hint_prediction.float() - hint_target.float()
    hint_weight = mask.to(dtype=torch.float32)[..., None]
    return ObjectiveNumerators(
        (label_error.square() * weight).sum(),
        (teacher_error.square() * weight).sum(),
        (hint_error.square() * hint_weight).sum(),
    )


def scaled_objective(
    numerators: ObjectiveNumerators,
    denominators: torch.Tensor,
    world_size: int,
    kd_weight: float,
    hint_weight: float,
) -> torch.Tensor:
    """Scale local sums for DDP's gradient averaging across the full window."""
    label_denom = float(denominators[0].item())
    hint_denom = float(denominators[1].item())
    label_scale = world_size / label_denom if label_denom > 0 else 0.0
    hint_scale = world_size / hint_denom if hint_denom > 0 else 0.0
    return (
        numerators.label * label_scale
        + numerators.distillation * (kd_weight * label_scale)
        + numerators.hints * (hint_weight * hint_scale)
    )


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
