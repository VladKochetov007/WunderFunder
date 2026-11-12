"""Label-only future-horizon WMSE sums and distributed scaling for card07."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

FUTURE_OFFSETS = (1, 16, 64)
TARGETS = 2
FUTURE_HEAD_COEFFICIENT = 0.1


@dataclass(frozen=True)
class FutureObjectiveSums:
    """Local horizon-by-target WMSE numerators, masses, and valid-element counts."""

    loss_numerator: Tensor
    weight_mass: Tensor
    eligible_count: Tensor


def future_window_denominators(
    target: Tensor, source_mask: Tensor, start: int, stop: int
) -> Tensor:
    """Return local clipped-label weight mass with shape ``[3, 2]``.

    Sources are restricted by the primary mask. Each horizon target is read from
    the same sequence, including beyond the model chunk, and must be in bounds
    and finite. The caller sums chunks and targets, then all-reduces per horizon.
    """
    clipped, valid = _future_targets(target, source_mask, start, stop)
    return (clipped.abs() * valid.to(dtype=torch.float32)).sum(dim=(0, 1))


def future_objective_sums(
    predictions: Tensor, target: Tensor, source_mask: Tensor, start: int, stop: int
) -> FutureObjectiveSums:
    """Compute differentiable WMSE sums for the three label-only future offsets."""
    clipped, valid = _future_targets(target, source_mask, start, stop)
    expected_shape = (target.shape[0], stop - start, len(FUTURE_OFFSETS), TARGETS)
    if predictions.shape != expected_shape or predictions.device != target.device:
        raise ValueError("future predictions must match [batch, chunk, 3, 2] and target device")
    safe_prediction = torch.where(valid, predictions.float(), torch.zeros_like(predictions))
    weight = clipped.abs() * valid.to(dtype=torch.float32)
    numerator = ((safe_prediction - clipped).square() * weight).sum(dim=(0, 1))
    mass = weight.sum(dim=(0, 1))
    count = valid.sum(dim=(0, 1), dtype=torch.int64)
    return FutureObjectiveSums(numerator, mass, count)


def future_scaled_loss(
    sums: FutureObjectiveSums, global_horizon_mass: Tensor, world_size: int
) -> Tensor:
    """Apply source-equivalent pooled-target WMSE and the fixed 0.1 horizon mean.

    ``global_horizon_mass`` has one denominator per horizon, pooled across both
    targets. Multiplication by ``world_size`` compensates for DDP's gradient
    averaging; chunks and ranks must use the same complete-window denominators.
    """
    if sums.loss_numerator.shape != (len(FUTURE_OFFSETS), TARGETS):
        raise ValueError("future numerator diagnostics must have shape [3, 2]")
    if global_horizon_mass.shape != (len(FUTURE_OFFSETS),) or world_size < 1:
        raise ValueError("future loss requires three pooled denominators and positive world size")
    denominator = global_horizon_mass.to(
        device=sums.loss_numerator.device, dtype=sums.loss_numerator.dtype
    )
    active = denominator > 0
    safe_denominator = torch.where(active, denominator, torch.ones_like(denominator))
    horizon_numerator = sums.loss_numerator.sum(dim=1)
    normalized = torch.where(
        active,
        horizon_numerator * (world_size / safe_denominator),
        torch.zeros_like(horizon_numerator),
    )
    return FUTURE_HEAD_COEFFICIENT * normalized.mean()


def _future_targets(
    target: Tensor, source_mask: Tensor, start: int, stop: int
) -> tuple[Tensor, Tensor]:
    """Build safe clipped labels and per-element eligibility for a source interval."""
    if target.ndim != 3 or target.shape[-1] != TARGETS or target.shape[1] <= 0:
        raise ValueError("future objective requires raw labels shaped [batch, rows, 2]")
    if source_mask.shape != target.shape[:2] or source_mask.dtype != torch.bool:
        raise ValueError("future objective requires a Boolean source mask shaped [batch, rows]")
    if source_mask.device != target.device:
        raise ValueError("future labels and source mask must use the same device")
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= target.shape[1]:
        raise ValueError("future source interval must be nonempty and within one sequence")

    source_indices = torch.arange(start, stop, device=target.device)
    source_valid = source_mask[:, start:stop].unsqueeze(-1)
    clipped_targets: list[Tensor] = []
    eligible_targets: list[Tensor] = []
    last_index = target.shape[1] - 1
    for offset in FUTURE_OFFSETS:
        future_indices = source_indices + offset
        in_bounds = (future_indices <= last_index).view(1, -1, 1)
        safe_indices = future_indices.clamp(max=last_index)
        future = target.index_select(1, safe_indices).float()
        eligible = source_valid & in_bounds & torch.isfinite(future)
        safe_future = torch.where(eligible, future, torch.zeros_like(future))
        clipped_targets.append(safe_future.clamp(-2.0, 2.0))
        eligible_targets.append(eligible)
    return torch.stack(clipped_targets, dim=-2), torch.stack(eligible_targets, dim=-2)
