from __future__ import annotations

from collections.abc import Sequence

import torch


def target_loss_parts(
    prediction: torch.Tensor, target: torch.Tensor, scored: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one canonical pooled WMSE into two additive target contributions.

    Both losses use the joint clipped-target weight mass as their denominator,
    so their sum is the unchanged canonical ``weighted_mse_loss``.
    """
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("prediction and target must have matching [batch,time,2] shapes")
    if scored.shape != prediction.shape[:2] or scored.dtype != torch.bool:
        raise ValueError("scored must be a bool [batch,time] mask")
    clipped = target.float().clamp(-2.0, 2.0)
    weights = clipped.abs() * scored[..., None]
    weight = weights.sum()
    if not torch.isfinite(weight) or float(weight.detach()) < 1e-8:
        raise ValueError("diagnostic window must exceed the canonical WMSE denominator floor")
    squared_error = (prediction.float() - clipped).square()
    parts = tuple(
        (weights[..., index] * squared_error[..., index]).sum() / weight for index in range(2)
    )
    if not all(torch.isfinite(part) for part in parts):
        raise FloatingPointError("nonfinite target WMSE contribution")
    return parts[0], parts[1], weight, clipped


def _inner(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return an FP64 inner product without retaining a flattened gradient vector."""
    if len(left) != len(right):
        raise ValueError("gradient parameter inventories differ")
    total = torch.zeros((), dtype=torch.float64)
    for left_tensor, right_tensor in zip(left, right, strict=True):
        if left_tensor.shape != right_tensor.shape:
            raise ValueError("gradient tensor shapes differ")
        total += (left_tensor.detach().double() * right_tensor.detach().double()).sum()
    return total


def pair_summary(
    gradient0: Sequence[torch.Tensor], gradient1: Sequence[torch.Tensor]
) -> dict[str, float | int | None]:
    """Summarize two target gradients in a parameter block using scalar outputs only."""
    norm0_sq = float(_inner(gradient0, gradient0))
    norm1_sq = float(_inner(gradient1, gradient1))
    dot = float(_inner(gradient0, gradient1))
    norm0, norm1 = norm0_sq**0.5, norm1_sq**0.5
    cosine = dot / (norm0 * norm1) if norm0 > 0.0 and norm1 > 0.0 else None
    return {
        "target0_gradient_norm": norm0,
        "target1_gradient_norm": norm1,
        "target0_over_target1_norm": norm0 / norm1 if norm1 > 0.0 else None,
        "gradient_inner_product": dot,
        "gradient_cosine": cosine,
        "cross_effect_sign_under_other_target_descent": (
            -1 if dot > 0.0 else 1 if dot < 0.0 else 0
        ),
    }


def add_weighted_pair(
    running0: Sequence[torch.Tensor],
    running1: Sequence[torch.Tensor],
    gradient0: Sequence[torch.Tensor],
    gradient1: Sequence[torch.Tensor],
    weight: float,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, float]]:
    """Add ``weight * gradient`` to FP64 pools and emit reconstructible scalar cross-products."""
    if not (weight > 0.0):
        raise ValueError("pool weight must be positive")
    if not (len(running0) == len(running1) == len(gradient0) == len(gradient1)):
        raise ValueError("pooled gradient inventories differ")
    weighted0 = tuple(grad.detach().double() * weight for grad in gradient0)
    weighted1 = tuple(grad.detach().double() * weight for grad in gradient1)
    audit = {
        "weighted_target0_norm_sq": float(_inner(weighted0, weighted0)),
        "weighted_target1_norm_sq": float(_inner(weighted1, weighted1)),
        "weighted_target_inner_product": float(_inner(weighted0, weighted1)),
        "cross_target0_with_previous0": float(_inner(weighted0, running0)),
        "cross_target0_with_previous1": float(_inner(weighted0, running1)),
        "cross_target1_with_previous0": float(_inner(weighted1, running0)),
        "cross_target1_with_previous1": float(_inner(weighted1, running1)),
    }
    next0 = tuple(previous + current for previous, current in zip(running0, weighted0, strict=True))
    next1 = tuple(previous + current for previous, current in zip(running1, weighted1, strict=True))
    return next0, next1, audit


def pooled_summary(
    pooled0: Sequence[torch.Tensor], pooled1: Sequence[torch.Tensor], total_weight: float
) -> dict[str, float | None]:
    """Return statistics of the joint-weighted pooled target gradients."""
    if total_weight <= 0.0:
        raise ValueError("pooled weight must be positive")
    norm0 = float(_inner(pooled0, pooled0)) ** 0.5 / total_weight
    norm1 = float(_inner(pooled1, pooled1)) ** 0.5 / total_weight
    dot = float(_inner(pooled0, pooled1)) / (total_weight**2)
    return {
        "pooled_target0_gradient_norm": norm0,
        "pooled_target1_gradient_norm": norm1,
        "pooled_gradient_inner_product": dot,
        "pooled_gradient_cosine": dot / (norm0 * norm1) if norm0 > 0 and norm1 > 0 else None,
    }
