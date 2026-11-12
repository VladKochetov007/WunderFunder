"""Masked, source-safe auxiliary objectives for PR81 GRU cards08–10."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

TARGETS = 2
NLL_COEFFICIENT = 0.05
EVENT_CE_COEFFICIENT = 0.05
PERTURBATION_COEFFICIENT = 0.01
PERTURBATION_SCALE_SQUARED = 0.01**2 + 1e-8


def _validate_masked_pair(values: Tensor, target: Tensor, mask: Tensor, label: str) -> None:
    """Check common target shapes and require finite requested training values."""
    if target.ndim != 3 or target.shape[-1] != TARGETS:
        raise ValueError(f"{label} target must have shape [batch,rows,2]")
    if mask.shape != target.shape[:-1] or mask.dtype != torch.bool or mask.device != target.device:
        raise ValueError(f"{label} mask must be Boolean [batch,rows] on the target device")
    if values.shape != target.shape or values.device != target.device:
        raise ValueError(f"{label} values must match the target shape and device")
    selected = mask.unsqueeze(-1)
    if not bool(torch.isfinite(target[selected.expand_as(target)]).all()):
        raise FloatingPointError(f"{label} selected raw targets are nonfinite")
    if not bool(torch.isfinite(values[selected.expand_as(values)]).all()):
        raise FloatingPointError(f"{label} selected predictions are nonfinite")


def perturbation_objective_sums(discrepancy: Tensor, mask_suffix: Tensor) -> dict[str, Tensor]:
    """Return per-target squared consistency sums and eligible element counts."""
    if discrepancy.ndim != 3 or discrepancy.shape[-1] != TARGETS:
        raise ValueError("card08 discrepancy must have shape [batch,32,2]")
    if mask_suffix.shape != discrepancy.shape[:-1] or mask_suffix.dtype != torch.bool:
        raise ValueError("card08 suffix mask must be Boolean [batch,32]")
    if mask_suffix.device != discrepancy.device:
        raise ValueError("card08 discrepancy and mask must share a device")
    selected = mask_suffix.unsqueeze(-1)
    if not bool(torch.isfinite(discrepancy[selected.expand_as(discrepancy)]).all()):
        raise FloatingPointError("card08 selected noisy/clean differences are nonfinite")
    safe = torch.where(selected, discrepancy.float(), torch.zeros_like(discrepancy.float()))
    numerator = safe.square().sum(dim=(0, 1))
    count = mask_suffix.sum(dtype=torch.float32).expand(TARGETS)
    return {"loss_numerator": numerator, "count": count}


def nll_objective_sums(
    prediction: Tensor, log_variance: Tensor, raw_target: Tensor, mask: Tensor
) -> dict[str, Tensor]:
    """Return clipped-label Gaussian NLL sums/counts and variance-bound diagnostics."""
    _validate_masked_pair(prediction, raw_target, mask, "card09 NLL")
    if log_variance.shape != raw_target.shape or log_variance.device != raw_target.device:
        raise ValueError("card09 log variance must match the two-target rows")
    selected = mask.unsqueeze(-1)
    if not bool(torch.isfinite(log_variance[selected.expand_as(log_variance)]).all()):
        raise FloatingPointError("card09 selected log variances are nonfinite")
    target = torch.where(selected, raw_target.float(), torch.zeros_like(raw_target.float()))
    mean = torch.where(selected, prediction.float(), torch.zeros_like(prediction.float()))
    raw_s = torch.where(selected, log_variance.float(), torch.zeros_like(log_variance.float()))
    s = raw_s.clamp(-5.0, 3.0)
    nll = 0.5 * torch.exp(-s) * (mean - target.clamp(-2.0, 2.0)).square() + 0.5 * s
    active = selected.expand_as(nll)
    return {
        "loss_numerator": torch.where(active, nll, torch.zeros_like(nll)).sum(dim=(0, 1)),
        "count": mask.sum(dtype=torch.float32).expand(TARGETS),
        "log_variance_sum": torch.where(active, s, torch.zeros_like(s)).sum(dim=(0, 1)),
        "lower_bound_count": ((raw_s <= -5.0) & active).sum(dim=(0, 1), dtype=torch.float32),
        "upper_bound_count": ((raw_s >= 3.0) & active).sum(dim=(0, 1), dtype=torch.float32),
    }


def clip_event_objective_sums(
    logits: Tensor, raw_target: Tensor, mask: Tensor
) -> dict[str, Tensor]:
    """Return raw-label three-class CE sums, class counts, and per-target confusion."""
    if raw_target.ndim != 3 or raw_target.shape[-1] != TARGETS:
        raise ValueError("card10 raw target must have shape [batch,rows,2]")
    if mask.shape != raw_target.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("card10 mask must be Boolean [batch,rows]")
    if logits.shape != (*raw_target.shape, 3) or logits.device != raw_target.device:
        raise ValueError("card10 logits must have shape [batch,rows,2,3]")
    if mask.device != logits.device:
        raise ValueError("card10 logits and mask must share a device")
    selected_rows = mask.unsqueeze(-1)
    if not bool(torch.isfinite(raw_target[selected_rows.expand_as(raw_target)]).all()):
        raise FloatingPointError("card10 selected raw training targets are nonfinite")
    selected_logits = selected_rows.unsqueeze(-1).expand_as(logits)
    if not bool(torch.isfinite(logits[selected_logits]).all()):
        raise FloatingPointError("card10 selected event logits are nonfinite")
    safe_target = torch.where(
        selected_rows, raw_target.float(), torch.zeros_like(raw_target.float())
    )
    classes = torch.where(
        safe_target < -2.0,
        0,
        torch.where(safe_target <= 2.0, 1, 2),
    ).to(torch.int64)
    safe_logits = torch.where(selected_logits, logits.float(), torch.zeros_like(logits.float()))
    per_element = F.cross_entropy(
        safe_logits.reshape(-1, 3), classes.reshape(-1), reduction="none"
    ).reshape_as(raw_target)
    active = selected_rows.expand_as(per_element)
    ce_by_class = torch.zeros((TARGETS, 3), dtype=torch.float32, device=logits.device)
    confusion = torch.zeros((TARGETS, 3, 3), dtype=torch.int64, device=logits.device)
    predicted = safe_logits.argmax(dim=-1)
    for target_index in range(TARGETS):
        for class_index in range(3):
            class_rows = mask & (classes[:, :, target_index] == class_index)
            ce_by_class[target_index, class_index] = per_element[:, :, target_index][
                class_rows
            ].sum()
        truth = classes[:, :, target_index][mask]
        guess = predicted[:, :, target_index][mask]
        confusion[target_index] = torch.bincount(truth * 3 + guess, minlength=9).reshape(3, 3)
    return {
        "loss_numerator": torch.where(active, per_element, torch.zeros_like(per_element)).sum(
            dim=(0, 1)
        ),
        "count": mask.sum(dtype=torch.float32).expand(TARGETS),
        "ce_by_class": ce_by_class,
        "confusion": confusion,
    }


def gru_scaled_loss(
    sums: Mapping[str, Tensor], global_count: Tensor, world_size: int, variant: str
) -> Tensor:
    """Scale one auxiliary objective by globally eligible target elements."""
    if variant not in ("state_perturbation", "nll_transfer", "clip_event"):
        raise ValueError(f"unknown PR81 GRU auxiliary objective {variant!r}")
    numerator = sums.get("loss_numerator")
    if not isinstance(numerator, Tensor) or numerator.shape != (TARGETS,):
        raise ValueError("GRU auxiliary sums must contain a two-target loss_numerator")
    if global_count.shape != (TARGETS,) or global_count.device != numerator.device:
        raise ValueError("GRU global eligible counts must be a two-target vector on-device")
    if (
        world_size < 1
        or not bool(torch.isfinite(global_count).all())
        or bool((global_count < 0).any())
    ):
        raise ValueError("GRU global eligible count or world size is invalid")
    count = global_count.sum().to(dtype=numerator.dtype)
    if not bool(count > 0):
        return numerator.sum() * 0.0
    coefficient = (
        PERTURBATION_COEFFICIENT
        if variant == "state_perturbation"
        else NLL_COEFFICIENT
        if variant == "nll_transfer"
        else EVENT_CE_COEFFICIENT
    )
    divisor = PERTURBATION_SCALE_SQUARED if variant == "state_perturbation" else 1.0
    loss = coefficient * world_size * numerator.sum() / count / divisor
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"{variant} auxiliary loss is nonfinite")
    return loss
