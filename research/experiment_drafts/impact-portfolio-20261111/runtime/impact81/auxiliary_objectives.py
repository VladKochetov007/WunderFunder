"""Pure tensor objectives for PR81 cards04 (depth supervision) and06 (ranking)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

DEEP_SUPERVISION_COEFFICIENT = 0.1
DEEP_SUPERVISION_HEAD_COUNT = 3  # Outputs after Transformer blocks 6, 12 and 18.
PAIRWISE_COEFFICIENT = 0.05
PAIRWISE_LAG_ROWS = 32
PAIRWISE_MIN_LABEL_DELTA = 0.5
PAIRWISE_TEMPERATURE = 1.0
TARGET_COUNT = 2


@dataclass(frozen=True)
class PairwiseObjectiveSums:
    """Local per-target pairwise sums for one sequence chunk.

    The loop reduces floating sums and integer counts over the full accumulation
    window before logging. `concordance_mass` awards half weight to exact ties.
    """

    loss_numerator: Tensor
    weight_denominator: Tensor
    eligible_count: Tensor
    concordant_count: Tensor
    tie_count: Tensor
    concordance_mass: Tensor


def _validate_targets(target: Tensor, mask: Tensor) -> None:
    """Require the fixed two-target sequence shape and a Boolean primary mask."""
    if target.ndim != 3 or target.shape[-1] != TARGET_COUNT:
        raise ValueError("auxiliary objectives require target shape [batch, rows, 2]")
    if mask.shape != target.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("auxiliary objectives require the Boolean primary row mask")
    if mask.device != target.device:
        raise ValueError("target and primary mask must be on the same device")


def deep_supervision_wmse_numerators(
    predictions: Sequence[Tensor], target: Tensor, mask: Tensor
) -> Tensor:
    """Return three local WMSE sums using the exact primary target mask/weights.

    The returned order is blocks6,12,18. The caller divides each by the global
    primary weighted-target mass for the complete accumulation window.
    """
    _validate_targets(target, mask)
    if len(predictions) != DEEP_SUPERVISION_HEAD_COUNT:
        raise ValueError("card04 requires heads after blocks 6, 12 and 18")
    selected = mask.unsqueeze(-1)
    safe_target = torch.where(selected, target.float(), torch.zeros_like(target))
    clipped = safe_target.clamp(-2.0, 2.0)
    weights = clipped.abs() * mask.to(dtype=torch.float32).unsqueeze(-1)
    numerators = []
    for prediction in predictions:
        if prediction.shape != target.shape or prediction.device != target.device:
            raise ValueError("each intermediate head must match the target shape/device")
        safe_prediction = torch.where(selected, prediction.float(), torch.zeros_like(prediction))
        error = safe_prediction - clipped
        numerators.append((error.square() * weights).sum())
    return torch.stack(numerators)


def deep_supervision_scaled_loss(
    head_numerators: Tensor, global_primary_weight: Tensor, world_size: int
) -> Tensor:
    """Scale the three heads by the global primary denominator and apply 0.1 mean."""
    if head_numerators.shape != (DEEP_SUPERVISION_HEAD_COUNT,):
        raise ValueError("card04 numerator vector must contain exactly three heads")
    if global_primary_weight.numel() != 1 or world_size < 1:
        raise ValueError("card04 requires one global denominator and positive world size")
    denominator = global_primary_weight.to(
        device=head_numerators.device, dtype=head_numerators.dtype
    )
    active = denominator > 0
    safe_denominator = torch.where(active, denominator, torch.ones_like(denominator))
    normalized = torch.where(
        active,
        head_numerators * (world_size / safe_denominator),
        torch.zeros_like(head_numerators),
    )
    return DEEP_SUPERVISION_COEFFICIENT * normalized.mean()


def _pairwise_label_terms(target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Build only eligible within-chunk target pairs and their clipped-label weights."""
    _validate_targets(target, mask)
    rows = target.shape[1]
    if rows <= PAIRWISE_LAG_ROWS:
        empty_shape = (target.shape[0], 0, TARGET_COUNT)
        empty_float = target.new_empty(empty_shape, dtype=torch.float32)
        empty_mask = mask.new_empty((target.shape[0], 0, TARGET_COUNT))
        return empty_float, empty_float, empty_mask
    safe_target = torch.where(mask.unsqueeze(-1), target.float(), torch.zeros_like(target))
    clipped = safe_target.clamp(-2.0, 2.0)
    current = clipped[:, PAIRWISE_LAG_ROWS:, :]
    previous = clipped[:, :-PAIRWISE_LAG_ROWS, :]
    both_after_warmup = (mask[:, PAIRWISE_LAG_ROWS:] & mask[:, :-PAIRWISE_LAG_ROWS]).unsqueeze(-1)
    eligible = both_after_warmup & ((current - previous).abs() >= PAIRWISE_MIN_LABEL_DELTA)
    pair_weight = (current.abs() * previous.abs()).sqrt()
    pair_weight = torch.where(eligible, pair_weight, torch.zeros_like(pair_weight))
    return current - previous, pair_weight, eligible


def pairwise_window_denominators(target: Tensor, mask: Tensor) -> Tensor:
    """Return local eligible-pair weight by target for one chunk.

    The loop sums these vectors over all chunks in an accumulation window and
    all-reduces once before backward; pairs never cross a chunk or sequence edge.
    """
    _, pair_weight, _ = _pairwise_label_terms(target, mask)
    return pair_weight.sum(dim=(0, 1))


def pairwise_objective_sums(
    prediction: Tensor, target: Tensor, mask: Tensor
) -> PairwiseObjectiveSums:
    """Return pair-loss sums and training-only concordance diagnostics by target.

    For each chunk, row t is paired only with row t−32 in that same sequence.
    Both rows must satisfy the supplied primary mask, and clipped-label delta
    magnitude must be at least0.5. Prediction ties count as half concordant mass.
    """
    _validate_targets(target, mask)
    if prediction.shape != target.shape or prediction.device != target.device:
        raise ValueError("pairwise predictions must match the target shape/device")
    label_delta, pair_weight, eligible = _pairwise_label_terms(target, mask)
    if label_delta.shape[1] == 0:
        safe_prediction = torch.where(
            mask.unsqueeze(-1), prediction.float(), torch.zeros_like(prediction)
        )
        loss_numerator = safe_prediction.sum(dim=(0, 1)) * 0.0
        zeros = torch.zeros(TARGET_COUNT, device=target.device, dtype=torch.float32)
        counts = torch.zeros(TARGET_COUNT, device=target.device, dtype=torch.int64)
        return PairwiseObjectiveSums(loss_numerator, zeros, counts, counts, counts, zeros)

    safe_prediction = torch.where(
        mask.unsqueeze(-1), prediction.float(), torch.zeros_like(prediction)
    )
    prediction_delta = (
        safe_prediction[:, PAIRWISE_LAG_ROWS:, :] - safe_prediction[:, :-PAIRWISE_LAG_ROWS, :]
    )
    selected_label_delta = torch.where(eligible, label_delta, torch.zeros_like(label_delta))
    selected_prediction_delta = torch.where(
        eligible, prediction_delta, torch.zeros_like(prediction_delta)
    )
    signed_margin = selected_label_delta.sign() * selected_prediction_delta
    pair_loss = F.softplus(-signed_margin / PAIRWISE_TEMPERATURE)
    loss_numerator = (pair_loss * pair_weight).sum(dim=(0, 1))
    weight_denominator = pair_weight.sum(dim=(0, 1))
    eligible_count = eligible.sum(dim=(0, 1), dtype=torch.int64)
    concordant = eligible & (signed_margin > 0)
    tied = eligible & (signed_margin == 0)
    concordant_count = concordant.sum(dim=(0, 1), dtype=torch.int64)
    tie_count = tied.sum(dim=(0, 1), dtype=torch.int64)
    concordance_credit = concordant.to(pair_weight.dtype) + 0.5 * tied.to(pair_weight.dtype)
    concordance_mass = (concordance_credit * pair_weight).sum(dim=(0, 1))
    return PairwiseObjectiveSums(
        loss_numerator=loss_numerator,
        weight_denominator=weight_denominator,
        eligible_count=eligible_count,
        concordant_count=concordant_count,
        tie_count=tie_count,
        concordance_mass=concordance_mass,
    )


def pairwise_scaled_loss(
    sums: PairwiseObjectiveSums, global_pair_weight: Tensor, world_size: int
) -> Tensor:
    """Normalize each target by its global window pair weight and apply 0.05 mean.

    A target with zero eligible-pair weight contributes zero; the final mean is
    always across the fixed two targets. The caller supplies an all-rank sum.
    """
    if global_pair_weight.shape != (TARGET_COUNT,) or world_size < 1:
        raise ValueError("card06 requires two global denominators and positive world size")
    denominator = global_pair_weight.to(
        device=sums.loss_numerator.device, dtype=sums.loss_numerator.dtype
    )
    active = denominator > 0
    safe_denominator = torch.where(active, denominator, torch.ones_like(denominator))
    normalized = torch.where(
        active,
        sums.loss_numerator * (world_size / safe_denominator),
        torch.zeros_like(sums.loss_numerator),
    )
    return PAIRWISE_COEFFICIENT * normalized.mean()
