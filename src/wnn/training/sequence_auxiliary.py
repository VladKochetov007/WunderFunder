from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from wnn.outcomes import N_SP_AUX, sequence_class, structure_labels
from wnn.training.targets import next_change_labels

STRUCTURE_WIDTH = N_SP_AUX + 3 + 4
NEXT_CHANGE_WIDTH = 2
SUPPORTED_AUXILIARY_HEADS = ("structure", "next_change")


@dataclass(frozen=True)
class SequenceAuxiliaryTargets:
    """Training-only labels and masks for the declared auxiliary objectives."""

    heads: tuple[str, ...]
    structure: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    structure_mask: torch.Tensor | None
    next_change: tuple[torch.Tensor, torch.Tensor] | None
    next_mask: torch.Tensor | None


def validate_auxiliary_heads(value: object) -> tuple[str, ...]:
    """Validate a canonical ordered subset of the supported row-level heads."""
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise TypeError("model.auxiliary_heads must be a list of head names")
    heads = tuple(value)
    if heads not in ((), ("structure",), ("next_change",), SUPPORTED_AUXILIARY_HEADS):
        raise ValueError("model.auxiliary_heads must be an ordered subset of supported heads")
    return heads


def auxiliary_output_width(value: object) -> int:
    """Return the total recurrent output width, including the canonical target pair."""
    heads = validate_auxiliary_heads(value)
    return 2 + sum(STRUCTURE_WIDTH if head == "structure" else NEXT_CHANGE_WIDTH for head in heads)


def build_auxiliary_targets(
    target: torch.Tensor, mask: torch.Tensor, value: object
) -> SequenceAuxiliaryTargets | None:
    """Build auxiliary labels from one fitting batch without changing its membership."""
    heads = validate_auxiliary_heads(value)
    if not heads:
        return None
    structure = None
    structure_mask = None
    next_change = None
    next_mask = None
    if "structure" in heads:
        classes = sequence_class(target)
        structure = structure_labels(target, classes)
        structure_mask = mask
    if "next_change" in heads:
        rows, direction, next_mask = next_change_labels(target, mask)
        next_change = (rows, direction)
    return SequenceAuxiliaryTargets(heads, structure, structure_mask, next_change, next_mask)


def auxiliary_denominators(
    targets: SequenceAuxiliaryTargets | None,
    intervals: Sequence[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """Sum original row-mask mass for each auxiliary loss across one update window."""
    result = torch.zeros(2, dtype=torch.float32, device=device)
    if targets is None:
        return result
    for start, stop in intervals:
        if targets.structure_mask is not None:
            result[0] += targets.structure_mask[:, start:stop].sum(dtype=torch.float32)
        if targets.next_mask is not None:
            result[1] += targets.next_mask[:, start:stop].sum(dtype=torch.float32)
    return result


def auxiliary_loss_sums(
    prediction: torch.Tensor,
    targets: SequenceAuxiliaryTargets | None,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Return unnormalized structural and next-change loss sums for a chunk."""
    result = prediction.new_zeros(2, dtype=torch.float32)
    if targets is None:
        return result
    offset = 2
    if targets.structure is not None:
        spread, branch, classes = targets.structure
        spread = spread[:, start:stop]
        branch = branch[:, start:stop]
        classes = classes[:, start:stop]
        row_mask = targets.structure_mask[:, start:stop].float()
        logits = prediction[..., offset : offset + STRUCTURE_WIDTH].float()
        spread_ce = F.cross_entropy(
            logits[..., :N_SP_AUX].reshape(-1, N_SP_AUX), spread.reshape(-1), reduction="none"
        ).reshape_as(row_mask)
        branch_start = N_SP_AUX
        branch_stop = branch_start + 3
        branch_ce = F.cross_entropy(
            logits[..., branch_start:branch_stop].reshape(-1, 3),
            branch.reshape(-1),
            reduction="none",
        ).reshape_as(row_mask)
        class_ce = F.cross_entropy(
            logits[..., branch_stop:].reshape(-1, 4), classes.reshape(-1), reduction="none"
        ).reshape_as(row_mask)
        result[0] = ((spread_ce + branch_ce + class_ce) * row_mask).sum()
        offset += STRUCTURE_WIDTH
    if targets.next_change is not None:
        rows, direction = targets.next_change
        rows = rows[:, start:stop]
        direction = direction[:, start:stop]
        row_mask = targets.next_mask[:, start:stop].float()
        logits = prediction[..., offset : offset + NEXT_CHANGE_WIDTH].float()
        regression_target = (rows - 3.0) / 2.0
        regression = (logits[..., 0] - regression_target).square()
        classification = F.binary_cross_entropy_with_logits(
            logits[..., 1], direction, reduction="none"
        )
        result[1] = ((regression + classification) * row_mask).sum()
    return result
