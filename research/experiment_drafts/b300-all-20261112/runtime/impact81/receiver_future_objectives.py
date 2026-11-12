"""Receiver-only semantic probes for card07 horizon labels and normalized loss."""

from __future__ import annotations

import torch
from torch import Tensor

from impact81.future_objective import (
    FutureObjectiveSums,
    future_objective_sums,
    future_scaled_loss,
)

OFFSETS = (1, 16, 64)


def check_future_objective_helpers(
    device: torch.device,
) -> dict[str, float | int | bool | list[int]]:
    """Check offsets, same-sequence bounds, masks, chunk additivity, and DDP math."""
    target = torch.zeros((2, 80, 2), dtype=torch.float32, device=device)
    rows = torch.arange(80, dtype=torch.float32, device=device)
    target[:, :, 0] = 0.1 + rows[None, :] * 0.001
    target[:, :, 1] = 0.2 + rows[None, :] * 0.001
    target[0, 26, 1] = torch.nan
    target[1, 76, 0] = torch.inf
    mask = torch.ones((2, 80), dtype=torch.bool, device=device)
    mask[1, 10] = False
    predictions = torch.zeros((2, 10, 3, 2), dtype=torch.float32, device=device)
    aligned = predictions.clone()
    for horizon_index, offset in enumerate(OFFSETS):
        for position, source_row in enumerate(range(10, 20)):
            if source_row + offset < target.shape[1]:
                aligned[:, position, horizon_index] = target[:, source_row + offset]
    aligned_sums = future_objective_sums(aligned, target, mask, 10, 20)
    if not bool(torch.isfinite(aligned_sums.loss_numerator).all()):
        raise RuntimeError("masked nonfinite future labels contaminated loss sums")
    if not torch.allclose(
        aligned_sums.loss_numerator,
        torch.zeros_like(aligned_sums.loss_numerator),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("future targets are not aligned to source row plus configured offset")
    expected_counts = torch.tensor([[19, 19], [19, 18], [10, 11]], dtype=torch.int64, device=device)
    if not torch.equal(aligned_sums.eligible_count, expected_counts):
        raise RuntimeError("future label masks or horizon offsets produced wrong counts")

    full = future_objective_sums(predictions, target, mask, 10, 20)
    left = future_objective_sums(predictions[:, :5], target, mask, 10, 15)
    right = future_objective_sums(predictions[:, 5:], target, mask, 15, 20)
    for field in ("loss_numerator", "weight_mass"):
        combined = getattr(left, field) + getattr(right, field)
        if not torch.allclose(combined, getattr(full, field), atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"future {field} does not add across source chunks")
    if not torch.equal(left.eligible_count + right.eligible_count, full.eligible_count):
        raise RuntimeError("future eligible counts do not add across source chunks")
    if not bool(left.weight_mass[1].sum() > 0):
        raise RuntimeError("future label lookup incorrectly stopped at the source chunk boundary")

    end_sums = future_objective_sums(
        torch.zeros((2, 10, 3, 2), dtype=torch.float32, device=device), target, mask, 70, 80
    )
    # The injected +inf at target[1, 76, 0] is in the one-step endpoint window
    # and is correctly excluded by the production finite-label mask.
    expected_end = torch.tensor([[17, 18], [0, 0], [0, 0]], dtype=torch.int64, device=device)
    if not torch.equal(end_sums.eligible_count, expected_end):
        raise RuntimeError("future targets crossed the end of a fitting sequence")

    rank_zero, rank_one = _rank_window_sums(target, mask, predictions)
    global_mass = (rank_zero.weight_mass + rank_one.weight_mass).sum(dim=1)
    loss_zero = future_scaled_loss(rank_zero, global_mass, 2)
    loss_one = future_scaled_loss(rank_one, global_mass, 2)
    ddp_mean = (loss_zero + loss_one) / 2
    pooled_numerator = (rank_zero.loss_numerator + rank_one.loss_numerator).sum(dim=1)
    expected_global = 0.1 * (pooled_numerator / global_mass).mean()
    if not torch.allclose(ddp_mean, expected_global, atol=1e-7, rtol=1e-6):
        raise RuntimeError("future world-size scaling does not reconstruct pooled global loss")
    return {
        "future_offsets": list(OFFSETS),
        "target_alignment_zero_loss": True,
        "masked_nonfinite_labels_excluded": True,
        "eligible_counts_checked": int(expected_counts.sum().item()),
        "cross_chunk_label_lookahead": True,
        "sequence_end_bounds": True,
        "microchunk_additivity": True,
        "ddp_world_size_scaling": True,
        "ddp_expected_loss": float(expected_global.item()),
    }


def _rank_window_sums(
    target: Tensor, mask: Tensor, predictions: Tensor
) -> tuple[FutureObjectiveSums, FutureObjectiveSums]:
    """Aggregate two source chunks for each synthetic rank."""
    rank_results = []
    for rank in range(2):
        rank_target = target[rank : rank + 1]
        rank_mask = mask[rank : rank + 1]
        first = future_objective_sums(
            predictions[rank : rank + 1, :5], rank_target, rank_mask, 10, 15
        )
        second = future_objective_sums(
            predictions[rank : rank + 1, 5:], rank_target, rank_mask, 15, 20
        )
        rank_results.append(
            FutureObjectiveSums(
                first.loss_numerator + second.loss_numerator,
                first.weight_mass + second.weight_mass,
                first.eligible_count + second.eligible_count,
            )
        )
    return rank_results[0], rank_results[1]
