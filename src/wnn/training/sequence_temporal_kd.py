from __future__ import annotations

import torch


def sequence_chunk_intervals(row_count: int, chunk_rows: int) -> list[tuple[int, int]]:
    """Cover a sequence in nonoverlapping chunks without crossing any boundary."""
    return [
        (start, min(start + chunk_rows, row_count)) for start in range(0, row_count, chunk_rows)
    ]


def temporal_pair_mask(mask: torch.Tensor, lag: int) -> torch.Tensor:
    """Select scored row pairs within a single microchunk; never cross its boundary."""
    if mask.shape[1] <= lag:
        return mask[:, :0].to(dtype=torch.float32)
    return (mask[:, lag:].bool() & mask[:, :-lag].bool()).float()


def temporal_mass(mask: torch.Tensor, lag: int) -> torch.Tensor:
    """Count eligible prediction elements for this chunk before distributed backward."""
    return temporal_pair_mask(mask, lag).sum() * 2


def temporal_numerator(
    prediction: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, lag: int
) -> torch.Tensor:
    """Compare clipped teacher differences and student differences without future rows."""
    student = prediction.float()
    target = teacher.float().clamp(-2.0, 2.0)
    pair_mask = temporal_pair_mask(mask, lag)
    error = (student[:, lag:] - student[:, :-lag]) - (target[:, lag:] - target[:, :-lag])
    return (error.square() * pair_mask[..., None]).sum()


def record_temporal(
    record: dict[str, object],
    stats: torch.Tensor,
    denominators: torch.Tensor,
    lag: int,
    weight: float,
) -> None:
    """Add globally reduced temporal loss and coverage to a window record."""
    mass = float(denominators[2].item())
    temporal_loss = float(stats[6].item()) / mass if mass > 0 else 0.0
    record["temporal_kd_loss"] = temporal_loss
    record["temporal_pair_elements"] = int(stats[7].item())
    record["temporal_kd_lag"] = lag
    record["total_loss"] = float(record["total_loss"]) + weight * temporal_loss
