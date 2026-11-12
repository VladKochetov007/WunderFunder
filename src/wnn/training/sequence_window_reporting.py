from __future__ import annotations

import torch


def window_record(
    stats: torch.Tensor,
    denominators: torch.Tensor,
    rank: int,
    epoch: int,
    step: int,
    learning_rate: float,
    gradient_norm: torch.Tensor,
    elapsed: float,
    peak_memory: int,
    gradient_clip: float,
    kd_weight: float,
    hint_weight: float,
) -> dict[str, object]:
    """Convert global sufficient sums and this rank's diagnostics into a JSON row."""
    label_denom = float(denominators[0].item())
    hint_denom = float(denominators[1].item())
    label_loss = float(stats[0].item()) / label_denom if label_denom > 0 else 0.0
    kd_loss = float(stats[1].item()) / label_denom if label_denom > 0 else 0.0
    hint_loss = float(stats[2].item()) / hint_denom if hint_denom > 0 else 0.0
    return {
        "rank": rank,
        "epoch": epoch,
        "step": step,
        "label_loss": label_loss,
        "kd_loss": kd_loss,
        "hint_loss": hint_loss,
        "total_loss": label_loss + kd_weight * kd_loss + hint_weight * hint_loss,
        "valid_rows": int(stats[3].item()),
        "forwarded_rows": int(stats[4].item()),
        "real_rows": int(stats[5].item()),
        "label_weight": label_denom,
        "hint_elements": hint_denom,
        "learning_rate": learning_rate,
        "gradient_norm_preclip": float(gradient_norm.item()),
        "gradient_clipped": int(float(gradient_norm.item()) > gradient_clip),
        "window_seconds": elapsed,
        "valid_rows_per_second": float(stats[3].item()) / max(elapsed, 1e-12),
        "forwarded_rows_per_second": float(stats[4].item()) / max(elapsed, 1e-12),
        "peak_memory_bytes": peak_memory,
    }
