"""Training-only causal multiscale residual penalties."""

from __future__ import annotations

import torch


def filtered_residual_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Penalize first differences and 16-row mean residual within eligible chunks.

    Every filter window is restricted to a single sequence and chunk. A window
    contributes only when every one of its labels has positive training mass and
    finite values. The filter contributes no serving-time operations.
    """
    target = target.float()
    prediction = prediction.float()
    eligible = (valid > 0) & torch.isfinite(target).all(-1)
    target = target.clamp(-2.0, 2.0)
    residual = torch.where(eligible[..., None], prediction - target, 0.0)
    total = prediction.sum() * 0.0
    for width, kind in ((2, "difference"), (16, "mean")):
        if prediction.shape[1] < width:
            continue
        windows = residual.unfold(1, width, 1)
        mask = eligible.unfold(1, width, 1).all(-1)
        filtered = windows[..., -1] - windows[..., 0] if kind == "difference" else windows.mean(-1)
        weight = mask[..., None].to(filtered.dtype)
        total = total + (filtered.square() * weight).sum() / (weight.sum() * 2).clamp_min(1.0)
    return total * 0.5
