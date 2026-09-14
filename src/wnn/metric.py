"""Vectorized clipped weighted Pearson, spec ``v2_wp_block_clip2_v1`` (see METRIC.md).

Arrays are batched over sequences: shape ``(B, T)`` per target or ``(B, T, 2)``.
Numerics follow the reference: float32 inputs, clip, then float64 accumulation.
"""

from __future__ import annotations

import numpy as np

METRIC_CLIP = 2.0
WARMUP = 99
_EPS = 1e-8


def weighted_pearson_rows(target, prediction, mask) -> np.ndarray:
    """Weighted Pearson per row over masked entries; degenerate rows score 0.0."""
    y = np.clip(np.asarray(target, dtype=np.float32), -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    p = np.asarray(prediction, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if y.shape != p.shape or y.shape != m.shape or y.ndim != 2:
        raise ValueError("expected equal (B, T) target, prediction and mask")
    # Unselected predictions (e.g. NaN warm-up) must not leak into sums.
    p = np.where(m, np.clip(p, -METRIC_CLIP, METRIC_CLIP), 0.0).astype(np.float64)
    if not np.isfinite(p).all() or not np.isfinite(np.where(m, y, 0.0)).all():
        raise ValueError("nonfinite selected target or prediction")
    w = np.where(m, np.abs(y), 0.0)
    total = w.sum(axis=1)
    safe = np.where(total < _EPS, 1.0, total)
    mean_y = (w * y).sum(axis=1) / safe
    mean_p = (w * p).sum(axis=1) / safe
    yc = y - mean_y[:, None]
    pc = p - mean_p[:, None]
    cov = (w * yc * pc).sum(axis=1) / safe
    sy = np.sqrt((w * yc * yc).sum(axis=1) / safe)
    sp = np.sqrt((w * pc * pc).sum(axis=1) / safe)
    ok = (total >= _EPS) & (sy > _EPS) & (sp > _EPS)
    rho = np.where(ok, cov / np.where(ok, sy * sp, 1.0), 0.0)
    return np.clip(rho, -1.0, 1.0)


def eligible_blocks(targets, mask) -> np.ndarray:
    """A block is eligible when each target has a positive and a negative selected value."""
    y = np.asarray(targets, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)[..., None]
    return (((y > 0) & m).any(axis=1) & ((y < 0) & m).any(axis=1)).all(axis=1)


def block_scores(targets, predictions, mask) -> np.ndarray:
    """Per-block, per-target WP of shape (B, 2); NaN rows mark ineligible blocks."""
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if y.ndim != 3 or y.shape[2] != 2 or p.shape != y.shape or m.shape != y.shape[:2]:
        raise ValueError("expected (B, T, 2) targets/predictions and (B, T) mask")
    if m[:, :WARMUP].any():
        raise ValueError("warm-up rows cannot be scored")
    out = np.stack([weighted_pearson_rows(y[..., k], p[..., k], m) for k in range(2)], axis=1)
    out[~eligible_blocks(y, m)] = np.nan
    return out


def score(targets, predictions, mask) -> dict:
    """Competition aggregate: equal mean over eligible blocks of the two-target mean."""
    per_block = block_scores(targets, predictions, mask)
    eligible = ~np.isnan(per_block[:, 0])
    if not eligible.any():
        raise ValueError("no eligible blocks")
    per_side = per_block[eligible].mean(axis=0)
    return {
        "t0": float(per_side[0]),
        "t1": float(per_side[1]),
        "weighted_pearson": float(per_side.mean()),
        "blocks": int(per_block.shape[0]),
        "eligible_blocks": int(eligible.sum()),
    }
