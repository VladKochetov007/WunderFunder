"""Vectorised replica of the competition metric (see wunder_docs/09_metric.md).

utils.BlockAccumulator is the reference; it scores one sequence at a time
through Python. This module computes the same numbers for a whole batch of
sequences at once so validation inside the training loop is cheap. Both are
cross-checked in test_contract.py.
"""

from __future__ import annotations

import numpy as np

CLIP = 2.0


def weighted_pearson_batch(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray):
    """Per-sequence, per-target clipped weighted Pearson.

    targets, predictions: (B, T, 2); mask: (B, T) bool of scored rows.
    Returns corr (B, 2) and eligible (B,) bool. Ineligible sequences (a target
    without both a positive and a negative scored value) get corr 0 and must be
    excluded from the mean by the caller.
    """
    y = np.clip(targets.astype(np.float64), -CLIP, CLIP)
    p = np.clip(predictions.astype(np.float64), -CLIP, CLIP)
    m = mask.astype(np.float64)[..., None]  # (B, T, 1)
    w = np.abs(y) * m
    total = w.sum(axis=1)  # (B, 2)
    safe_total = np.where(total < 1e-8, 1.0, total)
    mean_y = (w * y).sum(axis=1) / safe_total
    mean_p = (w * p).sum(axis=1) / safe_total
    yc = (y - mean_y[:, None, :]) * m
    pc = (p - mean_p[:, None, :]) * m
    cov = (w * yc * pc).sum(axis=1) / safe_total
    sy = np.sqrt((w * yc * yc).sum(axis=1) / safe_total)
    sp = np.sqrt((w * pc * pc).sum(axis=1) / safe_total)
    denom = sy * sp
    corr = np.where(
        (total < 1e-8) | (sy <= 1e-8) | (sp <= 1e-8), 0.0, cov / np.where(denom == 0, 1.0, denom)
    )
    corr = np.clip(corr, -1.0, 1.0)

    pos = ((y > 0) & (m > 0)).any(axis=1)  # (B, 2)
    neg = ((y < 0) & (m > 0)).any(axis=1)
    eligible = (pos & neg).all(axis=1)
    return corr, eligible


def global_weighted_pearson(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict:
    """The 2026-09-15 metric: ONE weighted Pearson per target, pooled over every
    scored row of every sequence, then averaged over the two targets.

    Differs from score_batch (the previous metric) only in aggregation: that one
    scored each sequence separately and took an equal-weight mean over
    sequences, which let a per-sequence level or gain offset cancel for free.
    Pooled globally there is a single shared mean, so such an offset is noise.

    targets, predictions: (B, T, 2); mask: (B, T) bool of scored rows.
    """
    y = np.clip(targets.astype(np.float64), -CLIP, CLIP).reshape(-1, targets.shape[-1])
    p = np.clip(predictions.astype(np.float64), -CLIP, CLIP).reshape(-1, predictions.shape[-1])
    m = mask.reshape(-1).astype(bool)
    y, p = y[m], p[m]
    w = np.abs(y)  # (N, 2)
    total = w.sum(axis=0)  # (2,)
    safe = np.where(total < 1e-8, 1.0, total)
    mean_y = (w * y).sum(axis=0) / safe
    mean_p = (w * p).sum(axis=0) / safe
    yc, pc = y - mean_y, p - mean_p
    cov = (w * yc * pc).sum(axis=0) / safe
    sy = np.sqrt((w * yc * yc).sum(axis=0) / safe)
    sp = np.sqrt((w * pc * pc).sum(axis=0) / safe)
    denom = sy * sp
    corr = np.where(
        (total < 1e-8) | (sy <= 1e-8) | (sp <= 1e-8), 0.0, cov / np.where(denom == 0, 1.0, denom)
    )
    corr = np.clip(corr, -1.0, 1.0)
    return {
        "t0": float(corr[0]),
        "t1": float(corr[1]),
        "weighted_pearson": float(corr.mean()),
        "rows": int(m.sum()),
    }


def score_batch(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict:
    corr, eligible = weighted_pearson_batch(targets, predictions, mask)
    if not eligible.any():
        raise ValueError("no eligible sequences")
    per_target = corr[eligible].mean(axis=0)
    return {
        "t0": float(per_target[0]),
        "t1": float(per_target[1]),
        "weighted_pearson": float(per_target.mean()),
        "blocks": int(len(eligible)),
        "eligible_blocks": int(eligible.sum()),
    }


class ScoreAccumulator:
    """Accumulates per-sequence scores across batches, mirroring BlockAccumulator."""

    def __init__(self):
        self.scores = []
        self.blocks = 0

    def add(self, targets, predictions, mask):
        corr, eligible = weighted_pearson_batch(targets, predictions, mask)
        self.blocks += len(eligible)
        self.scores.extend(corr[eligible].tolist())

    def result(self) -> dict:
        if not self.scores:
            raise ValueError("no eligible sequences")
        per_target = np.mean(np.asarray(self.scores, dtype=np.float64), axis=0)
        return {
            "t0": float(per_target[0]),
            "t1": float(per_target[1]),
            "weighted_pearson": float(per_target.mean()),
            "blocks": self.blocks,
            "eligible_blocks": len(self.scores),
        }
