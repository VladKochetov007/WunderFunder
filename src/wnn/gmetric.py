"""Global (pooled) clipped weighted Pearson, the competition metric as of 2026-09.

WP is computed per target over all rows selected by ``need_prediction AND
is_scored`` pooled across sequences (no per-sequence averaging, no block
eligibility), then the two targets are averaged. Targets and predictions are
clipped to [-2, 2]; weights are ``|clipped target|``.

Everything is expressed through six mergeable sums per target, so a score can be
accumulated in a stream and decomposed by any row grouping (class, sequence,
event distance) without holding predictions for the whole set.
"""

from __future__ import annotations

import numpy as np

METRIC_CLIP = 2.0
WARMUP = 99
_EPS = 1e-8

# Sum layout along the last axis.
W, WY, WP, WYY, WPP, WYP = range(6)


def wp_sums(target, prediction, mask=None) -> np.ndarray:
    """Sufficient statistics of shape (..., 6) for one target over the last axis."""
    y = np.clip(np.asarray(target, dtype=np.float32), -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    p = np.clip(np.asarray(prediction, dtype=np.float32), -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    w = np.abs(y)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        w = np.where(m, w, 0.0)
        p = np.where(m, p, 0.0)
        y = np.where(m, y, 0.0)
    if not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite selected target or prediction")
    wy, wp = w * y, w * p
    return np.stack([w.sum(-1), wy.sum(-1), wp.sum(-1), (wy * y).sum(-1), (wp * p).sum(-1), (wy * p).sum(-1)], -1)


def wp_from_sums(s) -> np.ndarray:
    """Weighted Pearson from sums (..., 6); degenerate sides give 0 like the reference."""
    s = np.asarray(s, dtype=np.float64)
    tot = s[..., W]
    safe = np.where(tot < _EPS, 1.0, tot)
    my, mp = s[..., WY] / safe, s[..., WP] / safe
    cov = s[..., WYP] / safe - my * mp
    vy = np.maximum(s[..., WYY] / safe - my * my, 0.0)
    vp = np.maximum(s[..., WPP] / safe - mp * mp, 0.0)
    sy, sp = np.sqrt(vy), np.sqrt(vp)
    ok = (tot >= _EPS) & (sy > _EPS) & (sp > _EPS)
    return np.clip(np.where(ok, cov / np.where(ok, sy * sp, 1.0), 0.0), -1.0, 1.0)


def global_wp(targets, predictions, mask) -> dict:
    """Score arrays shaped (..., T, 2) with mask (..., T); returns per-target and mean WP."""
    y = np.asarray(targets)
    p = np.asarray(predictions)
    m = np.asarray(mask, dtype=bool)
    out = {}
    for k in range(2):
        s = wp_sums(y[..., k].reshape(-1), p[..., k].reshape(-1), m.reshape(-1))
        out[f"t{k}"] = float(wp_from_sums(s))
    out["wp"] = 0.5 * (out["t0"] + out["t1"])
    return out


class GlobalAccumulator:
    """Streaming global WP with optional group labels for contribution analysis."""

    def __init__(self):
        self.sums = np.zeros((2, 6))
        self.groups: dict = {}

    def add(self, targets, predictions, mask, group=None):
        s = np.stack([wp_sums(targets[..., k], predictions[..., k], mask).reshape(-1, 6).sum(0) for k in range(2)])
        self.sums += s
        if group is not None:
            self.groups[group] = self.groups.get(group, 0.0) + s
        return s

    def result(self) -> dict:
        r = wp_from_sums(self.sums)
        return {"t0": float(r[0]), "t1": float(r[1]), "wp": float(r.mean())}


def reference_global_wp(targets, predictions, mask) -> float:
    """Literal transcription of the published formula, for testing the fast path."""
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    rhos = []
    for k in range(2):
        yk = np.clip(y[..., k][m], -2, 2).astype(np.float64)
        pk = np.clip(p[..., k][m], -2, 2).astype(np.float64)
        w = np.abs(yk)
        tot = w.sum()
        my, mp = (w * yk).sum() / tot, (w * pk).sum() / tot
        cov = (w * (yk - my) * (pk - mp)).sum() / tot
        sy = np.sqrt((w * (yk - my) ** 2).sum() / tot)
        sp = np.sqrt((w * (pk - mp) ** 2).sum() / tot)
        rhos.append(0.0 if sy <= _EPS or sp <= _EPS else float(np.clip(cov / (sy * sp), -1, 1)))
    return float(np.mean(rhos))
