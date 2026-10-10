"""Official Global WP and the legacy block WP diagnostic.

Float32 inputs are clipped to [-2, 2], weighted by absolute clipped targets,
and accumulated in float64. ``score`` pools selected rows across all sequences.
"""

from __future__ import annotations

import numpy as np

from wnn.gmetric import METRIC_ID, global_wp

METRIC_CLIP = 2.0
WARMUP = 99
_EPS = 1e-8


def weighted_pearson_rows(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Return centered WP per masked row; degenerate rows contribute zero."""
    y = np.asarray(target, dtype=np.float32)
    p = np.asarray(prediction, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if y.shape != p.shape or y.shape != m.shape or y.ndim != 2:
        raise ValueError("expected equal (B, T) target, prediction and mask")
    if not np.isfinite(y[m]).all() or not np.isfinite(p[m]).all():
        raise ValueError("nonfinite selected target or prediction")
    y = np.where(m, np.clip(y, -METRIC_CLIP, METRIC_CLIP), 0.0).astype(np.float64)
    p = np.where(m, np.clip(p, -METRIC_CLIP, METRIC_CLIP), 0.0).astype(np.float64)
    w = np.abs(y)
    total = w.sum(axis=1)
    safe = np.where(total < _EPS, 1.0, total)
    if y.shape[1] == 0:
        return np.zeros(y.shape[0], dtype=np.float64)
    first = (w > 0).argmax(axis=1)
    rows = np.arange(len(y))
    ys = y - y[rows, first, None]
    ps = p - p[rows, first, None]
    yc = ys - ((w * ys).sum(axis=1) / safe)[:, None]
    pc = ps - ((w * ps).sum(axis=1) / safe)[:, None]
    cov = (w * yc * pc).sum(axis=1) / safe
    sy = np.sqrt((w * yc * yc).sum(axis=1) / safe)
    sp = np.sqrt((w * pc * pc).sum(axis=1) / safe)
    ok = (total >= _EPS) & (sy > _EPS) & (sp > _EPS)
    return np.clip(np.where(ok, cov / np.where(ok, sy * sp, 1.0), 0.0), -1.0, 1.0)


def _validate_blocks(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if y.ndim != 3 or y.shape[2] != 2 or p.shape != y.shape or m.shape != y.shape[:2]:
        raise ValueError("expected (B, T, 2) targets/predictions and (B, T) mask")
    if m[:, :WARMUP].any():
        raise ValueError("warm-up rows cannot be scored")
    return y, p, m


def eligible_blocks(targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mark blocks with both signs in both selected targets for legacy diagnostics."""
    y = np.asarray(targets, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if y.ndim != 3 or y.shape[-1] != 2 or m.shape != y.shape[:2]:
        raise ValueError("expected (B, T, 2) targets and (B, T) mask")
    if not np.isfinite(y[m]).all():
        raise ValueError("nonfinite selected target")
    selected = m[..., None]
    return (((y > 0) & selected).any(axis=1) & ((y < 0) & selected).any(axis=1)).all(axis=1)


def block_scores(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return diagnostic WP of shape (B, 2), with NaN for ineligible blocks."""
    y, p, m = _validate_blocks(targets, predictions, mask)
    out = np.stack([weighted_pearson_rows(y[..., k], p[..., k], m) for k in range(2)], axis=1)
    out[~eligible_blocks(y, m)] = np.nan
    return out


def block_aggregate_score(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | str]:
    """Return the legacy equal mean of eligible block scores as a diagnostic."""
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
        "metric_id": "v2_wp_block_clip2_v1_diagnostic",
    }


def score(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | str]:
    """Return official WP pooled across all selected rows, then averaged by target.

    The supplied mask must already combine ``need_prediction AND is_scored``.
    Block eligibility is reported for compatibility and never filters this score.
    """
    y, p, m = _validate_blocks(targets, predictions, mask)
    result = global_wp(y, p, m)
    return {
        "t0": result["t0"],
        "t1": result["t1"],
        "weighted_pearson": result["wp"],
        "blocks": int(y.shape[0]),
        "eligible_blocks": int(eligible_blocks(y, m).sum()),
        "selected_rows": int(m.sum()),
        "metric_id": METRIC_ID,
    }
