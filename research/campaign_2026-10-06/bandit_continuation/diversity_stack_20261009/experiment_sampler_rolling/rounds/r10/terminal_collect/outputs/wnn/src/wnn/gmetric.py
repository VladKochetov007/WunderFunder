"""Official pooled clipped weighted Pearson with stable streaming accumulation.

Six additive raw sums remain available for contribution analyses. Official scores
use centered moments because raw second moments lose precision for nearly
constant float32 inputs.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np

METRIC_ID = "global_wp_clip2_v1"
METRIC_CLIP = 2.0
WARMUP = 99
_EPS = 1e-8

# Public raw sum layout; centered moments use weight, means, variances, covariance.
W, WY, WP, WYY, WPP, WYP = range(6)


def _selected_pair(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(target, dtype=np.float32)
    p = np.asarray(prediction, dtype=np.float32)
    if y.shape != p.shape or y.ndim < 1:
        raise ValueError("expected equal target and prediction arrays with a row axis")
    m = np.ones(y.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if m.shape != y.shape:
        raise ValueError("mask shape must match target and prediction")
    # Validate before clipping: infinity must never become a valid boundary value.
    if not np.isfinite(y[m]).all() or not np.isfinite(p[m]).all():
        raise ValueError("nonfinite selected target or prediction")
    y = np.where(m, np.clip(y, -METRIC_CLIP, METRIC_CLIP), 0.0).astype(np.float64)
    p = np.where(m, np.clip(p, -METRIC_CLIP, METRIC_CLIP), 0.0).astype(np.float64)
    return y, p, m


def _validate_targets(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if y.ndim < 2 or y.shape[-1] != 2 or p.shape != y.shape or m.shape != y.shape[:-1]:
        raise ValueError("expected (..., T, 2) targets/predictions and (..., T) mask")
    if not np.isfinite(y[m]).all() or not np.isfinite(p[m]).all():
        raise ValueError("nonfinite selected target or prediction")
    return y, p, m


def wp_sums(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Return six float64 additive raw statistics over the last row axis.

    Unselected nonfinite values are ignored. Selected raw nonfinite values raise.
    Use ``GlobalAccumulator`` for accurate scoring of nearly constant data.
    """
    y, p, _ = _selected_pair(target, prediction, mask)
    w = np.abs(y)
    wy, wp = w * y, w * p
    return np.stack(
        [
            w.sum(-1),
            wy.sum(-1),
            wp.sum(-1),
            (wy * y).sum(-1),
            (wp * p).sum(-1),
            (wy * p).sum(-1),
        ],
        -1,
    )


def wp_from_sums(s: np.ndarray) -> np.ndarray:
    """Compute WP from raw sums, suppressing roundoff-only constant variance.

    Six raw sums cannot recover tiny variances lost during subtraction. Official
    streaming scores retain centered moments separately for that case.
    """
    s = np.asarray(s, dtype=np.float64)
    if s.ndim < 1 or s.shape[-1] != 6 or not np.isfinite(s).all():
        raise ValueError("expected finite statistics with last dimension 6")
    if (s[..., W] < 0).any():
        raise ValueError("total weight cannot be negative")
    tot = s[..., W]
    safe = np.where(tot < _EPS, 1.0, tot)
    my, mp = s[..., WY] / safe, s[..., WP] / safe
    cov = s[..., WYP] / safe - my * mp
    second_y, second_p = s[..., WYY] / safe, s[..., WPP] / safe
    vy, vp = second_y - my * my, second_p - mp * mp
    tolerance = 8 * np.finfo(np.float64).eps
    vy = np.where(vy > tolerance * np.maximum(np.abs(second_y), my * my), vy, 0.0)
    vp = np.where(vp > tolerance * np.maximum(np.abs(second_p), mp * mp), vp, 0.0)
    sy, sp = np.sqrt(vy), np.sqrt(vp)
    ok = (tot >= _EPS) & (sy > _EPS) & (sp > _EPS)
    return np.clip(np.where(ok, cov / np.where(ok, sy * sp, 1.0), 0.0), -1.0, 1.0)


def _centered_moments(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    w = np.abs(y)
    positive = w > 0
    if not positive.any():
        return np.zeros(6, dtype=np.float64)
    y, p, w = y[positive], p[positive], w[positive]
    total = w.sum()
    # Shifting before finding means also makes exact constants stay exact.
    ys, ps = y - y[0], p - p[0]
    my, mp = np.sum(w * ys) / total, np.sum(w * ps) / total
    yc, pc = ys - my, ps - mp
    return np.array(
        [
            total,
            y[0] + my,
            p[0] + mp,
            np.sum(w * yc * yc),
            np.sum(w * pc * pc),
            np.sum(w * yc * pc),
        ],
        dtype=np.float64,
    )


def _merge_moments(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left[W] == 0:
        return right.copy()
    if right[W] == 0:
        return left.copy()
    total = left[W] + right[W]
    dy, dp = right[WY] - left[WY], right[WP] - left[WP]
    ratio = right[W] / total
    cross_weight = left[W] * ratio
    return np.array(
        [
            total,
            left[WY] + dy * ratio,
            left[WP] + dp * ratio,
            left[WYY] + right[WYY] + dy * dy * cross_weight,
            left[WPP] + right[WPP] + dp * dp * cross_weight,
            left[WYP] + right[WYP] + dy * dp * cross_weight,
        ],
        dtype=np.float64,
    )


def _wp_from_moments(moments: np.ndarray) -> float:
    total = moments[W]
    if total < _EPS:
        return 0.0
    sy, sp = np.sqrt(np.maximum(moments[[WYY, WPP]], 0.0) / total)
    if sy <= _EPS or sp <= _EPS:
        return 0.0
    return float(np.clip(moments[WYP] / total / (sy * sp), -1.0, 1.0))


def global_wp(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | str]:
    """Pool every selected row, then average the two targets' weighted Pearson.

    Raises:
        ValueError: Shapes are invalid, a selected value is nonfinite, or no rows
            are selected. Zero weights and constant values produce zero per target.
    """
    accumulator = GlobalAccumulator()
    accumulator.add(targets, predictions, mask)
    return accumulator.result()


class GlobalAccumulator:
    """Stream official WP while preserving additive raw sums and optional groups."""

    def __init__(self) -> None:
        self.sums = np.zeros((2, 6), dtype=np.float64)
        self.groups: dict[Hashable, np.ndarray] = {}
        self.selected_rows = 0
        self._moments = np.zeros((2, 6), dtype=np.float64)

    def add(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        mask: np.ndarray,
        group: Hashable | None = None,
    ) -> np.ndarray:
        """Add any row partition and return its legacy six raw sums per target."""
        y, p, m = _validate_targets(targets, predictions, mask)
        sums, moments = [], []
        for k in range(2):
            yk, pk, _ = _selected_pair(y[..., k].reshape(-1), p[..., k].reshape(-1), m.reshape(-1))
            sums.append(wp_sums(yk, pk))
            moments.append(_centered_moments(yk, pk))
        added = np.stack(sums)
        self.sums += added
        for k in range(2):
            self._moments[k] = _merge_moments(self._moments[k], moments[k])
        self.selected_rows += int(m.sum())
        if group is not None:
            self.groups[group] = self.groups.get(group, np.zeros((2, 6))) + added
        return added

    def result(self) -> dict[str, float | int | str]:
        """Return a finite score; an accumulator with no selected rows raises."""
        if self.selected_rows == 0:
            raise ValueError("no selected rows")
        r = [_wp_from_moments(side) for side in self._moments]
        return {
            "t0": r[0],
            "t1": r[1],
            "wp": 0.5 * (r[0] + r[1]),
            "selected_rows": self.selected_rows,
            "metric_id": METRIC_ID,
        }


def reference_global_wp(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> float:
    """Evaluate the centered published formula independently of streaming sums."""
    y, p, m = _validate_targets(targets, predictions, mask)
    if not m.any():
        raise ValueError("no selected rows")
    rhos = []
    for k in range(2):
        yk = np.clip(y[..., k][m], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        pk = np.clip(p[..., k][m], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        w = np.abs(yk)
        total = w.sum()
        if total < _EPS:
            rhos.append(0.0)
            continue
        my, mp = np.sum(w * yk) / total, np.sum(w * pk) / total
        cov = np.sum(w * (yk - my) * (pk - mp)) / total
        sy = np.sqrt(np.sum(w * (yk - my) ** 2) / total)
        sp = np.sqrt(np.sum(w * (pk - mp) ** 2) / total)
        rhos.append(0.0 if sy <= _EPS or sp <= _EPS else float(np.clip(cov / (sy * sp), -1, 1)))
    return float(np.mean(rhos))
