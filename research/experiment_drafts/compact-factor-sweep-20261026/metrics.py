"""Bounded streaming metrics for fixed-checkpoint validation probes."""

from __future__ import annotations

import math

import numpy as np

from wnn.gmetric import METRIC_CLIP, GlobalAccumulator

WMSE_METRIC_ID = "raw_prediction_weighted_mse_clip2_v1"


class StreamingMetrics:
    """Accumulate official pooled WP and weighted MSE without retaining batches.

    ``scored_mask`` must already exclude rows 0 through 98. ``wmse_mask`` may
    additionally include unscored training rows, but must also exclude those
    warmup rows. When omitted, WMSE uses the scored mask.
    """

    def __init__(self) -> None:
        self._wp = GlobalAccumulator()
        self._wmse_numerator = 0.0
        self._wmse_weight = 0.0
        self._wmse_rows = 0

    def update(
        self,
        y: np.ndarray,
        p: np.ndarray,
        scored_mask: np.ndarray,
        *,
        wmse_mask: np.ndarray | None = None,
    ) -> None:
        """Consume one aligned batch of targets, raw predictions and masks.

        Shapes are ``(..., rows, 2)`` for ``y`` and ``p``, and
        ``(..., rows)`` for the masks. WP uses the canonical float32 conversion,
        clipping and centered pooled moments in :mod:`wnn.gmetric`. WMSE clips
        only targets, weights by absolute clipped targets and keeps predictions
        raw, matching the canonical training loss.
        """
        targets = np.asarray(y, dtype=np.float32)
        predictions = np.asarray(p, dtype=np.float32)
        selected = np.asarray(scored_mask, dtype=bool)
        mse_selected = selected if wmse_mask is None else np.asarray(wmse_mask, dtype=bool)
        if (
            targets.ndim < 2
            or targets.shape[-1] != 2
            or predictions.shape != targets.shape
            or selected.shape != targets.shape[:-1]
            or mse_selected.shape != targets.shape[:-1]
        ):
            raise ValueError("expected aligned (..., rows, 2) values and (..., rows) masks")

        # Validate WMSE's potentially broader selection before mutating either
        # accumulator, so a rejected batch cannot leave partial statistics.
        if mse_selected.any() and (
            not np.isfinite(targets[mse_selected]).all()
            or not np.isfinite(predictions[mse_selected]).all()
        ):
            raise ValueError("nonfinite WMSE-selected target or prediction")
        # GlobalAccumulator checks selected WP values before clipping.
        self._wp.add(targets, predictions, selected)
        if not mse_selected.any():
            return

        clipped_target = np.clip(targets[mse_selected], -METRIC_CLIP, METRIC_CLIP).astype(
            np.float64
        )
        raw_prediction = predictions[mse_selected].astype(np.float64)
        weight = np.abs(clipped_target)
        numerator = float(
            np.sum(weight * np.square(raw_prediction - clipped_target), dtype=np.float64)
        )
        total_weight = float(np.sum(weight, dtype=np.float64))
        if not (math.isfinite(numerator) and math.isfinite(total_weight)):
            raise ValueError("nonfinite batch WMSE accumulator")
        self._wmse_numerator += numerator
        self._wmse_weight += total_weight
        self._wmse_rows += int(mse_selected.sum())

    def result(self) -> dict[str, float | int | str]:
        """Return the official WP score and raw-prediction weighted MSE."""
        wp = self._wp.result()
        if not (
            math.isfinite(self._wmse_numerator)
            and math.isfinite(self._wmse_weight)
            and self._wmse_numerator >= 0
            and self._wmse_weight >= 0
        ):
            raise ValueError("nonfinite streaming WMSE accumulator")
        wmse = self._wmse_numerator / self._wmse_weight if self._wmse_weight > 0 else 0.0
        return {
            "wp": float(wp["wp"]),
            "t0": float(wp["t0"]),
            "t1": float(wp["t1"]),
            "selected_rows": int(wp["selected_rows"]),
            "metric_id": str(wp["metric_id"]),
            "wmse": wmse,
            "wmse_rows": self._wmse_rows,
            "wmse_weight": self._wmse_weight,
            "wmse_metric_id": WMSE_METRIC_ID,
        }
