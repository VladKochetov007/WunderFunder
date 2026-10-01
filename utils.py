"""Competition callback and official pooled clipped weighted Pearson scorer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEQUENCE_LENGTH = 20_000
WARMUP = 99
N_FEATURES = 112
FEATURE_COLUMNS = [
    f"i{i}_{g}{j}"
    for i in range(2)
    for g, n in [("p", 22), ("v", 22), ("dp", 4), ("dv", 4)]
    for j in range(n)
]
FEATURE_COLUMNS += [f"a{i}" for i in range(8)]
TARGET_COLUMNS = ("t0", "t1")
METRIC_CLIP = 2.0
METRIC_ID = "global_wp_clip2_v1"
_EPS = 1e-8


@dataclass(frozen=True)
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


class PredictionModel(Protocol):
    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """Update causal state and return two scores when prediction is required."""
        ...


def weighted_pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    """Evaluate the centered clipped formula for one target's selected rows."""
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("expected equal one-dimensional target and prediction arrays")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("target/prediction contains nonfinite values")
    target = np.clip(target, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    prediction = np.clip(prediction, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    weights = np.abs(target)
    total = weights.sum()
    if total < _EPS:
        return 0.0
    target_mean = np.sum(weights * target) / total
    prediction_mean = np.sum(weights * prediction) / total
    target_centered = target - target_mean
    prediction_centered = prediction - prediction_mean
    covariance = np.sum(weights * target_centered * prediction_centered) / total
    target_std = np.sqrt(np.sum(weights * target_centered**2) / total)
    prediction_std = np.sqrt(np.sum(weights * prediction_centered**2) / total)
    if target_std <= _EPS or prediction_std <= _EPS:
        return 0.0
    return float(np.clip(covariance / (target_std * prediction_std), -1.0, 1.0))


class GlobalAccumulator:
    """Standalone official scorer with stable float64 centered streaming moments.

    Selected rows are pooled across calls. No package imports or block eligibility
    rules are required, so this scorer can run in the competition environment.
    """

    def __init__(self) -> None:
        self.sums = np.zeros((2, 6), dtype=np.float64)
        self._moments = np.zeros((2, 6), dtype=np.float64)
        self.selected_rows = 0
        self.blocks_seen = 0

    def add(self, targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add a partition shaped (..., T, 2) with its combined selection mask."""
        y = np.asarray(targets, dtype=np.float32)
        p = np.asarray(predictions, dtype=np.float32)
        m = np.asarray(mask, dtype=bool)
        if y.ndim < 2 or y.shape[-1] != 2 or p.shape != y.shape or m.shape != y.shape[:-1]:
            raise ValueError("expected (..., T, 2) targets/predictions and (..., T) mask")
        if not np.isfinite(y[m]).all() or not np.isfinite(p[m]).all():
            raise ValueError("nonfinite selected target or prediction")
        y = np.clip(y[m], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        p = np.clip(p[m], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        added = np.zeros((2, 6), dtype=np.float64)
        for k in range(2):
            yk, pk = y[:, k], p[:, k]
            w = np.abs(yk)
            added[k] = [
                w.sum(),
                np.sum(w * yk),
                np.sum(w * pk),
                np.sum(w * yk * yk),
                np.sum(w * pk * pk),
                np.sum(w * yk * pk),
            ]
            positive = w > 0
            if not positive.any():
                continue
            yk, pk, w = yk[positive], pk[positive], w[positive]
            total = w.sum()
            ys, ps = yk - yk[0], pk - pk[0]
            my, mp = np.sum(w * ys) / total, np.sum(w * ps) / total
            yc, pc = ys - my, ps - mp
            batch = np.array(
                [
                    total,
                    yk[0] + my,
                    pk[0] + mp,
                    np.sum(w * yc * yc),
                    np.sum(w * pc * pc),
                    np.sum(w * yc * pc),
                ]
            )
            previous = self._moments[k]
            if previous[0] == 0:
                self._moments[k] = batch
                continue
            combined = previous[0] + total
            ratio = total / combined
            dy, dp = batch[1] - previous[1], batch[2] - previous[2]
            cross_weight = previous[0] * ratio
            self._moments[k] = [
                combined,
                previous[1] + dy * ratio,
                previous[2] + dp * ratio,
                previous[3] + batch[3] + dy * dy * cross_weight,
                previous[4] + batch[4] + dp * dp * cross_weight,
                previous[5] + batch[5] + dy * dp * cross_weight,
            ]
        self.sums += added
        self.selected_rows += int(m.sum())
        self.blocks_seen += 1
        return added

    def result(self) -> dict[str, float | int | str | list[float]]:
        """Return finite official scores, raising when no rows were selected."""
        if self.selected_rows == 0:
            raise ValueError("no selected rows")
        per_side = []
        for total, _my, _mp, vy, vp, covariance in self._moments:
            if total < _EPS:
                per_side.append(0.0)
                continue
            sy, sp = np.sqrt(np.maximum([vy, vp], 0.0) / total)
            rho = 0.0 if sy <= _EPS or sp <= _EPS else covariance / total / (sy * sp)
            per_side.append(float(np.clip(rho, -1.0, 1.0)))
        wp = 0.5 * (per_side[0] + per_side[1])
        return {
            "t0": per_side[0],
            "t1": per_side[1],
            "weighted_pearson": wp,
            "wp": wp,
            "blocks": self.blocks_seen,
            "selected_rows": self.selected_rows,
            "effective_scored_rows": self.selected_rows,
            "metric_clip": [-METRIC_CLIP, METRIC_CLIP],
            "metric_id": METRIC_ID,
            "aggregation": "pool all selected rows per target, then mean t0/t1",
        }


class BlockAccumulator:
    """Legacy equal block mean, retained only for diagnostic compatibility."""

    def __init__(self) -> None:
        self.scores: list[list[float]] = []
        self.blocks_seen = 0
        self.selected_rows = 0
        self.effective_rows = 0

    def add(self, targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> None:
        """Accumulate one complete 20k block under the legacy eligibility rule."""
        targets = np.asarray(targets, dtype=np.float32)
        predictions = np.asarray(predictions, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if (
            targets.shape != (SEQUENCE_LENGTH, 2)
            or predictions.shape != targets.shape
            or mask.shape != (SEQUENCE_LENGTH,)
        ):
            raise ValueError("one accumulator call must contain exactly one complete 20k block")
        if np.any(mask[:WARMUP]):
            raise ValueError("warm-up rows cannot be scored")
        if not np.isfinite(targets).all() or not np.isfinite(predictions[WARMUP:]).all():
            raise ValueError("nonfinite labels or required predictions")
        selected_targets = targets[mask]
        selected_predictions = predictions[mask]
        eligible = all(
            np.any(selected_targets[:, j] > 0) and np.any(selected_targets[:, j] < 0)
            for j in range(2)
        )
        self.blocks_seen += 1
        self.selected_rows += int(mask.sum())
        if eligible:
            values = [
                weighted_pearson(selected_targets[:, j], selected_predictions[:, j])
                for j in range(2)
            ]
            self.scores.append(values)
            self.effective_rows += int(mask.sum())

    def result(self) -> dict[str, float | int | str | list[float]]:
        """Return the legacy diagnostic, raising when no blocks are eligible."""
        if not self.scores:
            raise ValueError("no eligible blocks")
        per_side = np.mean(np.asarray(self.scores, dtype=np.float64), axis=0)
        return {
            "t0": float(per_side[0]),
            "t1": float(per_side[1]),
            "weighted_pearson": float(per_side.mean()),
            "blocks": self.blocks_seen,
            "eligible_blocks": len(self.scores),
            "excluded_blocks": self.blocks_seen - len(self.scores),
            "selected_rows": self.selected_rows,
            "effective_scored_rows": self.effective_rows,
            "metric_clip": [-METRIC_CLIP, METRIC_CLIP],
            "aggregation": "equal mean of eligible block-wise two-target clipped Weighted Pearson",
        }


def validate_sequence(table: pa.Table) -> tuple[int, np.ndarray]:
    """Validate one complete row group and return its sequence ID and need mask."""
    if table.num_rows != SEQUENCE_LENGTH:
        raise ValueError("each row group must be exactly one 20k sequence")
    ids = table["seq_ix"].to_numpy()
    if not np.all(ids == ids[0]):
        raise ValueError("multiple sequence IDs in one row group")
    if not np.array_equal(table["step_in_seq"].to_numpy(), np.arange(SEQUENCE_LENGTH)):
        raise ValueError("incorrect step order")
    need = table["need_prediction"].to_numpy().astype(bool)
    if not np.array_equal(need, np.arange(SEQUENCE_LENGTH) >= WARMUP):
        raise ValueError("incorrect warm-up contract")
    return int(ids[0]), need


class ScorerStepByStep:
    """Drive every causal row through a model and report official Global WP."""

    def __init__(self, dataset_path: str | Path) -> None:
        self.parquet = pq.ParquetFile(dataset_path)
        expected = [
            "seq_ix",
            "step_in_seq",
            "need_prediction",
            "is_scored",
            *FEATURE_COLUMNS,
            *TARGET_COLUMNS,
        ]
        if self.parquet.schema_arrow.names != expected:
            raise ValueError("local scoring requires the labeled validation schema")

    def score(self, model: PredictionModel) -> dict[str, float | int | str | list[float]]:
        """Require finite predictions on all requested rows and pool scored rows."""
        accumulator = GlobalAccumulator()
        seen = set()
        for group in range(self.parquet.num_row_groups):
            table = self.parquet.read_row_group(group, use_threads=False)
            seq, need = validate_sequence(table)
            if seq in seen:
                raise ValueError("duplicate sequence ID")
            seen.add(seq)
            features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(
                np.float32
            )
            targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(
                np.float32
            )
            predictions = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
            for step in range(SEQUENCE_LENGTH):
                point = DataPoint(seq, step, bool(need[step]), features[step])
                value = model.predict(point)
                if not need[step]:
                    if value is not None:
                        raise ValueError("return None during warm-up AFTER updating state")
                else:
                    value = np.asarray(value, dtype=np.float32)
                    if value.shape != (2,) or not np.isfinite(value).all():
                        raise ValueError("return two finite scores on every required row")
                    predictions[step] = value
            accumulator.add(targets, predictions, table["is_scored"].to_numpy().astype(bool) & need)
        result = accumulator.result()
        result["rows_seen"] = len(seen) * SEQUENCE_LENGTH
        result["predictions_required"] = len(seen) * (SEQUENCE_LENGTH - WARMUP)
        return result
