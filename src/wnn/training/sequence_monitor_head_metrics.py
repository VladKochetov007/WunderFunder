from __future__ import annotations

import numpy as np

from wnn.gmetric import GlobalAccumulator


class _StreamingMoments:
    """Keep bounded per-target raw and official-weighted prediction sufficient statistics."""

    def __init__(self) -> None:
        self.count = 0
        self.raw_sum = np.zeros(2, dtype=np.float64)
        self.raw_square_sum = np.zeros(2, dtype=np.float64)
        self.weight_mass = np.zeros(2, dtype=np.float64)
        self.weighted_sum = np.zeros(2, dtype=np.float64)
        self.weighted_square_sum = np.zeros(2, dtype=np.float64)
        self.clipped_count = np.zeros(2, dtype=np.int64)
        self.clipped_weight_mass = np.zeros(2, dtype=np.float64)

    def add(self, values: np.ndarray, weights: np.ndarray) -> None:
        """Add selected rows without retaining the underlying sequence arrays."""
        values = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1, 2)
        if values.shape != weights.shape:
            raise ValueError("monitor moments require matching two-target values and weights")
        if not np.isfinite(values).all() or not np.isfinite(weights).all():
            raise ValueError("monitor moments require finite selected values and weights")
        self.count += len(values)
        self.raw_sum += values.sum(axis=0)
        self.raw_square_sum += np.square(values).sum(axis=0)
        self.weight_mass += weights.sum(axis=0)
        self.weighted_sum += (weights * values).sum(axis=0)
        self.weighted_square_sum += (weights * np.square(values)).sum(axis=0)
        clipped = np.abs(values) > 2.0
        self.clipped_count += clipped.sum(axis=0)
        self.clipped_weight_mass += (weights * clipped).sum(axis=0)

    def metrics(
        self,
        prefix: str,
        value_name: str,
        *,
        clipping_name: str | None = None,
    ) -> dict[str, int | float]:
        """Return finite raw, weighted and optional clipping summaries by target."""
        means = self.raw_sum / self.count if self.count else np.zeros(2)
        variances = (
            self.raw_square_sum / self.count - np.square(means) if self.count else np.zeros(2)
        )
        weighted_means = np.divide(
            self.weighted_sum,
            self.weight_mass,
            out=np.zeros(2, dtype=np.float64),
            where=self.weight_mass > 0,
        )
        weighted_variances = np.divide(
            self.weighted_square_sum,
            self.weight_mass,
            out=np.zeros(2, dtype=np.float64),
            where=self.weight_mass > 0,
        ) - np.square(weighted_means)
        result: dict[str, int | float] = {}
        for target in range(2):
            suffix = f"t{target}"
            result[f"{prefix}/{value_name}_mean/{suffix}"] = float(means[target])
            result[f"{prefix}/{value_name}_std/{suffix}"] = float(
                np.sqrt(max(0.0, variances[target]))
            )
            result[f"{prefix}/weighted_{value_name}_mean/{suffix}"] = float(weighted_means[target])
            result[f"{prefix}/weighted_{value_name}_std/{suffix}"] = float(
                np.sqrt(max(0.0, weighted_variances[target]))
            )
            if clipping_name is not None:
                result[f"{prefix}/{clipping_name}_count/{suffix}"] = int(self.clipped_count[target])
                result[f"{prefix}/{clipping_name}_fraction/{suffix}"] = (
                    float(self.clipped_count[target] / self.count) if self.count else 0.0
                )
                result[f"{prefix}/{clipping_name}_weight_mass/{suffix}"] = float(
                    self.clipped_weight_mass[target]
                )
        return result


class TargetMoments:
    """Accumulate raw/clipped target moments and the official clipped-label weights."""

    def __init__(self) -> None:
        self.raw = _StreamingMoments()
        self.clipped = _StreamingMoments()
        self.raw_abs_mass = np.zeros(2, dtype=np.float64)
        self.official_weight_mass = np.zeros(2, dtype=np.float64)
        self.clipped_weight_mass_delta = np.zeros(2, dtype=np.float64)

    def add(self, target: np.ndarray, mask: np.ndarray) -> None:
        """Add targets selected by the actual cohort mask, preserving original values."""
        values = np.asarray(target, dtype=np.float64)[np.asarray(mask, dtype=bool)]
        clipped = np.clip(values, -2.0, 2.0)
        weights = np.abs(clipped)
        self.raw.add(values, weights)
        self.clipped.add(clipped, weights)
        raw_mass = np.abs(values).sum(axis=0)
        official_mass = weights.sum(axis=0)
        self.raw_abs_mass += raw_mass
        self.official_weight_mass += official_mass
        self.clipped_weight_mass_delta += raw_mass - official_mass

    def metrics(self, prefix: str) -> dict[str, int | float]:
        """Report per-target moments, official weights, and target clipping effects."""
        result = self.raw.metrics(f"{prefix}/target", "raw", clipping_name="clipped")
        result.update(self.clipped.metrics(f"{prefix}/target", "clipped"))
        for target in range(2):
            suffix = f"t{target}"
            result[f"{prefix}/target/raw_abs_mass/{suffix}"] = float(self.raw_abs_mass[target])
            result[f"{prefix}/target/official_weight_mass/{suffix}"] = float(
                self.official_weight_mass[target]
            )
            result[f"{prefix}/target/clipped_weight_mass_delta/{suffix}"] = float(
                self.clipped_weight_mass_delta[target]
            )
        return result


class HeadMetrics:
    """Accumulate one raw prediction component with the unchanged official grader."""

    def __init__(self) -> None:
        self.accumulator = GlobalAccumulator()
        self.error = np.zeros(2, dtype=np.float64)
        self.weight = np.zeros(2, dtype=np.float64)
        self.moments = _StreamingMoments()

    def add(self, target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> None:
        """Pool raw predictions using official absolute clipped-label weights."""
        self.accumulator.add(target, prediction, mask)
        clipped = np.where(mask[..., None], np.clip(target, -2, 2), 0).astype(np.float64)
        predicted = np.where(mask[..., None], prediction, 0).astype(np.float64)
        weight = np.abs(clipped)
        self.moments.add(prediction[mask], weight[mask])
        self.error += (weight * np.square(predicted - clipped)).sum(
            axis=tuple(range(target.ndim - 1))
        )
        self.weight += weight.sum(axis=tuple(range(target.ndim - 1)))

    def metrics(self, prefix: str, component: str = "z") -> dict[str, int | float]:
        """Report global standalone quality; no learned blend or extra inference is introduced."""
        score = self.accumulator.result()
        prefix = f"{prefix}/{component}"
        return {
            f"{prefix}/wp": float(score["wp"]),
            f"{prefix}/wp/t0": float(score["t0"]),
            f"{prefix}/wp/t1": float(score["t1"]),
            f"{prefix}/wmse": float(self.error.sum() / self.weight.sum())
            if self.weight.sum()
            else 0.0,
            f"{prefix}/wmse/t0": float(self.error[0] / self.weight[0]) if self.weight[0] else 0.0,
            f"{prefix}/wmse/t1": float(self.error[1] / self.weight[1]) if self.weight[1] else 0.0,
        } | self.moments.metrics(prefix, "raw", clipping_name="prediction_clipped")


def add_score_aliases(
    train_metrics: dict[str, int | float],
    role_metrics: dict[str, int | float],
    role: str,
) -> None:
    """Add plotting-friendly score aliases while retaining the raw WP fields."""
    train_metrics["train_eval/score"] = train_metrics["train_eval/wp"]
    train_metrics["train_eval/score/t0"] = train_metrics["train_eval/wp/t0"]
    train_metrics["train_eval/score/t1"] = train_metrics["train_eval/wp/t1"]
    role_metrics[f"{role}/score"] = role_metrics[f"{role}/wp"]
    role_metrics[f"{role}/score/t0"] = role_metrics[f"{role}/wp/t0"]
    role_metrics[f"{role}/score/t1"] = role_metrics[f"{role}/wp/t1"]
