from __future__ import annotations

import numpy as np

from wnn.gmetric import GlobalAccumulator


class HeadMetrics:
    """Accumulate standalone z from already-computed forwards with the unchanged official grader."""

    def __init__(self) -> None:
        self.accumulator = GlobalAccumulator()
        self.error = np.zeros(2, dtype=np.float64)
        self.weight = np.zeros(2, dtype=np.float64)

    def add(self, target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> None:
        """Pool both target metrics; WMSE uses raw z versus clipped y and original abs(y) weights."""
        self.accumulator.add(target, prediction, mask)
        clipped = np.where(mask[..., None], np.clip(target, -2, 2), 0).astype(np.float64)
        predicted = np.where(mask[..., None], prediction, 0).astype(np.float64)
        weight = np.abs(clipped)
        self.error += (weight * np.square(predicted - clipped)).sum(
            axis=tuple(range(target.ndim - 1))
        )
        self.weight += weight.sum(axis=tuple(range(target.ndim - 1)))

    def metrics(self, prefix: str, component: str = "z") -> dict[str, float]:
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
        }


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
