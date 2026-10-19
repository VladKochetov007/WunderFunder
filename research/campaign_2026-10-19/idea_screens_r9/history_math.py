"""Small scalar accumulators for the fixed recurrent-history comparison."""

from __future__ import annotations

import numpy as np
from attrs import define, field


@define
class ScalarDifference:
    """Additive scalar sufficient statistics for one reference/alternative pair."""

    count: int = 0
    delta_square_sum: float = 0.0
    reference_square_sum: float = 0.0
    absolute_difference_sum: float = 0.0
    max_absolute_difference: float = 0.0

    def add(self, reference: np.ndarray, alternative: np.ndarray) -> None:
        """Accumulate finite paired values without retaining either input array."""
        left = np.asarray(reference, dtype=np.float64)
        right = np.asarray(alternative, dtype=np.float64)
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("paired prediction values must have matching finite shapes")
        difference = right - left
        absolute = np.abs(difference)
        self.count += int(left.size)
        self.delta_square_sum += float(np.square(difference).sum())
        self.reference_square_sum += float(np.square(left).sum())
        self.absolute_difference_sum += float(absolute.sum())
        self.max_absolute_difference = max(
            self.max_absolute_difference, float(absolute.max(initial=0.0))
        )

    def merge(self, other: ScalarDifference) -> None:
        """Merge another accumulator's exact additive statistics."""
        self.count += other.count
        self.delta_square_sum += other.delta_square_sum
        self.reference_square_sum += other.reference_square_sum
        self.absolute_difference_sum += other.absolute_difference_sum
        self.max_absolute_difference = max(
            self.max_absolute_difference, other.max_absolute_difference
        )

    def summary(self) -> dict[str, int | float]:
        """Return RMS, relative RMS, and sufficient statistics for independent pooling."""
        if self.count <= 0:
            raise ValueError("cannot summarize an empty prediction accumulator")
        return {
            "count": self.count,
            "delta_square_sum": self.delta_square_sum,
            "reference_square_sum": self.reference_square_sum,
            "absolute_difference_sum": self.absolute_difference_sum,
            "max_absolute_difference": self.max_absolute_difference,
            "reference_rms": float(np.sqrt(self.reference_square_sum / self.count)),
            "rms_difference": float(np.sqrt(self.delta_square_sum / self.count)),
            "relative_rms_difference": float(
                np.sqrt(self.delta_square_sum / max(self.reference_square_sum, 1e-12))
            ),
            "mean_absolute_difference": self.absolute_difference_sum / self.count,
        }


@define
class PredictionDifference:
    """Pool a two-target prediction contrast with target and joint summaries."""

    by_target: list[ScalarDifference] = field(
        factory=lambda: [ScalarDifference(), ScalarDifference()]
    )
    pooled: ScalarDifference = field(factory=ScalarDifference)

    def add(self, reference: np.ndarray, alternative: np.ndarray) -> None:
        """Accumulate matching ``[..., 2]`` prediction blocks."""
        left = np.asarray(reference)
        right = np.asarray(alternative)
        if left.shape != right.shape or left.ndim < 1 or left.shape[-1] != 2:
            raise ValueError("prediction blocks must have matching final width two")
        for target in range(2):
            self.by_target[target].add(left[..., target], right[..., target])
        self.pooled.add(left, right)

    def merge(self, other: PredictionDifference) -> None:
        """Merge all target and pooled sufficient statistics."""
        for target in range(2):
            self.by_target[target].merge(other.by_target[target])
        self.pooled.merge(other.pooled)

    def summary(self) -> dict[str, object]:
        """Return per-target and pooled scalar summaries."""
        return {
            "target_0": self.by_target[0].summary(),
            "target_1": self.by_target[1].summary(),
            "pooled": self.pooled.summary(),
        }


def state_distance(reference: np.ndarray, alternative: np.ndarray) -> dict[str, float]:
    """Summarize one recurrent-state deviation using absolute and relative L2 scales."""
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(alternative, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("recurrent states must have matching nonempty shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise FloatingPointError("recurrent state is nonfinite")
    delta = right - left
    reference_l2 = float(np.linalg.norm(left.ravel()))
    difference_l2 = float(np.linalg.norm(delta.ravel()))
    return {
        "reference_l2": reference_l2,
        "alternative_l2": float(np.linalg.norm(right.ravel())),
        "reference_rms": float(np.sqrt(np.square(left).mean())),
        "difference_l2": difference_l2,
        "difference_rms": float(np.sqrt(np.square(delta).mean())),
        "relative_difference_l2": difference_l2 / max(reference_l2, 1e-12),
        "max_absolute_difference": float(np.abs(delta).max(initial=0.0)),
    }
