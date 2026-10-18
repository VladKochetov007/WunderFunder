"""Stable weighted-ridge accumulation and prediction for the screen."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RIDGE_HELPERS = ROOT / "research/campaign_2026-10-18/online_features_r1/causal_label_proxy_r1"
sys.path.insert(0, str(RIDGE_HELPERS))

from weighted_ridge import merge_weighted_batch  # noqa: E402


@dataclass
class WeightedStats:
    """Streaming weighted sufficient statistics for one target and one arm."""

    total_weight: float
    mean_x: np.ndarray
    mean_y: float
    centered_xx: np.ndarray
    centered_xy: np.ndarray
    rows: int

    @classmethod
    def empty(cls, width: int) -> WeightedStats:
        """Allocate an empty accumulator of the requested feature width."""
        return cls(
            total_weight=0.0,
            mean_x=np.zeros(width, dtype=np.float64),
            mean_y=0.0,
            centered_xx=np.zeros((width, width), dtype=np.float64),
            centered_xy=np.zeros(width, dtype=np.float64),
            rows=0,
        )

    def update(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> None:
        """Merge one sequence's sampled rows using stable centered moments."""
        (
            self.total_weight,
            self.mean_x,
            self.mean_y,
            self.centered_xx,
            self.centered_xy,
        ) = merge_weighted_batch(
            self.total_weight,
            self.mean_x,
            self.mean_y,
            self.centered_xx,
            self.centered_xy,
            features,
            targets,
            weights,
        )
        self.rows += int(len(targets))


@dataclass(frozen=True)
class WeightedRidge:
    """One target-specific, weighted-standardized ridge model."""

    mean_x: np.ndarray
    scale_x: np.ndarray
    coefficients: np.ndarray
    intercept: float

    @classmethod
    def fit(cls, stats: WeightedStats, alpha: float) -> WeightedRidge:
        """Fit standardized ridge from the accumulated centered moments."""
        if stats.total_weight <= 0.0:
            raise ValueError("ridge fit has no positive target weight")
        covariance = stats.centered_xx / stats.total_weight
        covariance = (covariance + covariance.T) * 0.5
        scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        normalized_covariance = covariance / np.outer(scale, scale)
        normalized_cross = (stats.centered_xy / stats.total_weight) / scale
        standardized_coef = np.linalg.solve(
            normalized_covariance + alpha * np.eye(len(scale)), normalized_cross
        )
        coefficients = standardized_coef / scale
        intercept = stats.mean_y - float(stats.mean_x @ coefficients)
        return cls(stats.mean_x.copy(), scale, coefficients, intercept)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict one target and reject non-finite model output."""
        predictions = np.asarray(features, dtype=np.float64) @ self.coefficients + self.intercept
        if not np.isfinite(predictions).all():
            raise FloatingPointError("ridge model produced non-finite predictions")
        return predictions.astype(np.float32)
