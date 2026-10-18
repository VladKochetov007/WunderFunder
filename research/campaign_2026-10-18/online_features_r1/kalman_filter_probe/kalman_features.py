"""Causal NumPy-only per-sequence Kalman input features for 112-wide states.

This is a deterministic reference feature transform, not a trained estimator.
It observes current/past input channels only and resets all state per sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from kalman_normalization import RAW_COLUMNS, InputNormalizationPrior, identity_unscaled_prior

FEATURE_WIDTH = 40
EPS = 1.0e-12

FEATURE_SEMANTICS = (
    "next_forecast_minus_current",
    "preupdate_innovation",
    "measurement_leverage",
    "log_innovation_variance",
    "velocity",
)


@dataclass(frozen=True)
class KalmanConfig:
    """One fixed, predeclared scalar and local-trend Q/R configuration."""

    rw_q: float = 0.02
    rw_r: float = 0.5
    trend_q_level: float = 0.02
    trend_q_velocity: float = 0.0002
    trend_r: float = 0.5
    trend_initial_level_variance: float = 1.0
    trend_initial_velocity_variance: float = 0.1

    def __post_init__(self) -> None:
        if any(x <= 0.0 or not math.isfinite(x) for x in self.__dict__.values()):
            raise ValueError("all fixed Kalman variances must be finite and positive")


def steady_state_random_walk_gain(q: float, r: float) -> float:
    """Return the steady-state scalar gain for random-walk Q and observation R."""
    if q <= 0.0 or r <= 0.0:
        raise ValueError("Q and R must be positive")
    predicted = (q + math.sqrt(q * q + 4.0 * q * r)) / 2.0
    return predicted / (predicted + r)


def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
    """Symmetrize a 2x2 covariance and floor its minimum eigenvalue."""
    result = 0.5 * (covariance + covariance.T)
    a, b, d = float(result[0, 0]), float(result[0, 1]), float(result[1, 1])
    minimum = 0.5 * (a + d - math.hypot(a - d, 2.0 * b))
    if minimum < EPS:
        result = result.copy()
        result[0, 0] += EPS - minimum
        result[1, 1] += EPS - minimum
    return result


def feature_names() -> tuple[str, ...]:
    """Return the frozen model-major/channel-major feature order."""
    names: list[str] = []
    for model in ("rw", "localtrend"):
        for column in RAW_COLUMNS:
            names.extend(f"{model}_{semantic}_raw{column}" for semantic in FEATURE_SEMANTICS)
    return tuple(names)


class OnlineKalmanFeatures:
    """Streaming causal 40-feature stack; call reset at every sequence boundary."""

    def __init__(
        self,
        normalization: InputNormalizationPrior | None = None,
        config: KalmanConfig | None = None,
    ) -> None:
        self.normalization = normalization or identity_unscaled_prior()
        self.config = config or KalmanConfig()
        self._rw_level = np.zeros(4, dtype=np.float64)
        self._rw_cov = np.zeros(4, dtype=np.float64)
        self._trend_state = np.zeros((4, 2), dtype=np.float64)
        self._trend_cov = np.zeros((4, 2, 2), dtype=np.float64)
        self._initialized = np.zeros(4, dtype=bool)
        self.minimum_covariance_eigenvalue = math.inf
        self.reset_sequence()

    def reset_sequence(self) -> None:
        """Reset all estimator states and covariances at a sequence boundary."""
        cfg = self.config
        self._initialized[:] = False
        self._rw_level.fill(0.0)
        self._rw_cov.fill((math.sqrt(cfg.rw_q**2 + 4 * cfg.rw_q * cfg.rw_r) - cfg.rw_q) / 2)
        self._trend_state.fill(0.0)
        self._trend_cov.fill(0.0)
        self._trend_cov[:, 0, 0] = cfg.trend_initial_level_variance
        self._trend_cov[:, 1, 1] = cfg.trend_initial_velocity_variance

    def _remember_covariance(self, covariance: np.ndarray) -> None:
        a, b, d = float(covariance[0, 0]), float(covariance[0, 1]), float(covariance[1, 1])
        minimum = 0.5 * (a + d - math.hypot(a - d, 2.0 * b))
        self.minimum_covariance_eigenvalue = min(self.minimum_covariance_eigenvalue, minimum)

    def step_selected(self, raw_selected: np.ndarray) -> np.ndarray:
        """Consume one current row's four raw channels and emit 40 causal features."""
        raw = np.asarray(raw_selected, dtype=np.float64)
        if raw.shape != (4,) or not np.isfinite(raw).all():
            raise ValueError("one finite four-channel input row is required")
        x = (raw - self.normalization.mean) / self.normalization.scale
        result = np.empty(FEATURE_WIDTH, dtype=np.float32)
        rw = result[:20].reshape(4, 5)
        trend = result[20:].reshape(4, 5)
        cfg = self.config
        for channel, value in enumerate(x):
            if not self._initialized[channel]:
                self._rw_level[channel] = value
                self._trend_state[channel] = (value, 0.0)
                self._initialized[channel] = True
                rw[channel] = (0.0, 0.0, 0.0, math.log(cfg.rw_r), 0.0)
                trend[channel] = (0.0, 0.0, 0.0, math.log(cfg.trend_r), 0.0)
                continue
            self._step_random_walk(channel, float(value), rw[channel])
            self._step_local_trend(channel, float(value), trend[channel])
        if not np.isfinite(result).all():
            raise FloatingPointError("Kalman feature transform produced nonfinite output")
        return result

    def _step_random_walk(self, channel: int, value: float, output: np.ndarray) -> None:
        """Update one scalar random-walk filter using a prior innovation."""
        cfg = self.config
        prior_level = self._rw_level[channel]
        prior_cov = self._rw_cov[channel] + cfg.rw_q
        innovation = value - prior_level
        variance = max(prior_cov + cfg.rw_r, EPS)
        gain = prior_cov / variance
        level = prior_level + gain * innovation
        cov = (1.0 - gain) ** 2 * prior_cov + gain**2 * cfg.rw_r
        self._rw_level[channel] = level
        self._rw_cov[channel] = max(cov, EPS)
        output[:] = (level - value, innovation, gain, math.log(variance), level - prior_level)

    def _step_local_trend(self, channel: int, value: float, output: np.ndarray) -> None:
        """Update a level/drift filter with the Joseph covariance equation."""
        cfg = self.config
        transition = np.array(((1.0, 1.0), (0.0, 1.0)))
        process = np.diag((cfg.trend_q_level, cfg.trend_q_velocity))
        prior = transition @ self._trend_state[channel]
        cov_prior = _stabilize_covariance(
            transition @ self._trend_cov[channel] @ transition.T + process
        )
        variance = max(float(cov_prior[0, 0]) + cfg.trend_r, EPS)
        gain = cov_prior[:, 0] / variance
        innovation = value - float(prior[0])
        posterior = prior + gain * innovation
        residual_map = np.array(((1.0 - gain[0], 0.0), (-gain[1], 1.0)))
        covariance = _stabilize_covariance(
            residual_map @ cov_prior @ residual_map.T + cfg.trend_r * np.outer(gain, gain)
        )
        self._trend_state[channel] = posterior
        self._trend_cov[channel] = covariance
        self._remember_covariance(cov_prior)
        self._remember_covariance(covariance)
        forecast = float(posterior[0] + posterior[1])
        output[:] = (
            forecast - value,
            innovation,
            gain[0],
            math.log(variance),
            posterior[1],
        )

    def step(self, raw_state: np.ndarray) -> np.ndarray:
        """Consume a full current 112-channel row without modifying its raw values."""
        state = np.asarray(raw_state)
        if state.shape != (112,):
            raise ValueError("raw state must have exactly 112 channels")
        return self.step_selected(state[list(RAW_COLUMNS)])

    def transform_selected_sequence(self, raw_sequence: np.ndarray) -> np.ndarray:
        """Reset and transform a bounded sequence containing only selected raw inputs."""
        values = np.asarray(raw_sequence)
        if values.ndim != 2 or values.shape[1] != 4 or len(values) > 20_000:
            raise ValueError("selected sequence must have at most 20,000 rows and four channels")
        self.reset_sequence()
        output = np.empty((len(values), FEATURE_WIDTH), dtype=np.float32)
        for row, value in enumerate(values):
            output[row] = self.step_selected(value)
        return output

    def transform_sequence(self, raw_sequence: np.ndarray) -> np.ndarray:
        """Reset and transform a bounded full-width sequence."""
        values = np.asarray(raw_sequence)
        if values.ndim != 2 or values.shape[1] != 112 or len(values) > 20_000:
            raise ValueError("raw sequence must have at most 20,000 rows and 112 channels")
        return self.transform_selected_sequence(values[:, RAW_COLUMNS])

    def filter_selected_sequence(
        self,
        raw_sequence: np.ndarray,
        model: Literal["random_walk", "local_trend"],
    ) -> np.ndarray:
        """Return four denormalized filtered levels for explicit input injection."""
        raw = np.asarray(raw_sequence, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 4 or len(raw) > 20_000:
            raise ValueError("selected sequence must have at most 20,000 rows and four channels")
        if model not in {"random_walk", "local_trend"}:
            raise ValueError("model must be random_walk or local_trend")
        features = self.transform_selected_sequence(raw)
        normalized = (raw - self.normalization.mean) / self.normalization.scale
        if model == "random_walk":
            filtered = np.empty_like(normalized)
            for channel in range(4):
                filtered[:, channel] = normalized[:, channel] + features[:, 5 * channel]
        else:
            filtered = np.empty_like(normalized)
            for channel in range(4):
                offset = 20 + 5 * channel
                filtered[:, channel] = (
                    normalized[:, channel] + features[:, offset] - features[:, offset + 4]
                )
        return (filtered * self.normalization.scale + self.normalization.mean).astype(np.float32)

    def step_filtered_selected(
        self, raw_selected: np.ndarray, model: Literal["random_walk", "local_trend"]
    ) -> np.ndarray:
        """Consume one row and return the selected model's posterior level."""
        if model not in {"random_walk", "local_trend"}:
            raise ValueError("model must be random_walk or local_trend")
        raw = np.asarray(raw_selected, dtype=np.float64)
        features = self.step_selected(raw)
        normalized = (raw - self.normalization.mean) / self.normalization.scale
        if model == "random_walk":
            filtered = np.array(
                [normalized[i] + features[5 * i] for i in range(4)], dtype=np.float64
            )
        else:
            filtered = np.array(
                [normalized[i] + features[20 + 5 * i] - features[20 + 5 * i + 4] for i in range(4)],
                dtype=np.float64,
            )
        return (filtered * self.normalization.scale + self.normalization.mean).astype(np.float32)

    def augment_sequence(self, raw_sequence: np.ndarray) -> np.ndarray:
        """Append causal features while retaining every untouched raw input channel."""
        raw = np.asarray(raw_sequence)
        if raw.ndim != 2 or raw.shape[1] != 112:
            raise ValueError("raw sequence must be an Nx112 array")
        return np.concatenate((raw, self.transform_sequence(raw)), axis=1)

    def covariance_min_eigenvalue(self) -> float:
        """Return the minimum local-trend covariance eigenvalue seen since construction."""
        return self.minimum_covariance_eigenvalue
