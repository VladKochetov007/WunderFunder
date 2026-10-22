"""Causal online i1-to-i0 relationship features for matched residual screens."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from wnn.data import FEATURE_COLUMNS, I0, I1, N_FEATURES
from wnn.features import N_EVENT_FEATURES

OnlineRelationshipArm = Literal["A", "B", "C"]
OnlineRelationshipLag = Literal[1, 16]
HORIZON = 16
CHANNELS = 52
FEATURE_WIDTH = 4 * CHANNELS
BASE_INPUT_WIDTH = N_FEATURES + N_EVENT_FEATURES
MODEL_INPUT_WIDTH = BASE_INPUT_WIDTH + FEATURE_WIDTH
FEATURE_NAMES = (
    tuple(f"matured_{name}" for name in FEATURE_COLUMNS[:CHANNELS])
    + tuple(f"forecast_{name}" for name in FEATURE_COLUMNS[:CHANNELS])
    + tuple(f"error_{name}" for name in FEATURE_COLUMNS[:CHANNELS])
    + tuple(f"slope_{name}" for name in FEATURE_COLUMNS[:CHANNELS])
)
NORMALIZER_FLOOR = np.float32(0.001)
FEATURE_CLIP = np.float32(8.0)
_DECAY = np.float32(math.exp2(-1.0 / 512.0))
_UPDATE = np.float32(1.0 - float(_DECAY))
_VARIANCE_EPSILON = np.float32(0.1)
_SCHEMA = "online_i1_i0_relationship_v3"
_I0_CHANNELS = slice(I0.bid_px.start, I0.trade_vol.stop)
_I1_CHANNELS = slice(I1.bid_px.start, I1.trade_vol.stop)


@dataclass(frozen=True)
class OnlineRelationshipNormalizer:
    """Fitting-only standardization for the 52-channel u and v differences."""

    u_mean: np.ndarray
    u_scale: np.ndarray
    v_mean: np.ndarray
    v_scale: np.ndarray
    lag: OnlineRelationshipLag

    def __post_init__(self) -> None:
        """Validate and freeze separate fitting statistics for both differences."""
        _validate_lag(self.lag)
        for name in ("u_mean", "u_scale", "v_mean", "v_scale"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (CHANNELS,) or not np.isfinite(value).all():
                raise ValueError("relationship normalization requires 52 finite values per vector")
            if name.endswith("scale") and np.any(value < NORMALIZER_FLOOR):
                raise ValueError("relationship normalization scales must be at least 0.001")
            value = value.copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-compatible normalizer representation."""
        return {
            "schema": _SCHEMA,
            "lag": self.lag,
            "u_mean": self.u_mean.tolist(),
            "u_scale": self.u_scale.tolist(),
            "v_mean": self.v_mean.tolist(),
            "v_scale": self.v_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OnlineRelationshipNormalizer:
        """Validate and restore a fitting-only normalizer representation."""
        if value.get("schema") != _SCHEMA:
            raise ValueError("relationship normalizer has an unsupported schema")
        return cls(
            np.asarray(value.get("u_mean"), dtype=np.float32),
            np.asarray(value.get("u_scale"), dtype=np.float32),
            np.asarray(value.get("v_mean"), dtype=np.float32),
            np.asarray(value.get("v_scale"), dtype=np.float32),
            cast(OnlineRelationshipLag, value.get("lag")),
        )


def fit_online_relationship_normalizer(
    raw_sequences: Iterable[np.ndarray], lag: OnlineRelationshipLag
) -> OnlineRelationshipNormalizer:
    """Fit separate u/v means and scales from the supplied fitting sequences only.

    The caller owns and records source-qualified membership. u statistics use
    rows where the selected i1 lag exists; v statistics use rows from maturity
    row 16. The downstream regression still consumes zero-u placeholders for
    earlier matured pairs (16 of its 83 B updates when lag is 16).
    """
    _validate_lag(lag)
    u_mean = np.zeros(CHANNELS, dtype=np.float32)
    u_m2 = np.zeros(CHANNELS, dtype=np.float32)
    v_mean = np.zeros(CHANNELS, dtype=np.float32)
    v_m2 = np.zeros(CHANNELS, dtype=np.float32)
    u_count = 0
    v_count = 0
    sequence_count = 0
    for raw in raw_sequences:
        sequence = _validate_sequence(raw)
        if not len(sequence):
            continue
        sequence_count += 1
        if len(sequence) > lag:
            u = sequence[lag:, _I1_CHANNELS] - sequence[:-lag, _I1_CHANNELS]
            u_mean, u_m2 = _merge_centered_moments(u_mean, u_m2, u_count, u)
            u_count += len(u)
        if len(sequence) > HORIZON:
            v = sequence[HORIZON:, _I0_CHANNELS] - sequence[:-HORIZON, _I0_CHANNELS]
            v_mean, v_m2 = _merge_centered_moments(v_mean, v_m2, v_count, v)
            v_count += len(v)
    if sequence_count == 0 or u_count == 0 or v_count == 0:
        raise ValueError(
            "cannot fit relationship normalization without fitting u and v differences"
        )
    u_scale = np.maximum(np.sqrt(u_m2 / np.float32(u_count)), NORMALIZER_FLOOR)
    v_scale = np.maximum(np.sqrt(v_m2 / np.float32(v_count)), NORMALIZER_FLOOR)
    return OnlineRelationshipNormalizer(u_mean, u_scale, v_mean, v_scale, lag)


class OnlineRelationshipState:
    """Maintain lagged causal i1-to-i0 statistics within one sequence."""

    def __init__(
        self,
        arm: OnlineRelationshipArm,
        lag: OnlineRelationshipLag,
        normalizer: OnlineRelationshipNormalizer | None = None,
    ) -> None:
        _validate_arm(arm)
        _validate_lag(lag)
        if normalizer is not None and normalizer.lag != lag:
            raise ValueError("relationship normalizer lag must match the stream")
        self.arm = arm
        self.lag = lag
        self.normalizer = normalizer
        self.reset()

    def reset(self) -> None:
        """Clear lagged input history and the online coefficient state."""
        self.rows = 0
        self.i1_history = np.zeros((self.lag, CHANNELS), dtype=np.float32)
        self.i0_history = np.zeros((HORIZON, CHANNELS), dtype=np.float32)
        self.u_history = np.zeros((HORIZON, CHANNELS), dtype=np.float32)
        self.mx = np.zeros(CHANNELS, dtype=np.float32)
        self.my = np.zeros(CHANNELS, dtype=np.float32)
        self.variance = np.ones(CHANNELS, dtype=np.float32)
        self.covariance = np.zeros(CHANNELS, dtype=np.float32)

    def update(self, row: np.ndarray) -> np.ndarray:
        """Return this row's 208 fields, using only matured pairs and its prefix."""
        raw = np.asarray(row, dtype=np.float32)
        if raw.shape != (N_FEATURES,) or not np.isfinite(raw).all():
            raise ValueError("expected one finite 112-wide raw row")
        current_i0 = raw[_I0_CHANNELS]
        current_i1 = raw[_I1_CHANNELS]
        lag_slot = self.rows % self.lag
        u_available = self.rows >= self.lag
        raw_u = current_i1 - self.i1_history[lag_slot] if u_available else None
        u = (
            self._standardize_u(raw_u)
            if raw_u is not None
            else np.zeros(CHANNELS, dtype=np.float32)
        )
        slot = self.rows % HORIZON
        features = np.zeros(FEATURE_WIDTH, dtype=np.float32)
        if self.rows >= HORIZON:
            matured_v = self._standardize_v(current_i0 - self.i0_history[slot])
            matured_u = self.u_history[slot]
            old_slope = self._slope()
            old_intercept = self.my - old_slope * self.mx
            error = matured_v - (old_intercept + old_slope * matured_u)
            if self._should_update():
                self._update_pair(matured_u, matured_v)
            slope = self._slope()
            intercept = self.my - slope * self.mx
            forecast = intercept + slope * u
            features[:] = np.clip(
                np.concatenate((matured_v, forecast, error, slope)),
                -FEATURE_CLIP,
                FEATURE_CLIP,
            )
        self.i0_history[slot] = current_i0
        self.u_history[slot] = u
        self.i1_history[lag_slot] = current_i1
        self.rows += 1
        return features

    def transform(self, raw_rows: np.ndarray) -> np.ndarray:
        """Process a sequence chunk while carrying state into the next chunk."""
        sequence = _validate_sequence(raw_rows)
        output = np.empty((len(sequence), FEATURE_WIDTH), dtype=np.float32)
        for index, row in enumerate(sequence):
            output[index] = self.update(row)
        return output

    def transform_compiled(self, raw_rows: np.ndarray) -> np.ndarray:
        """Process a chunk with the strict-FP32 serial Numba kernel."""
        from wnn.online_relationship_numba import transform_relationship_rows

        sequence = _validate_sequence(raw_rows)
        if self.normalizer is None:
            u_mean = np.zeros(CHANNELS, dtype=np.float32)
            u_scale = np.ones(CHANNELS, dtype=np.float32)
            v_mean = np.zeros(CHANNELS, dtype=np.float32)
            v_scale = np.ones(CHANNELS, dtype=np.float32)
            normalized = False
        else:
            u_mean = self.normalizer.u_mean
            u_scale = self.normalizer.u_scale
            v_mean = self.normalizer.v_mean
            v_scale = self.normalizer.v_scale
            normalized = True
        output, self.rows = transform_relationship_rows(
            sequence,
            self.rows,
            0 if self.arm == "A" else 1 if self.arm == "B" else 2,
            self.lag,
            normalized,
            u_mean,
            u_scale,
            v_mean,
            v_scale,
            _DECAY,
            _UPDATE,
            _VARIANCE_EPSILON,
            FEATURE_CLIP,
            self.i1_history,
            self.i0_history,
            self.u_history,
            self.mx,
            self.my,
            self.variance,
            self.covariance,
        )
        return output

    def state_dict(self) -> dict[str, object]:
        """Copy causal state and normalizer content for exact chunk continuation."""
        return {
            "schema": _SCHEMA,
            "arm": self.arm,
            "lag": self.lag,
            "normalizer": None if self.normalizer is None else self.normalizer.to_dict(),
            "rows": self.rows,
            "i1_history": self.i1_history.copy(),
            "i0_history": self.i0_history.copy(),
            "u_history": self.u_history.copy(),
            "mx": self.mx.copy(),
            "my": self.my.copy(),
            "variance": self.variance.copy(),
            "covariance": self.covariance.copy(),
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        """Restore compatible causal state, rejecting malformed or different arms."""
        if (
            value.get("schema") != _SCHEMA
            or value.get("arm") != self.arm
            or value.get("lag") != self.lag
        ):
            raise ValueError("relationship state schema, arm, or lag does not match")
        expected_normalizer = None if self.normalizer is None else self.normalizer.to_dict()
        if value.get("normalizer") != expected_normalizer:
            raise ValueError("relationship state normalizer content does not match")
        rows = value.get("rows")
        if not isinstance(rows, int) or rows < 0:
            raise ValueError("relationship state row count must be a nonnegative integer")
        expected = {
            "i1_history": (self.lag, CHANNELS),
            "i0_history": (HORIZON, CHANNELS),
            "u_history": (HORIZON, CHANNELS),
            "mx": (CHANNELS,),
            "my": (CHANNELS,),
            "variance": (CHANNELS,),
            "covariance": (CHANNELS,),
        }
        restored: dict[str, np.ndarray] = {}
        for name, shape in expected.items():
            array = np.asarray(value.get(name), dtype=np.float32)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"relationship state field {name!r} is malformed")
            restored[name] = array.copy()
        if np.any(restored["variance"] <= 0):
            raise ValueError("relationship state variance must stay positive")
        self.rows = rows
        for name, array in restored.items():
            setattr(self, name, array)

    def _standardize_u(self, value: np.ndarray) -> np.ndarray:
        if self.normalizer is None:
            normalized = value
        else:
            normalized = (value - self.normalizer.u_mean) / self.normalizer.u_scale
        return np.clip(normalized, -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)

    def _standardize_v(self, value: np.ndarray) -> np.ndarray:
        if self.normalizer is None:
            normalized = value
        else:
            normalized = (value - self.normalizer.v_mean) / self.normalizer.v_scale
        return np.clip(normalized, -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)

    def _slope(self) -> np.ndarray:
        if self.arm == "A":
            return np.zeros(CHANNELS, dtype=np.float32)
        return self.covariance / (self.variance + _VARIANCE_EPSILON)

    def _should_update(self) -> bool:
        return self.arm != "B" or self.rows <= 98

    def _update_pair(self, x: np.ndarray, y: np.ndarray) -> None:
        dx = x - self.mx
        dy = y - self.my
        dx_squared = dx * dx
        dx_dy = dx * dy
        self.mx += _UPDATE * dx
        self.my += _UPDATE * dy
        self.variance = _DECAY * (self.variance + _UPDATE * dx_squared)
        self.covariance = _DECAY * (self.covariance + _UPDATE * dx_dy)


def _validate_arm(arm: str) -> None:
    if not isinstance(arm, str) or arm not in ("A", "B", "C"):
        raise ValueError("relationship arm must be A, B, or C")


def _validate_lag(lag: int) -> None:
    if isinstance(lag, bool) or not isinstance(lag, int) or lag not in (1, 16):
        raise ValueError("relationship backward lag must be 1 or 16")


def _merge_centered_moments(
    mean: np.ndarray, m2: np.ndarray, count: int, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge one fitting sequence's FP32 centered moments into the running totals."""
    batch_count = len(values)
    batch_mean = values.mean(axis=0, dtype=np.float32)
    centered = values - batch_mean
    batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float32)
    if count == 0:
        return batch_mean, batch_m2
    old_count = np.float32(count)
    new_count = np.float32(batch_count)
    total_count = old_count + new_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (new_count / total_count)
    correction = (delta * delta) * ((old_count * new_count) / total_count)
    return merged_mean, m2 + batch_m2 + correction


def _validate_sequence(raw: np.ndarray) -> np.ndarray:
    sequence = np.asarray(raw, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != N_FEATURES or not np.isfinite(sequence).all():
        raise ValueError("expected finite raw rows with shape (T,112)")
    return sequence
