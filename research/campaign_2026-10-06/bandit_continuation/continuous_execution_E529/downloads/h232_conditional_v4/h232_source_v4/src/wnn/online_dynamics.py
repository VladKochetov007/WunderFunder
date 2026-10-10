from __future__ import annotations

import numpy as np
from attrs import define, field, frozen
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@frozen
class DynamicsObservation:
    """Available-now features, without access to task labels or future rows."""

    innovation: FloatArray
    next_forecast: FloatArray


@define
class OnlineDynamics:
    """Reference recursive ridge dynamics, reset independently for each sequence.

    At row t, the previous forecast is checked against the newly observed input.
    Only then is the completed transition used to update and forecast row t+1.
    The caller owns an offline-fitted input projection and must call reset when
    the sequence changes. This is a numerical reference, not a serving model.
    """

    initial_weights: FloatArray
    ridge: float = 1.0
    forgetting: float = 1.0
    _weights: FloatArray = field(init=False, repr=False)
    _precision_inverse: FloatArray = field(init=False, repr=False)
    _previous: FloatArray | None = field(init=False, default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        initial = np.array(self.initial_weights, dtype=np.float64, copy=True)
        if initial.ndim != 2 or initial.shape[0] != initial.shape[1] or not initial.size:
            raise ValueError("initial dynamics must be a nonempty square matrix")
        if not np.isfinite(initial).all() or not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("finite weights and positive finite ridge are required")
        if not 0 < self.forgetting <= 1:
            raise ValueError("forgetting must lie in (0, 1]")
        initial.setflags(write=False)
        self.initial_weights = initial
        self.reset()

    def reset(self) -> None:
        """Restore all adaptive state, including uncertainty and previous input."""
        self._weights = self.initial_weights.copy()
        self._precision_inverse = np.eye(len(self.initial_weights)) / self.ridge
        self._previous = None

    def observe(self, features: FloatArray) -> DynamicsObservation:
        """Consume one observable row and return its innovation and next forecast."""
        current = np.array(features, dtype=np.float64, copy=True)
        if current.shape != (len(self.initial_weights),) or not np.isfinite(current).all():
            raise ValueError("one finite projected feature vector is required")
        innovation = np.zeros_like(current)
        if self._previous is not None:
            innovation = current - self._weights @ self._previous
            projected = self._precision_inverse @ self._previous
            denominator = self.forgetting + self._previous @ projected
            if not np.isfinite(denominator) or denominator <= 0:
                raise FloatingPointError("invalid recursive dynamics denominator")
            weights = self._weights + np.outer(innovation, projected / denominator)
            covariance = (
                self._precision_inverse - np.outer(projected, projected) / denominator
            ) / self.forgetting
            covariance = (covariance + covariance.T) * 0.5
            if not np.isfinite(weights).all() or not np.isfinite(covariance).all():
                raise FloatingPointError("nonfinite recursive dynamics state")
            self._weights = weights
            self._precision_inverse = covariance
        forecast = self._weights @ current
        if not np.isfinite(forecast).all():
            raise FloatingPointError("nonfinite next-input forecast")
        self._previous = current
        return DynamicsObservation(innovation, forecast)
