from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from wnn.data import N_FEATURES
from wnn.i1_history import (
    I1_CHANNELS,
    I1_HISTORY_FAST_HALF_LIFE,
    I1_HISTORY_SLOW_HALF_LIFE,
    I1_HISTORY_WIDTH,
    I1_RAW_START,
    I1_RAW_STOP,
    _ema_weights,
    validate_i1_history_prior,
)


class StreamingI1History:
    """Track causal i1 residuals and change energy across streaming rows."""

    def __init__(self, prior: Mapping[str, object], control: str = "none") -> None:
        if control not in ("none", "all"):
            raise ValueError("i1 history control must be none or all")
        self.control = control
        self.prior = validate_i1_history_prior(prior)
        self.scale = np.asarray(self.prior["scale"], dtype=np.float32)
        self.fast_decay, self.fast_update = _ema_weights(I1_HISTORY_FAST_HALF_LIFE)
        self.slow_decay, self.slow_update = _ema_weights(I1_HISTORY_SLOW_HALF_LIFE)
        self.reset()

    def reset(self) -> None:
        """Start a fresh sequence with no previous i1 observation."""
        self.previous: np.ndarray | None = None
        self.mean32: np.ndarray | None = None
        self.mean512: np.ndarray | None = None
        self.energy512 = np.zeros(I1_CHANNELS, dtype=np.float32)

    def update(self, row: np.ndarray) -> np.ndarray:
        """Return this row's history using only current and earlier raw values."""
        raw = np.asarray(row, dtype=np.float32)
        if raw.shape != (N_FEATURES,) or not np.isfinite(raw).all():
            raise ValueError("Expected one finite float32 raw row of width 112")
        current = raw[I1_RAW_START:I1_RAW_STOP]
        if self.previous is None:
            self.previous = current.copy()
            self.mean32 = np.zeros(I1_CHANNELS, dtype=np.float32)
            self.mean512 = np.zeros(I1_CHANNELS, dtype=np.float32)
            return np.zeros(I1_HISTORY_WIDTH, dtype=np.float32)
        if self.mean32 is None or self.mean512 is None:
            raise RuntimeError("i1 history state is incomplete")
        delta = current - self.previous
        residual32 = delta - self.mean32
        residual512 = delta - self.mean512
        self.energy512 = self.slow_decay * self.energy512 + self.slow_update * np.square(delta)
        energy = np.log1p(np.sqrt(self.energy512) / self.scale)
        self.mean32 = (self.fast_decay * self.mean32 + self.fast_update * delta) - delta
        self.mean512 = (self.slow_decay * self.mean512 + self.slow_update * delta) - delta
        self.previous = current.copy()
        result = np.concatenate((residual32, residual512, energy)).astype(np.float32)
        if self.control == "all":
            result.fill(0.0)
        return result

    def update_chunk(self, rows: np.ndarray) -> np.ndarray:
        """Process one finite chunk while preserving state for the following chunk."""
        raw = np.asarray(rows, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != N_FEATURES or not np.isfinite(raw).all():
            raise ValueError("Expected finite raw rows with shape (T,112)")
        if len(raw) == 0:
            return np.empty((0, I1_HISTORY_WIDTH), dtype=np.float32)
        return np.stack([self.update(row) for row in raw])
