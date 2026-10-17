"""Small Python-only references for the packaged native geometry streamer."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

_SELECTED = np.asarray((0, 3, 52, 55), dtype=np.intp)
_DWELL_CAP = 10_000


class GeometryReferenceStream:
    """Incremental FP64 reference for the 60 causal geometry/discrete features."""

    def __init__(self, prior: Mapping[str, object]) -> None:
        self.beta = np.asarray(prior["beta"], dtype=np.float32).astype(np.float64)
        self.intercept = np.asarray(prior["intercept"], dtype=np.float32).astype(np.float64)
        self.atoms = np.asarray(prior["atoms"], dtype=np.float32).astype(np.float64)
        self.vocabulary = np.asarray(prior["vocabulary"], dtype=np.float32).astype(np.float64)
        self.out = np.empty(60, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Clear all causal accumulators at a sequence boundary."""
        self.position = 0
        self.count = 0
        self.previous: np.ndarray | None = None
        self.total = np.zeros(4, dtype=np.float64)
        self.total_squared = np.zeros(4, dtype=np.float64)
        self.atom_count = np.zeros(4, dtype=np.float64)
        self.occupancy = np.zeros((4, 4), dtype=np.float64)
        self.event_count = np.zeros(4, dtype=np.int64)
        self.dominant_count = np.zeros(4, dtype=np.int64)
        self.last_change = np.zeros(4, dtype=np.int64)
        self.category = np.empty((4, 4), dtype=np.bool_)

    def update(self, row: np.ndarray) -> np.ndarray:
        """Advance once and return the current 60-feature reference vector."""
        values = row[_SELECTED].astype(np.float64)
        changed = np.ones(4, dtype=np.bool_) if self.previous is None else values != self.previous
        self.total += values
        self.total_squared += values * values
        self.count += 1
        mean = self.total / self.count
        second = self.total_squared / self.count
        std = np.sqrt(np.maximum(second - mean * mean, 0.0))
        self.atom_count += values == self.atoms
        known = values[:, None] == self.vocabulary
        self.category[:, :3] = known
        self.category[:, 3] = ~known.any(axis=1)
        self.occupancy += self.category
        self.event_count += changed
        self.dominant_count += changed & self.category[:, 0]
        self.last_change[changed] = self.position

        out = self.out
        out[:4] = mean
        out[4:8] = std
        out[8:12] = second - self.beta * mean - self.intercept
        out[12:16] = self.atom_count / self.count
        out[16:20] = np.clip((values - mean) / np.maximum(std, 0.05), -8.0, 8.0)
        np.copyto(out[20:36], self.category.reshape(16), casting="unsafe")
        out[36:52] = (self.occupancy / self.count).reshape(16)
        out[52:56] = self.dominant_count / np.maximum(self.event_count, 1)
        out[56:60] = np.log1p(np.minimum(self.position - self.last_change, _DWELL_CAP))
        self.previous = values.copy()
        self.position += 1
        return out
