from __future__ import annotations

import time

import numpy as np


class Ridge:
    """Fit-only standardized ridge; one fixed alpha, no evaluation calibration."""

    def __init__(self) -> None:
        self.mean = np.empty(0)
        self.scale = np.empty(0)
        self.beta = np.empty((0, 2))
        self.intercept = np.empty(2)
        self.fit_seconds = 0.0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        start = time.monotonic()
        self.mean = features.mean(axis=0)
        self.scale = np.maximum(features.std(axis=0), 1e-6)
        x = (features - self.mean) / self.scale
        self.intercept = labels.mean(axis=0)
        self.beta = np.linalg.solve(
            x.T @ x + 10 * np.eye(x.shape[1]), x.T @ (labels - self.intercept)
        )
        self.fit_seconds = time.monotonic() - start
        if self.fit_seconds > 120:
            raise TimeoutError("Fit exceeded frozen 120s allowance")

    def predict(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.scale) @ self.beta + self.intercept


def features(observed: dict[str, np.ndarray], arm: str) -> np.ndarray:
    """Read only four observable arrays; no latent or future-label arguments used.

    EMA changes only at observed updates, using observable elapsed time since the
    previous update. Pooling downweights stale peers. Reset uses identical columns
    with EMA=current held sensor, preventing history-only capacity confounding.
    """
    values, ages = observed["values"], observed["ages"]
    updates, dt = observed["updates"], observed["dt"]
    row = np.concatenate([values, np.log1p(ages), updates.astype(float), dt[:, :, None]], axis=2)
    if arm == "row":
        return row
    if arm not in ("history", "reset"):
        raise ValueError(arm)
    count, steps, streams = values.shape
    ema, elapsed = values[:, 0].copy(), np.zeros((count, streams))
    output = np.zeros((count, steps, row.shape[2] + 9))
    for t in range(steps):
        if t > 0:
            elapsed += dt[:, t, None]
            alpha = 1 - np.exp(-elapsed / 4)
            ema = np.where(updates[:, t], ema + alpha * (values[:, t] - ema), ema)
            elapsed = np.where(updates[:, t], 0, elapsed)
        active = ema if arm == "history" else values[:, t]
        weights = np.exp(-ages[:, t, 1:] / 4)
        denominator = weights.sum(axis=1)
        pool = (weights * active[:, 1:]).sum(axis=1) / denominator
        variance = (weights * (active[:, 1:] - pool[:, None]) ** 2).sum(axis=1) / denominator
        extra = np.c_[active, pool, variance, active[:, 0] - pool]
        output[:, t] = np.c_[row[:, t], extra]
    return output
