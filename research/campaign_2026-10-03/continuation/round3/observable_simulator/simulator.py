from __future__ import annotations

import numpy as np

ORIGINS = np.arange(48, 281, 8)


def simulate(seed: int, count: int, world: str, domain: str = "nominal") -> dict[str, np.ndarray]:
    """Generate observable sensors separately from latent quote/label variables.

    Noise draws are unconditional, so nuisance domains share exact latent paths.
    Anonymous sensors have no quote units or target-derived decoder.
    """
    rng = np.random.default_rng(seed)
    steps, streams = 320, 6
    factor = rng.normal(size=count)
    specific = rng.normal(size=(count, streams))
    obs = np.zeros((count, steps, streams))
    age, masks = np.zeros_like(obs), np.zeros_like(obs, dtype=bool)
    dt = np.zeros((count, steps))
    mid = np.zeros_like(dt)
    latent = np.zeros_like(dt)
    last, elapsed = np.zeros((count, streams)), np.zeros((count, streams))
    probabilities = np.array([0.4] + [0.22] * 5)
    if domain == "stale":
        probabilities *= 0.25
    loading = 0 if world == "unidentifiable" else -1 if domain == "sign_flip" else 1
    for t in range(steps):
        delta = rng.exponential(0.3, count) + 1e-4
        dt[:, t] = delta
        decay = np.exp(-delta / 12)
        factor = decay * factor + np.sqrt(1 - decay**2) * rng.normal(size=count)
        if world == "absent":
            factor[:] = 0
        decay_i = np.exp(-delta[:, None] / 2.5)
        specific = decay_i * specific + np.sqrt(1 - decay_i**2) * rng.normal(size=(count, streams))
        price_noise = rng.normal(size=count)
        drift = np.zeros(count) if world == "absent" else 0.45 * factor + 0.22 * specific[:, 0]
        # In unidentifiable control quotes use an independent idiosyncratic shock.
        # This prevents the observed main-specific component from revealing drift.
        if world == "unidentifiable":
            drift = 0.45 * factor
        if t > 0:
            mid[:, t] = mid[:, t - 1] + drift * delta + 0.6 * np.sqrt(delta) * price_noise
        latent[:, t] = factor
        updates = rng.random((count, streams)) < probabilities
        if t == 0:
            updates[:] = True
        sensor = loading * factor[:, None] + specific + rng.normal(0, 2.8, (count, streams))
        last = np.where(updates, sensor, last)
        elapsed = np.where(updates, 0, elapsed + delta[:, None])
        obs[:, t], age[:, t], masks[:, t] = last, elapsed, updates
    return {"values": obs, "ages": age, "updates": masks, "dt": dt, "mid": mid, "factor": latent}


def stopping_labels(mid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Future first displacement barrier, with explicit forced exit at event32."""
    count = len(mid)
    labels = np.zeros((count, len(ORIGINS), 2))
    hits = np.zeros((count, len(ORIGINS)), dtype=bool)
    lags = np.full((count, len(ORIGINS)), 32)
    for k, origin in enumerate(ORIGINS):
        future = mid[:, origin + 1 : origin + 33] - mid[:, origin, None]
        touched = np.abs(future) >= 2
        hit = touched.any(axis=1)
        lag = np.where(hit, touched.argmax(axis=1), 31)
        change = future[np.arange(count), lag]
        labels[:, k] = np.c_[change - 0.2, -change - 0.2]
        hits[:, k], lags[:, k] = hit, lag + 1
    return labels, hits, lags


def scalar_label(mid: np.ndarray, origin: int) -> np.ndarray:
    """Independent scalar oracle including the declared finite-cap exit."""
    exit_mid = mid[origin + 32]
    for step in range(origin + 1, origin + 33):
        if abs(mid[step] - mid[origin]) >= 2:
            exit_mid = mid[step]
            break
    displacement = exit_mid - mid[origin]
    return np.array([displacement - 0.2, -displacement - 0.2])
