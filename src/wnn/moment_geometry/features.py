"""Causal feature channels for observed support occupancy and moment geometry."""

from __future__ import annotations

import numpy as np

SELECTED = (0, 3, 52, 55)
EXTRA_WIDTH = 20


def geometry_features(
    raw: np.ndarray, beta: np.ndarray, intercept: np.ndarray, atoms: np.ndarray
) -> np.ndarray:
    """Return expanding moments, arc residual, atom occupancy and normalized current values.

    Every row includes its current observation and earlier rows only. Constants must
    come exclusively from fitting membership; all accumulators reset per sequence.
    A fixed0.05 scale floor and clipped normalized current channel protect startup.
    """
    values = raw[:, SELECTED].astype(np.float64)
    count = np.arange(1, len(values) + 1, dtype=np.float64)[:, None]
    mean = np.cumsum(values, axis=0) / count
    second = np.cumsum(values * values, axis=0) / count
    variance = np.maximum(second - mean * mean, 0.0)
    std = np.sqrt(variance)
    residual = second - beta * mean - intercept
    occupancy = np.cumsum(values == atoms, axis=0) / count
    normalized = np.clip((values - mean) / np.maximum(std, 0.05), -8, 8)
    return np.concatenate([mean, std, residual, occupancy, normalized], axis=-1).astype(np.float32)


def infer_geometry(raw_sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit arc consensus and frequently observed anchor atoms from fitting data only."""
    from wnn.moment_geometry.analysis import robust_arc

    means, variances, unique_values, counts = [], [], [], []
    for raw in raw_sequences:
        values = raw[:, SELECTED].astype(np.float64)
        means.append(values.mean(axis=0))
        variances.append(values.var(axis=0))
    means, variances = np.stack(means), np.stack(variances)
    beta, intercept = [], []
    for column in range(4):
        arc = robust_arc(means[:, column], variances[:, column])
        beta.append(arc["beta"])
        intercept.append(arc["intercept"])
        for raw in raw_sequences:
            u, n = np.unique(raw[:, SELECTED[column]], return_counts=True)
            unique_values.append(u)
            counts.append(n)
        u, inverse = np.unique(np.concatenate(unique_values), return_inverse=True)
        n = np.bincount(inverse, weights=np.concatenate(counts))
        candidates = u[np.argsort(n)[-8:]]
        center = arc["beta"] / 2
        endpoint = center + np.sqrt(max(arc["intercept"] + center**2, 0))
        anchor = candidates[np.argmin(np.abs(candidates - endpoint))]
        if column == 0:
            atoms = []
        atoms.append(float(anchor))
        unique_values.clear()
        counts.clear()
    return np.asarray(beta), np.asarray(intercept), np.asarray(atoms)
