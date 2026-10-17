"""Causal feature channels for observed support occupancy and moment geometry."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

SELECTED = (0, 3, 52, 55)
EXTRA_WIDTH = 20
MAX_UNIQUE_LEVELS_PER_FIELD = 65_536


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


def infer_geometry(
    raw_sequences: np.ndarray, after_sequence: Callable[[int], None] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit arc consensus and frequently observed anchor atoms from fitting data only."""
    from wnn.moment_geometry.analysis import robust_arc

    means = np.empty((len(raw_sequences), len(SELECTED)), dtype=np.float64)
    variances = np.empty_like(means)
    level_counts: list[dict[float | None, int]] = [{} for _ in SELECTED]
    for index, raw in enumerate(raw_sequences):
        values = raw[:, SELECTED].astype(np.float64)
        means[index] = values.mean(axis=0)
        variances[index] = values.var(axis=0)
        for column, selected_column in enumerate(SELECTED):
            levels, frequencies = np.unique(raw[:, selected_column], return_counts=True)
            counts = level_counts[column]
            for level, frequency in zip(levels, frequencies, strict=True):
                key = None if np.isnan(level) else float(level)
                counts[key] = counts.get(key, 0) + int(frequency)
            if len(counts) > MAX_UNIQUE_LEVELS_PER_FIELD:
                raise ValueError("geometry level cardinality exceeds the bounded exact reducer")
        if after_sequence is not None:
            after_sequence(index)

    beta, intercept = [], []
    atoms = []
    for column in range(len(SELECTED)):
        arc = robust_arc(means[:, column], variances[:, column])
        beta.append(arc["beta"])
        intercept.append(arc["intercept"])
        counts = level_counts[column]
        ordered = sorted(counts, key=lambda value: (value is None, 0.0 if value is None else value))
        values = np.asarray(
            [np.nan if value is None else value for value in ordered], dtype=raw_sequences.dtype
        )
        frequencies = np.asarray([counts[value] for value in ordered], dtype=np.float64)
        candidates = values[np.argsort(frequencies)[-8:]]
        center = arc["beta"] / 2
        endpoint = center + np.sqrt(max(arc["intercept"] + center**2, 0))
        anchor = candidates[np.argmin(np.abs(candidates - endpoint))]
        atoms.append(float(anchor))
    return np.asarray(beta), np.asarray(intercept), np.asarray(atoms)
