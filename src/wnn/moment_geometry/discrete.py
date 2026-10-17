"""Fitting-only level dictionaries and reset-local row/event occupancy heuristics."""

from __future__ import annotations

import numpy as np

from wnn.moment_geometry.features import SELECTED

DISCRETE_WIDTH = 40


def fit_dictionary(raw_sequences: np.ndarray) -> np.ndarray:
    """Select three frequent exact FP32 levels per field using only fitting membership."""
    vocabulary = []
    for column in SELECTED:
        levels, frequencies = [], []
        for raw in raw_sequences:
            values, counts = np.unique(raw[:, column], return_counts=True)
            levels.append(values)
            frequencies.append(counts)
        values, inverse = np.unique(np.concatenate(levels), return_inverse=True)
        counts = np.bincount(inverse, weights=np.concatenate(frequencies))
        if len(values) < 3:
            raise ValueError("need at least three observed levels for this dictionary recipe")
        vocabulary.append(values[np.argsort(counts, kind="stable")[-3:][::-1]])
    return np.asarray(vocabulary, dtype=np.float32)


def discrete_features(raw: np.ndarray, vocabulary: np.ndarray) -> np.ndarray:
    """Encode current category, row occupancy, event occupancy and dwell, without future rows.

    Four fields each have three learned levels and an explicit unknown category.
    Current categories and cumulative row occupancy each use16channels. Four
    event-clock dominant-category frequencies and four log dwell ages complete40.
    Raw-value changes define events, including transitions inside the unknown bin.
    """
    values = raw[:, SELECTED]
    if vocabulary.shape != (4, 3):
        raise ValueError("dictionary must have shape(4,3)")
    known = values[:, :, None] == vocabulary[None]
    category = np.concatenate([known, ~known.any(axis=-1, keepdims=True)], axis=-1)
    count = np.arange(1, len(raw) + 1, dtype=np.float64)[:, None, None]
    occupancy = np.cumsum(category, axis=0, dtype=np.float64) / count
    changed = np.ones(values.shape, dtype=bool)
    changed[1:] = values[1:] != values[:-1]
    event_count = np.cumsum(changed, axis=0)
    dominant_events = np.cumsum(changed & category[:, :, 0], axis=0) / event_count
    positions = np.arange(len(raw), dtype=np.int64)[:, None]
    last = np.maximum.accumulate(np.where(changed, positions, 0), axis=0)
    dwell = np.log1p(np.minimum(positions - last, 10000))
    return np.concatenate(
        [category.reshape(len(raw), 16), occupancy.reshape(len(raw), 16), dominant_events, dwell],
        axis=-1,
    ).astype(np.float32)
