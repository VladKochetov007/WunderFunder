"""Fitting-only level dictionaries and reset-local row/event occupancy heuristics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wnn.moment_geometry.features import SELECTED

DISCRETE_WIDTH = 40
MAX_UNIQUE_LEVELS_PER_FIELD = 65_536


def fit_dictionary(
    raw_sequences: np.ndarray, after_sequence: Callable[[int], None] | None = None
) -> np.ndarray:
    """Select three frequent exact FP32 levels per field using only fitting membership."""
    level_counts: list[dict[float | None, int]] = [{} for _ in SELECTED]
    for index, raw in enumerate(raw_sequences):
        for position, column in enumerate(SELECTED):
            values, counts = np.unique(raw[:, column], return_counts=True)
            aggregate = level_counts[position]
            for value, count in zip(values, counts, strict=True):
                key = None if np.isnan(value) else float(value)
                aggregate[key] = aggregate.get(key, 0) + int(count)
            if len(aggregate) > MAX_UNIQUE_LEVELS_PER_FIELD:
                raise ValueError("dictionary level cardinality exceeds the bounded exact reducer")
        if after_sequence is not None:
            after_sequence(index)

    vocabulary = []
    for aggregate in level_counts:
        ordered = sorted(
            aggregate, key=lambda value: (value is None, 0.0 if value is None else value)
        )
        values = np.asarray([np.nan if value is None else value for value in ordered])
        counts = np.asarray([aggregate[value] for value in ordered], dtype=np.float64)
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
