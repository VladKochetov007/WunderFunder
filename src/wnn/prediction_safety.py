from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

PREDICTION_CACHE_SCHEMA = "wunder.full-prediction-cache.v1"


def validate_full_teacher_cache(
    predictions: np.ndarray,
    sidecar: Mapping[str, object],
    *,
    split: str,
    row_groups: Sequence[int],
    prediction_name: str,
) -> np.ndarray:
    """Require a teacher cache and sidecar to cover a split in canonical row order.

    Args:
        predictions: Memory-mapped or in-memory ``(groups, 20000, 2)`` values.
        sidecar: Metadata written by the prediction exporter.
        split: Dataset split named by the consumer.
        row_groups: Canonical row-group sequence for that split.
        prediction_name: Expected basename of the prediction array.

    Returns:
        The original array as a NumPy view.

    Raises:
        ValueError: If cache shape, dtype, split, or full membership differs.
    """
    array = np.asarray(predictions)
    expected_groups = [int(group) for group in row_groups]
    expected_shape = (len(expected_groups), 20_000, 2)
    if array.shape != expected_shape:
        raise ValueError(f"teacher cache shape {array.shape} does not match {expected_shape}")
    if array.dtype != np.float32:
        raise ValueError("teacher cache dtype must be float32")
    if len(set(expected_groups)) != len(expected_groups):
        raise ValueError("canonical row groups contain duplicates")
    expected = {
        "schema": PREDICTION_CACHE_SCHEMA,
        "split": split,
        "prediction_name": prediction_name,
        "complete": True,
        "row_groups": expected_groups,
        "shape": list(expected_shape),
        "dtype": array.dtype.name,
    }
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise ValueError(f"teacher cache sidecar {key} does not match the requested split")
    return array


def validate_teacher_rows(predictions: np.ndarray, row_positions: Sequence[int]) -> np.ndarray:
    """Read requested teacher rows after checking shape and finite values.

    Args:
        predictions: Cache array with shape ``(groups, 20000, 2)``.
        row_positions: Integer row offsets to read.

    Returns:
        Requested rows as float32.

    Raises:
        ValueError: If positions are invalid or any requested value is nonfinite.
    """
    array = np.asarray(predictions)
    positions = np.asarray(row_positions, dtype=np.int64)
    if array.ndim != 3 or array.shape[-2:] != (20_000, 2):
        raise ValueError("teacher cache must have shape (groups, 20000, 2)")
    if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= array.shape[0]):
        raise ValueError("teacher row positions are malformed or out of range")
    rows = np.asarray(array[positions], dtype=np.float32)
    if not np.isfinite(rows).all():
        raise ValueError("teacher cache contains nonfinite values in requested training rows")
    return rows


def prepare_scored_prediction_arrays(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    *,
    clip: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate scored values and zero unscored values before moment calculations.

    Args:
        targets: Target values with shape ``(timesteps, 2)``.
        predictions: Predictions aligned with ``targets``.
        mask: Scored rows, with shape ``(timesteps,)`` or ``(timesteps, 2)``.
        clip: Symmetric clipping threshold.

    Returns:
        Clipped safe targets, clipped safe predictions, target-magnitude weights,
        and the broadcast scoring mask.

    Raises:
        ValueError: If shapes differ or any selected value is nonfinite.
    """
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if y.ndim != 2 or y.shape[1] != 2 or p.shape != y.shape:
        raise ValueError("targets and predictions must have matching shape (timesteps, 2)")
    if selected.shape == (y.shape[0],):
        selected = np.broadcast_to(selected[:, None], y.shape)
    elif selected.shape != y.shape:
        raise ValueError("scoring mask must have shape (timesteps,) or (timesteps, 2)")
    if not np.isfinite(y[selected]).all() or not np.isfinite(p[selected]).all():
        raise ValueError("selected targets and predictions must be finite")
    safe_y = np.clip(np.where(selected, y, 0.0), -clip, clip)
    safe_p = np.clip(np.where(selected, p, 0.0), -clip, clip)
    weights = np.abs(safe_y) * selected
    return safe_y, safe_p, weights, selected
