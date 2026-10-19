"""Small semantic operations for the fixed three-teacher cleanfold cache."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

T3_TEACHER_ORDER = (
    "fg19_t3_h384x2_e3_s3_v1",
    "fg19_t3_h384x2_e3_s4_v1",
    "fg19_t3_h384x2_e3_s5_v1",
)
T3_HOMOGENEOUS_SCALE = np.float32(0.8125)


def fit_prediction_positions(
    prediction_groups: Sequence[int],
    target_groups: Sequence[int],
    expected_fit_groups: Sequence[int],
) -> tuple[tuple[int, ...], np.ndarray]:
    """Map fit-only prediction rows into canonical target-index order.

    Args:
        prediction_groups: Row-group identities in the prediction sidecar order.
        target_groups: Row-group identities in the canonical target-index order.
        expected_fit_groups: Exact allowed membership for this train or valid split.

    Returns:
        Canonical fit row groups and the corresponding prediction-array row indices.

    Raises:
        ValueError: If any identity list repeats, omits, or introduces a row group.
    """
    identities = {
        "prediction groups": prediction_groups,
        "target groups": target_groups,
        "expected fit groups": expected_fit_groups,
    }
    for label, groups in identities.items():
        if any(type(group) is not int for group in groups):
            raise ValueError(f"{label} must contain integer row-group identities")
    predicted = tuple(prediction_groups)
    targets = tuple(target_groups)
    expected = tuple(expected_fit_groups)
    if len(set(predicted)) != len(predicted):
        raise ValueError("fit-only prediction sidecar repeats a row group")
    if len(set(targets)) != len(targets):
        raise ValueError("canonical target index repeats a row group")
    if len(set(expected)) != len(expected):
        raise ValueError("expected fit membership repeats a row group")

    predicted_set = set(predicted)
    expected_set = set(expected)
    missing = sorted(expected_set - predicted_set)
    outer = sorted(predicted_set - expected_set)
    if missing or outer:
        raise ValueError(
            f"fit-only prediction membership differs: missing={missing[:8]}, outer={outer[:8]}"
        )

    target_fit_order = tuple(group for group in targets if group in expected_set)
    if set(target_fit_order) != expected_set:
        raise ValueError("canonical target index omits expected fitting groups")
    prediction_position = {group: index for index, group in enumerate(predicted)}
    positions = np.asarray(
        [prediction_position[group] for group in target_fit_order], dtype=np.int64
    )
    return target_fit_order, positions


def combine_cleanfold_t3_chunk(predictions: Sequence[np.ndarray]) -> np.ndarray:
    """Average raw FP32 S3/S4/S5 predictions and apply the bound scale once.

    Args:
        predictions: Three raw ``(sequences, rows, 2)`` arrays in S3/S4/S5 order.

    Returns:
        A new FP32 target chunk with no clipping or per-teacher normalization.

    Raises:
        ValueError: If source count, shape, dtype, or finite values are invalid.
    """
    if len(predictions) != 3:
        raise ValueError("cleanfold T3 requires predictions in exact S3/S4/S5 order")
    shape = predictions[0].shape
    if len(shape) != 3 or shape[-1] != 2:
        raise ValueError("cleanfold T3 predictions must have shape (sequences, rows, 2)")
    if any(value.shape != shape or value.dtype != np.float32 for value in predictions):
        raise ValueError("cleanfold T3 prediction shapes and dtypes must match FP32")
    if any(not np.isfinite(value).all() for value in predictions):
        raise ValueError("cleanfold T3 predictions must be finite")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = np.add(predictions[0], predictions[1], dtype=np.float32)
        np.add(mean, predictions[2], out=mean)
        np.divide(mean, np.float32(3.0), out=mean)
        np.multiply(mean, T3_HOMOGENEOUS_SCALE, out=mean)
    if not np.isfinite(mean).all():
        raise ValueError("cleanfold T3 FP32 mean and scale produced nonfinite values")
    return mean
