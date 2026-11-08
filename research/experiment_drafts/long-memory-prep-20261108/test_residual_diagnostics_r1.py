"""Focused invariants for the read-only TL3 residual diagnostic."""

from __future__ import annotations

import numpy as np
import pytest
from residual_diagnostics_r1 import (
    BOUNDARY_RADIUS,
    CHUNK_SIZES,
    POSITION_BINS,
    _MetricPair,
    _region_masks,
    _WeightedCorrelation,
    _WeightedMSE,
)

from wnn.data import SEQUENCE_LENGTH, WARMUP


def test_position_and_chunk_regions_are_fixed_disjoint_partitions() -> None:
    """Every selected scored position belongs to one position and chunk region."""
    scored = np.ones(SEQUENCE_LENGTH, dtype=bool)
    scored[:WARMUP] = False
    excluded = [211, 257, 513, 19_999]
    scored[excluded] = False
    regions = _region_masks(scored)
    position_names = [name for _start, _end, name in POSITION_BINS]
    position_coverage = np.sum([regions[name] for name in position_names], axis=0)
    np.testing.assert_array_equal(position_coverage, scored.astype(np.int64))
    for chunk_size in CHUNK_SIZES:
        near = regions[f"chunk{chunk_size}_boundary_near"]
        interior = regions[f"chunk{chunk_size}_interior"]
        assert not np.any(near & interior)
        np.testing.assert_array_equal(near | interior, scored)
        boundary_count = (SEQUENCE_LENGTH - 1) // chunk_size
        expected_near = boundary_count * 2 * BOUNDARY_RADIUS
        excluded_near = sum(
            any(
                boundary - BOUNDARY_RADIUS <= position < boundary + BOUNDARY_RADIUS
                for boundary in range(chunk_size, SEQUENCE_LENGTH, chunk_size)
            )
            for position in excluded
        )
        assert int(near.sum()) == expected_near - excluded_near


def test_wmse_ignores_unselected_nonfinite_rows_and_rejects_selected_nonfinite() -> None:
    """Masking precedes squared-error arithmetic, as in the official evaluator."""
    targets = np.array([[1.0, -3.0], [np.nan, np.inf], [0.5, -0.25]], dtype=np.float32)
    predictions = np.array([[2.0, -1.0], [np.inf, np.nan], [0.0, 0.25]], dtype=np.float32)
    mask = np.array([True, False, True])
    accumulator = _WeightedMSE()
    accumulator.add(targets, predictions, mask)
    result = accumulator.result()
    clipped = np.clip(targets[mask].astype(np.float64), -2.0, 2.0)
    weight = np.abs(clipped)
    error = predictions[mask].astype(np.float64) - clipped
    expected = np.sum(weight * error**2, axis=0) / weight.sum(axis=0)
    np.testing.assert_allclose([result["t0"], result["t1"]], expected)
    assert result["weight_mass_by_target"] == pytest.approx(weight.sum(axis=0).tolist())
    with pytest.raises(ValueError, match="nonfinite selected"):
        accumulator.add(targets, predictions, np.array([True, True, False]))


def test_empty_region_scores_are_explicitly_undefined() -> None:
    """Empty diagnostic slices remain JSON-safe and do not fabricate zero WP."""
    result = _MetricPair.create().result()
    assert result["wp"] == {
        "t0": None,
        "t1": None,
        "wp": None,
        "selected_rows": 0,
        "metric_id": "global_wp_clip2_v1",
    }
    assert result["wmse"]["t0"] is None
    assert result["wmse"]["t1"] is None
    assert result["wmse"]["pooled"] is None
    assert result["wmse"]["weight_mass_by_target"] == [0.0, 0.0]


def test_weighted_residual_pearson_matches_direct_weighted_oracle_across_chunks() -> None:
    """Online merged weighted covariance equals an independently centered formula."""
    x = np.array([-0.8, 0.2, 1.7, -1.2, 0.5, 2.3], dtype=np.float64)
    y = np.array([-0.4, -0.1, 1.1, 0.3, 0.9, 1.7], dtype=np.float64)
    weights = np.array([0.2, 1.0, 1.5, 0.7, 2.0, 0.4], dtype=np.float64)
    mask = np.array([True, True, False, True, True, True])
    accumulator = _WeightedCorrelation()
    accumulator.add(x[:3], y[:3], weights[:3], mask[:3])
    accumulator.add(x[3:], y[3:], weights[3:], mask[3:])
    active = mask & (weights > 0)
    w = weights[active]
    x_active, y_active = x[active], y[active]
    mean_x, mean_y = np.sum(w * x_active) / w.sum(), np.sum(w * y_active) / w.sum()
    cov = np.sum(w * (x_active - mean_x) * (y_active - mean_y)) / w.sum()
    std_x = np.sqrt(np.sum(w * (x_active - mean_x) ** 2) / w.sum())
    std_y = np.sqrt(np.sum(w * (y_active - mean_y) ** 2) / w.sum())
    result = accumulator.result()
    assert result["pearson"] == pytest.approx(cov / (std_x * std_y), abs=1e-14)
    assert result["selected_rows"] == int(mask.sum())
    assert result["positive_weight_rows"] == int(active.sum())
    assert result["weight_mass"] == pytest.approx(float(w.sum()))


def test_empty_residual_slice_is_null_but_constant_observations_are_zero() -> None:
    """No observations are undefined; nonempty zero-variance remains zero."""
    empty = _WeightedCorrelation().result()
    assert empty["pearson"] is None
    assert empty["selected_rows"] == 0
    constant = _WeightedCorrelation()
    constant.add(
        np.array([1.0, 1.0]),
        np.array([2.0, 2.0]),
        np.array([1.0, 1.0]),
        np.array([True, True]),
    )
    assert constant.result()["pearson"] == 0.0
