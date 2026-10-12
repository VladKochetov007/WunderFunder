"""Synthetic checks for bounded two-target blend optimization primitives."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from wnn.blend_optimization import (
    analytic_shared_initializer,
    clip2_wp_surrogate_and_gradient,
    fit_shared_clip2_weights,
    official_clip2_wp,
    ridge_simplex_initializer,
    unclipped_correlation_and_gradient,
    weighted_prediction_moments,
)
from wnn.gmetric import global_wp


def _central_difference(
    function: Callable[[np.ndarray], float], point: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    result = np.empty_like(point, dtype=np.float64)
    for index in range(point.size):
        delta = np.zeros_like(point, dtype=np.float64)
        delta[index] = step
        result[index] = (function(point + delta) - function(point - delta)) / (2.0 * step)
    return result


def test_unclipped_moments_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(801)
    target = rng.uniform(-1.0, 1.0, 40).astype(np.float32)
    predictions = np.column_stack(
        (0.45 * target + rng.normal(0.0, 0.16, 40), -0.2 * target + rng.normal(0, 0.22, 40))
    ).astype(np.float32)
    weights = np.abs(np.clip(target, -2.0, 2.0)).astype(np.float64)
    moments = weighted_prediction_moments(target, predictions, weights)
    point = np.array([0.4, 0.6], dtype=np.float64)

    value, gradient = unclipped_correlation_and_gradient(point, moments)
    numerical = _central_difference(
        lambda coefficients: unclipped_correlation_and_gradient(coefficients, moments)[0], point
    )
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-7)

    mixture = (predictions.astype(np.float64) @ point).astype(np.float32)
    target_pair = np.column_stack((target, target))
    mixture_pair = np.column_stack((mixture, mixture))
    expected = float(global_wp(target_pair, mixture_pair, np.ones(target.size, dtype=bool))["wp"])
    assert value == pytest.approx(expected, abs=2e-7)


def test_clip2_surrogate_gradient_and_official_candidate_score() -> None:
    rng = np.random.default_rng(1204)
    rows = 64
    targets = rng.normal(0.0, 1.5, (rows, 2)).astype(np.float32)
    predictions = np.stack(
        (
            2.5 * targets + rng.normal(0.0, 0.6, targets.shape),
            0.75 * targets + rng.normal(0.0, 0.45, targets.shape),
            -0.3 * targets + rng.normal(0.0, 0.5, targets.shape),
        ),
        axis=1,
    ).astype(np.float32)
    mask = rng.random(rows) > 0.15
    targets[~mask] = np.nan
    predictions[~mask] = np.inf
    coefficients = np.array([0.35, 0.5, 0.15], dtype=np.float64)

    _, gradient = clip2_wp_surrogate_and_gradient(targets, predictions, mask, coefficients)
    raw_mixture = np.einsum("nmk,m->nk", predictions, coefficients, dtype=np.float64)
    reference_prediction = raw_mixture.astype(np.float32)
    reference = global_wp(targets, reference_prediction, mask)
    assert official_clip2_wp(targets, predictions, mask, coefficients) == pytest.approx(
        float(reference["wp"]), abs=0.0
    )

    numerical = _central_difference(
        lambda value: clip2_wp_surrogate_and_gradient(targets, predictions, mask, value)[0],
        coefficients,
    )
    np.testing.assert_allclose(gradient, numerical, atol=2e-6, rtol=2e-5)


def test_float32_ulp_rounding_can_change_the_official_candidate_score() -> None:
    unit = float(np.spacing(np.float32(1.0)))
    targets = np.repeat(np.array([[0.5], [1.0]], dtype=np.float32), 2, axis=1)
    predictions = np.empty((2, 2, 2), dtype=np.float32)
    predictions[:, 0, :] = np.float32(1.0)
    predictions[:, 1, 0] = np.array([1.0 + unit, 1.0 + 2.0 * unit], dtype=np.float32)
    predictions[:, 1, 1] = predictions[:, 1, 0]
    mask = np.ones(2, dtype=bool)
    coefficients = np.array([0.75, 0.25], dtype=np.float64)

    surrogate, _ = clip2_wp_surrogate_and_gradient(targets, predictions, mask, coefficients)
    official = official_clip2_wp(targets, predictions, mask, coefficients)
    assert surrogate == pytest.approx(1.0, abs=1e-12)
    assert official == 0.0


def test_official_score_uses_selected_rows_and_rejects_selected_nonfinite() -> None:
    targets = np.array([[0.0, 0.0], [0.5, -0.5], [1.0, 1.0]], dtype=np.float32)
    predictions = np.array([[[1.0, -1.0]], [[0.2, -0.2]], [[0.8, 0.8]]], dtype=np.float32)
    mask = np.array([True, True, False])
    targets[2] = np.nan
    predictions[2] = np.inf
    coefficients = np.array([1.0], dtype=np.float64)
    expected = global_wp(targets, predictions[:, 0, :], mask)
    assert official_clip2_wp(targets, predictions, mask, coefficients) == float(expected["wp"])
    with pytest.raises(ValueError, match="no selected rows"):
        official_clip2_wp(targets, predictions, np.zeros_like(mask), coefficients)

    targets[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite selected"):
        official_clip2_wp(targets, predictions, mask, coefficients)


def test_clipping_can_reverse_an_unclipped_initializer_preference() -> None:
    target_1d = np.array(
        [-1.9, -1.6, -1.3, -0.9, -0.45, -0.15, 0.2, 0.55, 0.9, 1.25, 1.6, 1.9],
        dtype=np.float32,
    )
    noise = np.array(
        [0.14, -0.12, 0.10, -0.16, 0.11, -0.08, 0.12, -0.10, 0.09, -0.13, 0.08, -0.11],
        dtype=np.float32,
    )
    target = np.repeat(target_1d[:, None], 2, axis=1)
    pool_1d = np.column_stack((10.0 * target_1d, target_1d + noise))
    predictions = np.repeat(pool_1d[:, :, None], 2, axis=2).astype(np.float32)
    mask = np.ones(target.shape[0], dtype=bool)
    moments = weighted_prediction_moments(target_1d, pool_1d, np.abs(np.clip(target_1d, -2.0, 2.0)))
    initializer = ridge_simplex_initializer(moments, ridge=1e-10)
    assert initializer[0] > 1.0 - 1e-5

    model_zero = clip2_wp_surrogate_and_gradient(target, predictions, mask, np.array([1.0, 0.0]))[0]
    model_one = clip2_wp_surrogate_and_gradient(target, predictions, mask, np.array([0.0, 1.0]))[0]
    assert model_one > model_zero + 0.01


def test_near_singular_moments_use_declared_ridge_without_inverse() -> None:
    target = np.linspace(-1.0, 1.0, 30, dtype=np.float64)
    base = target + 0.05 * np.sin(np.arange(target.size))
    predictions = np.column_stack((base, base + 1e-9 * np.cos(np.arange(target.size))))
    weights = np.abs(target)
    moments = weighted_prediction_moments(target, predictions, weights)

    result = ridge_simplex_initializer(moments, ridge=1e-8)
    assert np.isfinite(result).all()
    assert np.all(result >= 0.0)
    assert result.sum() == pytest.approx(1.0, abs=1e-14)
    singular = weighted_prediction_moments(target, np.column_stack((base, base)), weights)
    with pytest.raises(ValueError, match="singular covariance"):
        ridge_simplex_initializer(singular, ridge=0.0)


def test_bounded_slsqp_starts_constraints_determinism_and_fixed_shrinkage() -> None:
    rng = np.random.default_rng(512)
    targets = rng.normal(size=(48, 2)).astype(np.float32)
    predictions = np.stack(
        (
            targets + rng.normal(0.0, 0.8, targets.shape),
            0.7 * targets + rng.normal(0.0, 0.6, targets.shape),
            -0.15 * targets + rng.normal(0.0, 0.9, targets.shape),
        ),
        axis=1,
    ).astype(np.float32)
    mask = np.ones(len(targets), dtype=bool)
    current = np.array([0.6, 0.3, 0.1], dtype=np.float64)

    first = fit_shared_clip2_weights(
        targets, predictions, mask, current, ridge=1e-5, shrink_to_equal=0.25
    )
    second = fit_shared_clip2_weights(
        targets, predictions, mask, current, ridge=1e-5, shrink_to_equal=0.25
    )
    np.testing.assert_allclose(first.weights, second.weights, atol=1e-12, rtol=0.0)
    assert len(first.attempts) == 3
    assert [attempt.start_name for attempt in first.attempts] == ["equal", "current", "analytic"]
    assert all(0 <= attempt.iterations <= 50 for attempt in first.attempts)
    assert np.all(first.weights >= 0.0)
    assert first.weights.sum() == pytest.approx(1.0, abs=1e-14)
    equal = np.full(3, 1.0 / 3.0)
    expected = 0.75 * first.unshrunk_weights + 0.25 * equal
    np.testing.assert_allclose(first.weights, expected, atol=1e-14, rtol=0.0)
    assert np.isfinite(first.official_score)
    assert np.isfinite(first.surrogate_score)

    analytic = analytic_shared_initializer(targets, predictions, mask, ridge=1e-5)
    solver_by_name = {attempt.start_name: attempt for attempt in first.attempts}
    ranked: list[tuple[str, float]] = []
    for name, start in (("equal", equal), ("current", current), ("analytic", analytic)):
        attempt = solver_by_name[name]
        np.testing.assert_allclose(attempt.start_weights, start, atol=1e-15, rtol=0.0)
        start_score = official_clip2_wp(targets, predictions, mask, attempt.start_weights)
        assert attempt.start_official_score == start_score
        assert (
            attempt.start_surrogate_score
            == clip2_wp_surrogate_and_gradient(targets, predictions, mask, attempt.start_weights)[0]
        )
        ranked.append((f"{name}:start", start_score))
        attempt_score = official_clip2_wp(targets, predictions, mask, attempt.weights)
        assert attempt.official_score == attempt_score
        ranked.append((f"{name}:slsqp", attempt_score))
    selected_name, selected_score = max(ranked, key=lambda item: item[1])
    assert first.selected_candidate == selected_name
    assert first.unshrunk_official_score == selected_score
    assert first.official_score == official_clip2_wp(targets, predictions, mask, first.weights)
    assert (
        first.surrogate_score
        == clip2_wp_surrogate_and_gradient(targets, predictions, mask, first.weights)[0]
    )


def test_shared_analytic_initializer_is_feasible_and_nonnegative() -> None:
    rng = np.random.default_rng(74)
    targets = rng.normal(size=(24, 2)).astype(np.float32)
    predictions = rng.normal(size=(24, 4, 2)).astype(np.float32)
    weights = analytic_shared_initializer(targets, predictions, np.ones(24, dtype=bool), ridge=1e-4)
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-14)
