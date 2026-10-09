from __future__ import annotations

import numpy as np
import pytest

from wnn.online_dynamics import OnlineDynamics


@pytest.mark.parametrize("forgetting", [1.0, 0.98])
def test_each_prefix_matches_direct_weighted_ridge(forgetting: float) -> None:
    rng = np.random.default_rng(240)
    rows = rng.normal(size=(40, 5))
    initial = rng.normal(scale=0.1, size=(5, 5))
    ridge = 2.0
    model = OnlineDynamics(initial, ridge, forgetting)
    prior_weights = initial.copy()
    for index, row in enumerate(rows):
        result = model.observe(row)
        expected_innovation = np.zeros(5) if index == 0 else row - prior_weights @ rows[index - 1]
        np.testing.assert_allclose(result.innovation, expected_innovation, atol=2e-12)
        weights = forgetting ** np.arange(index - 1, -1, -1)
        design, observed = rows[:index], rows[1 : index + 1]
        gram = design.T @ (weights[:, None] * design) + ridge * forgetting**index * np.eye(5)
        cross = observed.T @ (weights[:, None] * design) + ridge * forgetting**index * initial
        prior_weights = np.linalg.solve(gram, cross.T).T
        np.testing.assert_allclose(result.next_forecast, prior_weights @ row, atol=2e-12)


def test_future_changes_cannot_change_prefix_and_reset_replays() -> None:
    rng = np.random.default_rng(241)
    rows = rng.normal(size=(100, 8))
    model = OnlineDynamics(np.zeros((8, 8)))
    first = [model.observe(row) for row in rows]
    rows[40:] *= -100
    model.reset()
    second = [model.observe(row) for row in rows]
    for left, right in zip(first[:40], second[:40], strict=True):
        np.testing.assert_array_equal(left.innovation, right.innovation)
        np.testing.assert_array_equal(left.next_forecast, right.next_forecast)
    model.reset()
    independent = OnlineDynamics(np.zeros((8, 8)))
    for row in rows:
        left, right = model.observe(row), independent.observe(row)
        np.testing.assert_array_equal(left.innovation, right.innovation)
        np.testing.assert_array_equal(left.next_forecast, right.next_forecast)


def test_full_length_constant_sequence_stays_finite() -> None:
    model = OnlineDynamics(np.zeros((16, 16)))
    row = np.ones(16)
    for _ in range(20_000):
        result = model.observe(row)
    assert np.isfinite(result.innovation).all()
    np.testing.assert_allclose(result.next_forecast, row, atol=1e-5)


def test_rejects_invalid_input_and_does_not_alias_callers_array() -> None:
    model = OnlineDynamics(np.zeros((2, 2)))
    first = np.array([1.0, 0.0])
    model.observe(first)
    first[:] = 50
    with pytest.raises(ValueError, match="finite"):
        model.observe(np.array([np.nan, 1.0]))
    result = model.observe(np.array([0.0, 1.0]))
    np.testing.assert_allclose(result.next_forecast, [0.0, 0.0])
