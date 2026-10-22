"""Small synthetic invariants for the online i1-to-i0 relationship state."""

from __future__ import annotations

import numpy as np
import pytest

from wnn.online_relationship import (
    FEATURE_WIDTH,
    HORIZON,
    MODEL_INPUT_WIDTH,
    OnlineRelationshipNormalizer,
    OnlineRelationshipState,
    fit_online_relationship_normalizer,
)


def _sequence(rows: int = 128) -> np.ndarray:
    """Build matched synthetic i0/i1 channels with distinct temporal changes."""
    time = np.arange(rows, dtype=np.float32)
    raw = np.zeros((rows, 112), dtype=np.float32)
    for channel in range(52):
        frequency = np.float32(0.004 + channel * 0.0001)
        raw[:, channel] = (channel + 1) * 0.001 * time + 0.01 * np.sin(time * frequency)
        raw[:, 52 + channel] = (channel + 1) * 0.002 * time + 0.02 * np.cos(time * frequency)
    return raw


@pytest.mark.parametrize("lag", [1, 16])
@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_relationship_timing_width_and_arm_contract(arm: str, lag: int) -> None:
    """Matured i0 changes begin at row 16 and every arm emits 208 fields."""
    raw = _sequence()
    state = OnlineRelationshipState(arm, lag)
    features = state.transform(raw)

    assert FEATURE_WIDTH == 208
    assert MODEL_INPUT_WIDTH == 419
    assert features.shape == (len(raw), FEATURE_WIDTH)
    np.testing.assert_array_equal(features[:HORIZON], 0.0)
    np.testing.assert_allclose(features[HORIZON, :52], raw[16, :52] - raw[0, :52])
    np.testing.assert_allclose(features[HORIZON, 104:156], features[HORIZON, :52])
    np.testing.assert_array_equal(features[HORIZON, 156:208], 0.0)
    if arm == "A":
        np.testing.assert_array_equal(features[:, 156:208], 0.0)


@pytest.mark.parametrize("lag", [1, 16])
def test_prefix_future_invariance_and_sequence_reset(lag: int) -> None:
    """Later rows do not alter a prefix, and reset reproduces the same sequence."""
    raw = _sequence()
    altered = raw.copy()
    altered[40:, :104] += np.float32(3.0)
    baseline = OnlineRelationshipState("C", lag).transform(raw)
    changed = OnlineRelationshipState("C", lag).transform(altered)
    np.testing.assert_array_equal(baseline[:40], changed[:40])

    state = OnlineRelationshipState("C", lag)
    streamed = state.transform(raw[:23])
    state.reset()
    np.testing.assert_array_equal(state.transform(raw), baseline)
    assert streamed.shape == (23, FEATURE_WIDTH)


def test_a_uses_the_old_mean_for_error_then_the_updated_mean_for_forecast() -> None:
    """The mean-only arm's row-17 error precedes its update and forecast follows it."""
    raw = _sequence(24)
    values = OnlineRelationshipState("A", 1).transform(raw)
    decay = np.float32(2.0 ** (-1.0 / 512.0))
    update = np.float32(1.0 - float(decay))
    y16 = raw[16, :52] - raw[0, :52]
    y17 = raw[17, :52] - raw[1, :52]
    mean16 = update * y16
    np.testing.assert_allclose(values[16, 52:104], mean16, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(values[17, 104:156], y17 - mean16, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(values[:, 156:208], 0.0)


def test_lag_16_uses_only_real_regressors_after_row_31() -> None:
    """Lag-16 maturity keeps zero-u pairs through row 31, then sees real u values."""
    raw = _sequence(40)
    lag_one = OnlineRelationshipState("C", 1)
    lag_sixteen = OnlineRelationshipState("C", 16)
    lag_one.transform(raw[:18])
    lag_sixteen.transform(raw[:32])
    assert np.any(lag_one.covariance != 0.0)
    np.testing.assert_array_equal(lag_sixteen.covariance, 0.0)
    lag_sixteen.transform(raw[32:33])
    assert np.any(lag_sixteen.covariance != 0.0)


@pytest.mark.parametrize("lag", [1, 16])
@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_chunk_restore_matches_uninterrupted_scan(arm: str, lag: int) -> None:
    """Restoring the complete ring and coefficient state preserves exact outputs."""
    raw = _sequence(83)
    normalizer = fit_online_relationship_normalizer([raw], lag)
    complete = OnlineRelationshipState(arm, lag, normalizer).transform(raw)

    first_state = OnlineRelationshipState(arm, lag, normalizer)
    first = first_state.transform(raw[:37])
    restored = OnlineRelationshipState(arm, lag, normalizer)
    restored.load_state_dict(first_state.state_dict())
    second = restored.transform(raw[37:])
    np.testing.assert_array_equal(np.concatenate((first, second)), complete)


def test_arm_b_freezes_after_row_98_and_constant_inputs_stay_finite() -> None:
    """B sees 83 matured updates; constant channels remain stable under scaling."""
    raw = _sequence(120)
    state = OnlineRelationshipState("B", 1)
    state.transform(raw[:99])
    frozen = state.state_dict()
    mx = np.zeros(52, dtype=np.float32)
    my = np.zeros(52, dtype=np.float32)
    variance = np.ones(52, dtype=np.float32)
    covariance = np.zeros(52, dtype=np.float32)
    decay = np.float32(2.0 ** (-1.0 / 512.0))
    update = np.float32(1.0 - float(decay))
    for row_index in range(16, 99):
        x = (
            np.zeros(52, dtype=np.float32)
            if row_index == 16
            else raw[row_index - 16, 52:104] - raw[row_index - 17, 52:104]
        )
        y = raw[row_index, :52] - raw[row_index - 16, :52]
        dx, dy = x - mx, y - my
        dx_squared = dx * dx
        dx_dy = dx * dy
        mx += update * dx
        my += update * dy
        variance = decay * (variance + update * dx_squared)
        covariance = decay * (covariance + update * dx_dy)
    for name, expected in (
        ("mx", mx),
        ("my", my),
        ("variance", variance),
        ("covariance", covariance),
    ):
        np.testing.assert_allclose(frozen[name], expected, rtol=1e-6, atol=1e-6)
    state.transform(raw[99:])
    for name in ("mx", "my", "variance", "covariance"):
        np.testing.assert_array_equal(state.state_dict()[name], frozen[name])

    constant = np.ones((40, 112), dtype=np.float32)
    normalizer = fit_online_relationship_normalizer([constant], 16)
    assert np.all(normalizer.u_scale >= np.float32(0.001))
    assert np.all(normalizer.v_scale >= np.float32(0.001))
    values = OnlineRelationshipState("C", 16, normalizer).transform(constant)
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values, 0.0)


def test_standardized_differences_and_outputs_clip_and_normalizer_is_lag_bound() -> None:
    """Input-difference scaling clips at eight and is tied to the selected lag."""
    mean = np.zeros(52, dtype=np.float32)
    scale = np.full(52, 0.001, dtype=np.float32)
    normalizer = OnlineRelationshipNormalizer(mean, scale, mean, scale, 1)
    features = OnlineRelationshipState("C", 1, normalizer).transform(_sequence(24))
    assert np.max(np.abs(features)) <= 8.0
    with pytest.raises(ValueError, match="lag must match"):
        OnlineRelationshipState("C", 16, normalizer)


@pytest.mark.parametrize("lag", [1, 16])
def test_fitting_normalizer_uses_only_available_raw_differences(lag: int) -> None:
    """Separate u/v moments exclude each difference's unavailable prefix."""
    raw = _sequence(90)
    normalizer = fit_online_relationship_normalizer([raw], lag)
    expected_u = raw[lag:, 52:104] - raw[:-lag, 52:104]
    expected_v = raw[HORIZON:, :52] - raw[:-HORIZON, :52]
    np.testing.assert_allclose(normalizer.u_mean, expected_u.mean(axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(normalizer.v_mean, expected_v.mean(axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        normalizer.u_scale, np.maximum(expected_u.std(axis=0), 0.001), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        normalizer.v_scale, np.maximum(expected_v.std(axis=0), 0.001), rtol=1e-6, atol=1e-6
    )
