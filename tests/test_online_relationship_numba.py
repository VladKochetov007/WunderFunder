"""Exact parity checks for the optional serial Numba relationship scan."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wnn.online_relationship import (
    FEATURE_WIDTH,
    OnlineRelationshipState,
    fit_online_relationship_normalizer,
)


def _sequence(rows: int = 144) -> np.ndarray:
    """Build finite channels with distinct trends and nonlinear changes."""
    time = np.arange(rows, dtype=np.float32)
    raw = np.zeros((rows, 112), dtype=np.float32)
    for channel in range(52):
        frequency = np.float32(0.013 + channel * 0.0002)
        raw[:, channel] = (channel + 1) * 0.003 * time + 0.02 * np.sin(time * frequency)
        raw[:, 52 + channel] = (channel + 1) * 0.001 * time + 0.03 * np.cos(time * frequency)
    return raw


def _clip(value: np.float32) -> np.float32:
    """Apply the feature limit with NumPy-compatible NaN behavior."""
    if value > np.float32(8.0):
        return np.float32(8.0)
    if value < np.float32(-8.0):
        return np.float32(-8.0)
    return value


def _scalar_oracle(
    raw: np.ndarray, arm: str, lag: int, normalizer: object
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    """Evaluate every channel independently using the written FP32 recurrence."""
    decay = np.float32(math.exp2(-1.0 / 512.0))
    update = np.float32(1.0 - float(decay))
    epsilon = np.float32(0.1)
    i1_history = np.zeros((lag, 52), dtype=np.float32)
    i0_history = np.zeros((16, 52), dtype=np.float32)
    u_history = np.zeros((16, 52), dtype=np.float32)
    mean_x = np.zeros(52, dtype=np.float32)
    mean_y = np.zeros(52, dtype=np.float32)
    variance = np.ones(52, dtype=np.float32)
    covariance = np.zeros(52, dtype=np.float32)
    output = np.zeros((len(raw), FEATURE_WIDTH), dtype=np.float32)
    has_normalizer = normalizer is not None

    for row_index, row in enumerate(raw):
        maturity_slot = row_index % 16
        lag_slot = row_index % lag
        for channel in range(52):
            current_i0 = np.float32(row[channel])
            current_i1 = np.float32(row[52 + channel])
            if row_index >= lag:
                raw_u = np.float32(current_i1 - i1_history[lag_slot, channel])
                if has_normalizer:
                    current_u = np.float32(
                        (raw_u - normalizer.u_mean[channel]) / normalizer.u_scale[channel]
                    )
                else:
                    current_u = raw_u
                current_u = _clip(current_u)
            else:
                current_u = np.float32(0.0)

            if row_index >= 16:
                matured_v = np.float32(current_i0 - i0_history[maturity_slot, channel])
                if has_normalizer:
                    matured_v = np.float32(
                        (matured_v - normalizer.v_mean[channel]) / normalizer.v_scale[channel]
                    )
                matured_v = _clip(matured_v)
                matured_u = np.float32(u_history[maturity_slot, channel])
                old_slope = (
                    np.float32(0.0)
                    if arm == "A"
                    else np.float32(covariance[channel] / np.float32(variance[channel] + epsilon))
                )
                old_intercept = np.float32(
                    mean_y[channel] - np.float32(old_slope * mean_x[channel])
                )
                old_prediction = np.float32(old_intercept + np.float32(old_slope * matured_u))
                error = np.float32(matured_v - old_prediction)

                if arm != "B" or row_index <= 98:
                    dx = np.float32(matured_u - mean_x[channel])
                    dy = np.float32(matured_v - mean_y[channel])
                    dx_squared = np.float32(dx * dx)
                    dx_dy = np.float32(dx * dy)
                    mean_x[channel] = np.float32(mean_x[channel] + np.float32(update * dx))
                    mean_y[channel] = np.float32(mean_y[channel] + np.float32(update * dy))
                    variance[channel] = np.float32(
                        decay * np.float32(variance[channel] + np.float32(update * dx_squared))
                    )
                    covariance[channel] = np.float32(
                        decay * np.float32(covariance[channel] + np.float32(update * dx_dy))
                    )

                slope = (
                    np.float32(0.0)
                    if arm == "A"
                    else np.float32(covariance[channel] / np.float32(variance[channel] + epsilon))
                )
                intercept = np.float32(mean_y[channel] - np.float32(slope * mean_x[channel]))
                forecast = np.float32(intercept + np.float32(slope * current_u))
                output[row_index, channel] = matured_v
                output[row_index, 52 + channel] = _clip(forecast)
                output[row_index, 104 + channel] = _clip(error)
                output[row_index, 156 + channel] = _clip(slope)

            i0_history[maturity_slot, channel] = current_i0
            u_history[maturity_slot, channel] = current_u
            i1_history[lag_slot, channel] = current_i1

    return output, {
        "i1_history": i1_history,
        "i0_history": i0_history,
        "u_history": u_history,
        "mx": mean_x,
        "my": mean_y,
        "variance": variance,
        "covariance": covariance,
        "rows": len(raw),
    }


def _assert_state_equal(actual: dict[str, object], expected: dict[str, np.ndarray | int]) -> None:
    """Check every causal ring and coefficient against the independent oracle."""
    for name, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[name], value)
        else:
            assert actual[name] == value


@pytest.mark.parametrize("lag", [1, 16])
@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_compiled_matches_scalar_oracle_and_python_reference(arm: str, lag: int) -> None:
    """Both implementations preserve exact FP32 fields and final state."""
    raw = _sequence()
    normalizer = fit_online_relationship_normalizer([raw], lag)
    expected, expected_state = _scalar_oracle(raw, arm, lag, normalizer)
    reference = OnlineRelationshipState(arm, lag, normalizer)
    compiled = OnlineRelationshipState(arm, lag, normalizer)

    reference_values = reference.transform(raw)
    compiled_values = compiled.transform_compiled(raw)

    np.testing.assert_array_equal(reference_values, expected)
    np.testing.assert_array_equal(compiled_values, expected)
    _assert_state_equal(reference.state_dict(), expected_state)
    _assert_state_equal(compiled.state_dict(), expected_state)


@pytest.mark.parametrize("lag", [1, 16])
@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_compiled_prefix_chunks_and_checkpoint_restore_are_exact(arm: str, lag: int) -> None:
    """Compiled prefixes and restored/mixed chunks equal an uninterrupted scan."""
    raw = _sequence()
    normalizer = fit_online_relationship_normalizer([raw], lag)
    baseline_state = OnlineRelationshipState(arm, lag, normalizer)
    baseline = baseline_state.transform_compiled(raw)

    altered = raw.copy()
    altered[80:, :104] += np.float32(5.0)
    changed = OnlineRelationshipState(arm, lag, normalizer).transform_compiled(altered)
    np.testing.assert_array_equal(changed[:80], baseline[:80])

    chunk_state = OnlineRelationshipState(arm, lag, normalizer)
    first = chunk_state.transform_compiled(raw[:37])
    restored = OnlineRelationshipState(arm, lag, normalizer)
    restored.load_state_dict(chunk_state.state_dict())
    second = restored.transform_compiled(raw[37:80])
    third = restored.transform(raw[80:])
    np.testing.assert_array_equal(np.concatenate((first, second, third)), baseline)
    for name in ("i1_history", "i0_history", "u_history", "mx", "my", "variance", "covariance"):
        np.testing.assert_array_equal(
            restored.state_dict()[name], baseline_state.state_dict()[name]
        )


def test_compiled_without_fitting_normalizer_matches_reference() -> None:
    """The optional kernel also preserves the identity-scaling path."""
    raw = _sequence(40)
    reference = OnlineRelationshipState("C", 16)
    compiled = OnlineRelationshipState("C", 16)
    np.testing.assert_array_equal(reference.transform(raw), compiled.transform_compiled(raw))
