"""Strict-FP32 serial kernel for causal online relationship features."""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=False, parallel=False)
def transform_relationship_rows(
    sequence: np.ndarray,
    rows_seen: int,
    arm_code: int,
    lag: int,
    normalized: bool,
    u_mean: np.ndarray,
    u_scale: np.ndarray,
    v_mean: np.ndarray,
    v_scale: np.ndarray,
    decay: np.float32,
    update: np.float32,
    variance_epsilon: np.float32,
    feature_clip: np.float32,
    i1_history: np.ndarray,
    i0_history: np.ndarray,
    u_history: np.ndarray,
    mean_x: np.ndarray,
    mean_y: np.ndarray,
    variance: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Mutate one stream state and return its features and next row count."""
    output = np.zeros((len(sequence), 208), dtype=np.float32)
    for row_index in range(len(sequence)):
        row = sequence[row_index]
        absolute_row = rows_seen
        maturity_slot = absolute_row % 16
        lag_slot = absolute_row % lag
        for channel in range(52):
            current_i0 = row[channel]
            current_i1 = row[52 + channel]
            if absolute_row >= lag:
                raw_u = np.float32(current_i1 - i1_history[lag_slot, channel])
                if normalized:
                    current_u = np.float32((raw_u - u_mean[channel]) / u_scale[channel])
                else:
                    current_u = raw_u
                if current_u > feature_clip:
                    current_u = feature_clip
                elif current_u < -feature_clip:
                    current_u = -feature_clip
            else:
                current_u = np.float32(0.0)

            if absolute_row >= 16:
                matured_v = np.float32(current_i0 - i0_history[maturity_slot, channel])
                if normalized:
                    matured_v = np.float32((matured_v - v_mean[channel]) / v_scale[channel])
                if matured_v > feature_clip:
                    matured_v = feature_clip
                elif matured_v < -feature_clip:
                    matured_v = -feature_clip
                matured_u = u_history[maturity_slot, channel]

                if arm_code == 0:
                    old_slope = np.float32(0.0)
                else:
                    old_denominator = np.float32(variance[channel] + variance_epsilon)
                    old_slope = np.float32(covariance[channel] / old_denominator)
                old_intercept = np.float32(
                    mean_y[channel] - np.float32(old_slope * mean_x[channel])
                )
                old_product = np.float32(old_slope * matured_u)
                old_prediction = np.float32(old_intercept + old_product)
                error = np.float32(matured_v - old_prediction)

                if arm_code != 1 or absolute_row <= 98:
                    dx = np.float32(matured_u - mean_x[channel])
                    dy = np.float32(matured_v - mean_y[channel])
                    dx_squared = np.float32(dx * dx)
                    dx_dy = np.float32(dx * dy)
                    mean_x[channel] = np.float32(mean_x[channel] + np.float32(update * dx))
                    mean_y[channel] = np.float32(mean_y[channel] + np.float32(update * dy))
                    inner_variance = np.float32(variance[channel] + np.float32(update * dx_squared))
                    variance[channel] = np.float32(decay * inner_variance)
                    inner_covariance = np.float32(covariance[channel] + np.float32(update * dx_dy))
                    covariance[channel] = np.float32(decay * inner_covariance)

                if arm_code == 0:
                    slope = np.float32(0.0)
                else:
                    denominator = np.float32(variance[channel] + variance_epsilon)
                    slope = np.float32(covariance[channel] / denominator)
                intercept = np.float32(mean_y[channel] - np.float32(slope * mean_x[channel]))
                forecast = np.float32(intercept + np.float32(slope * current_u))

                output[row_index, channel] = _clip(matured_v, feature_clip)
                output[row_index, 52 + channel] = _clip(forecast, feature_clip)
                output[row_index, 104 + channel] = _clip(error, feature_clip)
                output[row_index, 156 + channel] = _clip(slope, feature_clip)

            i0_history[maturity_slot, channel] = current_i0
            u_history[maturity_slot, channel] = current_u
            i1_history[lag_slot, channel] = current_i1
        rows_seen += 1
    return output, rows_seen


@nb.njit(inline="always", fastmath=False)
def _clip(value: np.float32, limit: np.float32) -> np.float32:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value
