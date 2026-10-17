"""Small deterministic checks for the fixed no-fit prediction transforms."""

from __future__ import annotations

import numpy as np
import pytest
from gmetric_standalone import GlobalAccumulator
from prediction_audit_core import (
    SCENARIO_NAMES,
    age_buckets,
    build_scenarios,
    causal_ema,
    clipped_label_run_oracle,
    event_flags,
    hold_until_event,
)


def test_event_and_or_clocks_reset_at_sequence_start() -> None:
    p0 = np.asarray([[0, 1, 1, 2], [7, 7, 8, 8]], dtype=np.float32)
    p11 = np.asarray([[0, 1, 2, 2], [0, 1, 1, 2]], dtype=np.float32)
    both, either = event_flags(p0, p11)
    np.testing.assert_array_equal(both, [[False, True, False, False], [False, False, False, False]])
    np.testing.assert_array_equal(either, [[False, True, True, True], [False, True, True, True]])
    np.testing.assert_array_equal(age_buckets(both)[0], [0, 0, 1, 2])


def test_hold_emits_baseline_first_then_only_at_triggers() -> None:
    prediction = np.arange(10, dtype=np.float32).reshape(1, 5, 2)
    trigger = np.asarray([[False, False, True, False, False]])
    held, emitted = hold_until_event(prediction, trigger)
    np.testing.assert_array_equal(emitted, [[True, False, True, False, False]])
    np.testing.assert_array_equal(held, prediction[:, [0, 0, 2, 2, 2]])


def test_ema_uses_fixed_alpha_and_resets_per_sequence() -> None:
    prediction = np.asarray([[[1.0], [0.0], [0.0]], [[4.0], [0.0], [0.0]]], dtype=np.float32)
    output = causal_ema(prediction, tau=4)
    expected = np.asarray([[[1.0], [0.75], [0.5625]], [[4.0], [3.0], [2.25]]], dtype=np.float32)
    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-7)


def test_label_run_oracle_preserves_covariance_and_partitions_variance() -> None:
    target = np.asarray([[[0.5, 0.5], [0.5, 0.5], [1.0, 0.5], [1.0, 0.5]]], dtype=np.float32)
    prediction = np.asarray([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]], dtype=np.float32)
    mask = np.ones((1, 4), dtype=bool)
    oracle, within = clipped_label_run_oracle(target, prediction, mask)
    np.testing.assert_array_equal(oracle[0, :, 0], [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_array_equal(oracle[0, :, 1], [0.5, 0.5, 0.5, 0.5])
    base_acc, oracle_acc = GlobalAccumulator(), GlobalAccumulator()
    base_acc.add(target, prediction, mask)
    oracle_acc.add(target, oracle, mask)
    np.testing.assert_allclose(oracle_acc._moments[:, 5], base_acc._moments[:, 5], atol=1e-14)
    np.testing.assert_allclose(within, base_acc._moments[:, 4], atol=1e-14)
    np.testing.assert_allclose(
        oracle_acc._moments[:, 4] + within, base_acc._moments[:, 4], atol=1e-14
    )


def test_scenario_matrix_has_declared_order_and_keeps_baseline() -> None:
    prediction = np.linspace(-1, 1, 24, dtype=np.float32).reshape(2, 6, 2)
    target = prediction.copy()
    mask = np.ones((2, 6), dtype=bool)
    p0 = np.asarray([[0, 1, 1, 2, 2, 2], [0, 0, 1, 1, 2, 2]], dtype=np.float32)
    p11 = np.asarray([[0, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3]], dtype=np.float32)
    and_event, or_event = event_flags(p0, p11)
    scenarios = build_scenarios(prediction, and_event, or_event, mask, target)
    assert tuple(scenarios) == SCENARIO_NAMES
    np.testing.assert_array_equal(scenarios["baseline"], prediction)
    assert all(value.shape == prediction.shape for value in scenarios.values())


@pytest.mark.parametrize("bad_shape", [((1, 2, 2), (1, 3)), ((1, 3), (1, 3))])
def test_oracle_rejects_misaligned_shapes(
    bad_shape: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    prediction_shape, mask_shape = bad_shape
    with pytest.raises(ValueError, match="shapes do not align"):
        clipped_label_run_oracle(
            np.zeros(prediction_shape), np.zeros(prediction_shape), np.ones(mask_shape, dtype=bool)
        )
