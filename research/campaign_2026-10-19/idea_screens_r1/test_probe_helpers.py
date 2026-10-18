"""Tiny analytic checks for causality, sequence reset and ridge algebra."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from feature_views import build_view
from ridge_probe import ProbeSequence, RidgeModel, _score_view, _weighted_ridge


def test_lag_view_is_causal_and_resets_at_each_sequence() -> None:
    raw = np.arange(8 * 112, dtype=np.float32).reshape(8, 112)
    view, _ = build_view(raw, "raw_lags_1_16_128")
    np.testing.assert_array_equal(view[0, 112:], np.zeros(336, dtype=np.float32))
    np.testing.assert_array_equal(view[1, 112:224], raw[0])
    np.testing.assert_array_equal(view[7, 112:224], raw[6])

    changed_future = raw.copy()
    changed_future[5:] *= -3.0
    changed_view, _ = build_view(changed_future, "raw_lags_1_16_128")
    np.testing.assert_array_equal(view[:5], changed_view[:5])

    second_sequence, _ = build_view(raw[4:], "raw_lags_1_16_128")
    np.testing.assert_array_equal(second_sequence[0, 112:], np.zeros(336, dtype=np.float32))


def test_ema_and_warmup_context_do_not_use_future_rows() -> None:
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(120, 112)).astype(np.float32)
    changed_future = raw.copy()
    changed_future[100:] += 50.0
    before, _ = build_view(raw, "raw_net_absinnovation_ema")
    after, _ = build_view(changed_future, "raw_net_absinnovation_ema")
    np.testing.assert_array_equal(before[:100], after[:100])
    beta_before, _ = build_view(raw, "raw_rolling_beta_residual")
    beta_after, _ = build_view(changed_future, "raw_rolling_beta_residual")
    np.testing.assert_array_equal(beta_before[:100], beta_after[:100])
    reset_beta, _ = build_view(raw[10:], "raw_rolling_beta_residual")
    np.testing.assert_array_equal(reset_beta[0, 112:], np.zeros(5, dtype=np.float32))

    context, _ = build_view(raw, "raw_warmup99_context")
    changed_context, _ = build_view(changed_future, "raw_warmup99_context")
    np.testing.assert_allclose(context[99, 112:], raw[:99].mean(axis=0), atol=1e-7)
    np.testing.assert_array_equal(context[99, 112:], changed_context[99, 112:])


def test_weighted_ridge_matches_one_dimensional_closed_form() -> None:
    features = np.array([[-1.0], [1.0]], dtype=np.float64)
    target = np.array([-2.0, 2.0], dtype=np.float64)
    weights = np.ones(2, dtype=np.float64)
    coefficient, intercept = _weighted_ridge(features, target, weights, alpha=1.0)
    np.testing.assert_allclose(coefficient, np.array([4.0 / 3.0]), rtol=1e-12)
    np.testing.assert_allclose(intercept, 0.0, atol=1e-12)


def test_component_scores_pool_sequences_with_different_source_pins() -> None:
    x = np.zeros((2, 112), dtype=np.float32)
    model = RidgeModel(np.zeros(112), np.ones(112), np.zeros((112, 2)), np.zeros(2), (0.0, 0.0))
    common = dict(x=x, scored=np.ones(2, dtype=bool), role="eval", component="C")
    first = ProbeSequence(
        y=np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
        source="s1",
        group="g1",
        **common,
    )
    second = ProbeSequence(
        y=np.array([[3.0, 1.0], [1.0, 3.0]], dtype=np.float32),
        source="s2",
        group="g2",
        **common,
    )

    global_metric, components, rows = _score_view([first, second], "raw", model)

    assert rows == 4
    assert global_metric is not None and global_metric["selected_rows"] == 4
    assert len(components) == 1
    assert components[0]["component"] == "C"
    assert components[0]["selected_rows"] == 4
    assert np.asarray(components[0]["pooled_raw_moments"]).shape == (2, 6)
