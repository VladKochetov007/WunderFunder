from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RESEARCH_MODULES = (
    Path(__file__).parents[1]
    / "research/campaign_2026-10-06/bandit_continuation/disagreement_guided_H204"
)
sys.path.insert(0, str(RESEARCH_MODULES))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from metric_aligned_gate import (  # noqa: E402
    BETA_GRID,
    _apply,
    _fit_exact_wp_beta,
    _select_smallest_beta_within_tolerance,
)


def test_exact_metric_fit_uses_registered_grid_and_prefers_smallest_tie() -> None:
    y = np.asarray([[[1.0, 0.5], [2.0, -0.5], [-1.0, 0.75], [-2.0, -0.75]]], dtype=np.float32)
    anchor = np.zeros_like(y)
    candidate = y.copy()
    mask = np.ones((1, 4), dtype=bool)
    route = np.ones_like(mask)

    beta, score = _fit_exact_wp_beta(y, anchor, candidate, mask, route)

    assert tuple(index / 100 for index in range(101)) == BETA_GRID
    assert beta == 0.01
    assert score > 0.999999


def test_apply_routes_only_target_zero_and_clips_after_blending() -> None:
    anchor = np.asarray([[[1.5, -0.25], [-1.5, 0.5]]], dtype=np.float32)
    candidate = np.asarray([[[3.0, 1.25], [-3.0, -1.25]]], dtype=np.float32)
    route = np.asarray([[True, False]])

    result = _apply(anchor, candidate, 1.0, route)

    np.testing.assert_array_equal(result[:, :, 0], [[2.0, -1.5]])
    np.testing.assert_array_equal(result[:, :, 1], anchor[:, :, 1])


def test_beta_tie_break_uses_global_maximum_not_a_moving_threshold() -> None:
    scores = tuple(index * 0.6e-12 for index in range(len(BETA_GRID)))

    beta, selected_score = _select_smallest_beta_within_tolerance(scores)

    assert beta == 0.99
    assert selected_score == scores[99]
    assert scores[-1] - selected_score <= 1e-12


def test_zero_calibration_weight_returns_a_beta_score_pair() -> None:
    targets = np.zeros((1, 4, 2), dtype=np.float32)
    anchor = np.zeros_like(targets)
    candidate = np.ones_like(targets)

    beta, score = _fit_exact_wp_beta(
        targets,
        anchor,
        candidate,
        np.ones((1, 4), dtype=bool),
        np.ones((1, 4), dtype=bool),
    )

    assert (beta, score) == (0.0, 0.0)
