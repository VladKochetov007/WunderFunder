from __future__ import annotations

import numpy as np
import pytest

from wnn.domain_stress import freeze_binary_panel, mixture_counts, paired_mixture_scores
from wnn.evaluation import paired_component_moments
from wnn.gmetric import global_wp


def test_cutoff_uses_only_fit_components_and_retains_undersized_status() -> None:
    panel = freeze_binary_panel(
        np.array([0.0, 2.0, 4.0]),
        np.array([0, 0, 1]),
        np.array([-100.0, 100.0, 1000.0]),
        np.array([2, 3, 4]),
        minimum_components=2,
    )
    assert panel["cutoff"] == 2.5
    assert panel["component_counts"] == [1, 2]
    assert panel["status"] == "undersized"
    with pytest.raises(ValueError, match="overlap"):
        freeze_binary_panel(np.array([1.0]), np.array([2]), np.array([0.0]), np.array([2]))


def test_component_mass_does_not_reweight_large_components_twice() -> None:
    counts = mixture_counts(np.array([0, 0, 1, 2]), np.array([0, 0, 0, 1]), 0.75, 1)
    np.testing.assert_allclose(counts, [0.375, 0.375, 0.375, 2.25])
    with pytest.raises(ValueError, match="straddle"):
        mixture_counts(np.array([0, 0, 1]), np.array([0, 1, 1]), 0.25, 1)
    with pytest.raises(ValueError, match="too few"):
        mixture_counts(np.arange(4), np.array([0, 0, 1, 1]), 0.25)


def test_mixture_equals_actual_complete_component_replication() -> None:
    rng = np.random.default_rng(20261013)
    y = rng.normal(size=(4, 17, 2)).astype(np.float32)
    prediction = y * 0.4 + rng.normal(size=y.shape)
    reference = y * 0.6 + rng.normal(size=y.shape)
    mask = rng.uniform(size=y.shape[:2]) > 0.25
    a, b = paired_component_moments(y, prediction, reference, mask)
    panels = paired_mixture_scores(a, b, np.arange(4), np.array([0, 0, 1, 1]), 1)
    for result, repetitions in zip(panels, ([3, 3, 1, 1], [1, 1, 3, 3]), strict=True):
        indices = np.repeat(np.arange(4), repetitions)
        expected = global_wp(y[indices], prediction[indices], mask[indices])["wp"]
        baseline = global_wp(y[indices], reference[indices], mask[indices])["wp"]
        assert result["candidate_wp"] == pytest.approx(expected, abs=1e-12)
        assert result["paired_delta"] == pytest.approx(expected - baseline, abs=1e-12)
    altered = b.copy()
    altered[0, 0, 0] *= 1.01
    with pytest.raises(ValueError, match="label moments"):
        paired_mixture_scores(a, altered, np.arange(4), np.array([0, 0, 1, 1]), 1)
