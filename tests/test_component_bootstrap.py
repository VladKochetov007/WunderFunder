from __future__ import annotations

import numpy as np
import pytest

from wnn.evaluation import paired_component_bootstrap, paired_component_moments
from wnn.gmetric import global_wp, wp_sums


def sums(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.stack([wp_sums(y[..., k], p[..., k], m) for k in range(2)], axis=1)


def test_component_bootstrap_recalculates_global_scores() -> None:
    rng = np.random.default_rng(9)
    y = rng.normal(size=(18, 300, 2)).astype(np.float32)
    reference = (y + rng.normal(size=y.shape)).astype(np.float32)
    candidate = (y + 0.4 * rng.normal(size=y.shape)).astype(np.float32)
    mask = rng.random(y.shape[:2]) < 0.2
    components = np.repeat(np.arange(6), 3)
    result = paired_component_bootstrap(
        sums(y, candidate, mask), sums(y, reference, mask), components, n_boot=200
    )
    expected = global_wp(y, candidate, mask)["wp"] - global_wp(y, reference, mask)["wp"]
    assert result["mean_diff"] == pytest.approx(expected, abs=1e-10)
    assert result["ci95"][0] > 0
    assert result["n_components"] == 6
    assert result["bootstrap_sd"] > 0


def test_component_bootstrap_is_paired_and_deterministic() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=(5, 100, 2)).astype(np.float32)
    p = rng.normal(size=y.shape).astype(np.float32)
    stats = sums(y, p, np.ones(y.shape[:2], dtype=bool))
    result = paired_component_bootstrap(stats, stats, np.arange(5), n_boot=20)
    assert result["mean_diff"] == 0
    assert result["ci95"] == [0, 0]
    assert result == paired_component_bootstrap(stats, stats, np.arange(5), n_boot=20)


def test_component_bootstrap_exposes_draws_for_independent_fold_aggregation() -> None:
    rng = np.random.default_rng(71)
    y = rng.normal(size=(6, 80, 2)).astype(np.float32)
    candidate = (y + 0.2 * rng.normal(size=y.shape)).astype(np.float32)
    reference = (y + 0.8 * rng.normal(size=y.shape)).astype(np.float32)
    mask = np.ones(y.shape[:2], dtype=bool)
    candidate_moments, reference_moments = paired_component_moments(y, candidate, reference, mask)
    result = paired_component_bootstrap(
        candidate_moments,
        reference_moments,
        np.arange(6),
        n_boot=25,
        seed=221022,
        centered=True,
        return_draws=True,
    )
    draws = result["bootstrap_deltas"]
    assert isinstance(draws, list)
    assert len(draws) == 25
    assert (
        draws
        == paired_component_bootstrap(
            candidate_moments,
            reference_moments,
            np.arange(6),
            n_boot=25,
            seed=221022,
            centered=True,
            return_draws=True,
        )["bootstrap_deltas"]
    )


def test_component_bootstrap_rejects_misalignment() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_component_bootstrap(np.zeros((2, 2, 6)), np.zeros((3, 2, 6)), np.arange(2))


def test_centered_bootstrap_preserves_float32_near_constant_variance() -> None:
    lower = np.nextafter(np.float32(1), np.float32(0))
    y = np.ones((2, 200, 2), dtype=np.float32)
    y[:, ::2] = lower
    candidate = y.copy()
    reference = np.ones_like(y)
    mask = np.ones(y.shape[:2], dtype=bool)
    centered_a, centered_b = paired_component_moments(y, candidate, reference, mask)
    result = paired_component_bootstrap(
        centered_a, centered_b, np.arange(2), n_boot=20, centered=True
    )
    assert result["mean_diff"] == pytest.approx(1.0)
    assert result["ci95"] == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError, match="precision-ambiguous"):
        paired_component_bootstrap(sums(y, candidate, mask), sums(y, reference, mask), np.arange(2))


def test_centered_bootstrap_handles_opposite_component_means() -> None:
    center = np.float32(0.35)
    lower = np.nextafter(center, np.float32(0))
    one = np.tile(np.array([[lower, lower], [center, center]]), (16, 1))
    y = np.stack((one, -one))
    reference = np.ones_like(y)
    moments_a, moments_b = paired_component_moments(
        y, y, reference, np.ones(y.shape[:2], dtype=bool)
    )
    result = paired_component_bootstrap(
        moments_a, moments_b, np.arange(2), n_boot=50, centered=True
    )
    assert result["mean_diff"] == pytest.approx(1.0)
    assert result["ci95"] == pytest.approx([1.0, 1.0])
