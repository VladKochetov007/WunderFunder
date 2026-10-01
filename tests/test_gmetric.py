"""Official pooled WP must match an independently centered reference."""

from __future__ import annotations

import numpy as np
import pytest

import utils
from wnn.gmetric import GlobalAccumulator, global_wp, reference_global_wp, wp_from_sums, wp_sums


def _data(seed: int = 0, b: int = 5, t: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.round(rng.standard_t(3, (b, t, 2)) * 2) * 0.25
    p = 0.4 * y + rng.normal(0, 1, y.shape) * rng.uniform(0.1, 3, (b, 1, 1))
    m = rng.random((b, t)) < 0.3
    return y.astype(np.float32), p.astype(np.float32), m


def _centered_reference(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> np.ndarray:
    results = []
    for k in range(2):
        target = np.clip(np.asarray(y, dtype=np.float32)[..., k][m], -2, 2).astype(np.float64)
        prediction = np.clip(np.asarray(p, dtype=np.float32)[..., k][m], -2, 2).astype(np.float64)
        weights = np.abs(target)
        if weights.sum() < 1e-8:
            results.append(0.0)
            continue
        centered_y = target - np.average(target, weights=weights)
        centered_p = prediction - np.average(prediction, weights=weights)
        variance_y = np.average(centered_y**2, weights=weights)
        variance_p = np.average(centered_p**2, weights=weights)
        if np.sqrt(variance_y) <= 1e-8 or np.sqrt(variance_p) <= 1e-8:
            results.append(0.0)
            continue
        covariance = np.average(centered_y * centered_p, weights=weights)
        results.append(float(np.clip(covariance / np.sqrt(variance_y * variance_p), -1, 1)))
    return np.array(results)


@pytest.mark.parametrize("seed", range(8))
def test_matches_reference(seed: int) -> None:
    y, p, m = _data(seed)
    want = _centered_reference(y, p, m)
    got = global_wp(y, p, m)
    np.testing.assert_allclose([got["t0"], got["t1"]], want, atol=1e-12, rtol=0)
    assert got["wp"] == pytest.approx(reference_global_wp(y, p, m), abs=1e-12)
    independent = utils.GlobalAccumulator()
    independent.add(y, p, m)
    assert got["wp"] == pytest.approx(independent.result()["weighted_pearson"], abs=1e-12)


def test_sums_merge_equals_pooled() -> None:
    y, p, m = _data(1)
    acc = GlobalAccumulator()
    for b in range(len(y)):
        acc.add(y[b], p[b], m[b], group=b % 2)
    assert acc.result()["wp"] == pytest.approx(global_wp(y, p, m)["wp"], abs=1e-12)
    merged = sum(acc.groups.values())
    np.testing.assert_allclose(merged, acc.sums)
    np.testing.assert_allclose(wp_from_sums(acc.sums), _centered_reference(y, p, m), atol=1e-12)


def test_clip_and_clipped_weights() -> None:
    y = np.array([3.0, -3.0, 1.0, 0.0])
    s = wp_sums(y, np.full(4, 5.0))
    assert wp_from_sums(s) == 0.0
    assert s[0] == 2 + 2 + 1
    y = np.array([[10, -7], [-0.2, 0.4], [0.8, -1], [-3, 5]], dtype=np.float32)
    p = np.array([[9, -10], [0.7, 0.2], [-4, 1.3], [1, 4]], dtype=np.float32)
    m = np.ones(4, dtype=bool)
    got = global_wp(y, p, m)
    want = _centered_reference(y, p, m)
    np.testing.assert_allclose([got["t0"], got["t1"]], want, atol=1e-12)
    assert got["wp"] == pytest.approx(
        global_wp(np.clip(y, -2, 2), np.clip(p, -2, 2), m)["wp"], abs=1e-12
    )


def test_global_differs_from_sequence_average() -> None:
    y = np.array(
        [
            [[-2, -2], [-1, -1], [1, 1], [2, 2]],
            [[-0.4, -0.4], [-0.2, -0.2], [0.2, 0.2], [0.4, 0.4]],
        ],
        dtype=np.float32,
    )
    p = y.copy()
    p[1] *= -1
    m = np.ones(y.shape[:-1], dtype=bool)
    sequence_mean = np.mean([global_wp(y[i], p[i], m[i])["wp"] for i in range(2)])
    assert sequence_mean == pytest.approx(0.0)
    assert global_wp(y, p, m)["wp"] > 0.9


@pytest.mark.parametrize("value", [0.0, 0.3, -0.3, 2.0])
def test_constant_and_zero_weight_results_are_finite(value: float) -> None:
    y = np.full((100_000, 2), value, dtype=np.float32)
    p = np.random.default_rng(3).normal(size=y.shape).astype(np.float32)
    m = np.ones(len(y), dtype=bool)
    assert wp_from_sums(wp_sums(y[:, 0], p[:, 0])) == 0.0
    assert global_wp(y, p, m)["wp"] == 0.0
    for accumulator_type in (GlobalAccumulator, utils.GlobalAccumulator):
        acc = accumulator_type()
        for start in range(0, len(y), 997):
            acc.add(y[start : start + 997], p[start : start + 997], m[start : start + 997])
        got = acc.result()
        assert got["t0"] == got["t1"] == 0.0
        assert all(np.isfinite(got[key]) for key in ("t0", "t1", "wp"))


def test_nearly_constant_centered_scoring_and_partition_invariance() -> None:
    rng = np.random.default_rng(88)
    y = np.float32(0.3) + np.spacing(np.float32(0.3)) * rng.integers(0, 4, (4096, 2))
    p = np.float32(0.7) + np.spacing(np.float32(0.7)) * (
        rng.integers(0, 2, y.shape) + 2 * ((y - np.float32(0.3)) / np.spacing(np.float32(0.3)))
    )
    m = rng.random(len(y)) > 0.1
    want = _centered_reference(y, p, m)
    got = global_wp(y, p, m)
    np.testing.assert_allclose([got["t0"], got["t1"]], want, atol=1e-9, rtol=0)
    for accumulator_type in (GlobalAccumulator, utils.GlobalAccumulator):
        acc = accumulator_type()
        for start in range(0, len(y), 17):
            acc.add(y[start : start + 17], p[start : start + 17], m[start : start + 17])
        streamed = acc.result()
        np.testing.assert_allclose([streamed["t0"], streamed["t1"]], want, atol=1e-9, rtol=0)


def test_arbitrary_partitions_ignore_unselected_nonfinite() -> None:
    y, p, m = _data(14)
    y[~m] = np.nan
    p[~m] = np.inf
    want = global_wp(y, p, m)
    y, p, m = y.reshape(-1, 2), p.reshape(-1, 2), m.reshape(-1)
    for accumulator_type in (GlobalAccumulator, utils.GlobalAccumulator):
        acc = accumulator_type()
        for start in range(0, len(y), 37):
            acc.add(y[start : start + 37], p[start : start + 37], m[start : start + 37])
        np.testing.assert_allclose(
            [acc.result()["t0"], acc.result()["t1"]], [want["t0"], want["t1"]], atol=1e-12
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("which", ["target", "prediction"])
def test_raw_selected_nonfinite_rejected(value: float, which: str) -> None:
    y = np.ones((4, 2), dtype=np.float32)
    p = y.copy()
    m = np.ones(4, dtype=bool)
    (y if which == "target" else p)[0, 0] = value
    with pytest.raises(ValueError, match="nonfinite selected"):
        wp_sums(y[:, 0], p[:, 0], m)
    with pytest.raises(ValueError, match="nonfinite selected"):
        global_wp(y, p, m)
    with pytest.raises(ValueError, match="nonfinite selected"):
        reference_global_wp(y, p, m)
    for accumulator_type in (GlobalAccumulator, utils.GlobalAccumulator):
        with pytest.raises(ValueError, match="nonfinite selected"):
            accumulator_type().add(y, p, m)


def test_empty_selection_is_distinct_from_zero_weights() -> None:
    y = np.zeros((4, 2), dtype=np.float32)
    p = np.ones_like(y)
    m = np.zeros(4, dtype=bool)
    for accumulator_type in (GlobalAccumulator, utils.GlobalAccumulator):
        acc = accumulator_type()
        acc.add(y, p, m)
        with pytest.raises(ValueError, match="no selected rows"):
            acc.result()
        acc.add(y, p, ~m)
        assert acc.result()["wp"] == 0.0
    with pytest.raises(ValueError, match="no selected rows"):
        global_wp(y, p, m)
    with pytest.raises(ValueError, match="no selected rows"):
        reference_global_wp(y, p, m)
    assert reference_global_wp(y, p, ~m) == 0.0


@pytest.mark.parametrize("kind", ["prediction", "mask", "targets"])
def test_shapes_are_validated(kind: str) -> None:
    y = np.ones((2, 4, 2), dtype=np.float32)
    p, m = y.copy(), np.ones(y.shape[:-1], dtype=bool)
    if kind == "prediction":
        p = p[:, :2]
    elif kind == "mask":
        m = m[..., None]
    else:
        y = y[..., :1]
    for function in (global_wp, reference_global_wp):
        with pytest.raises(ValueError, match="expected"):
            function(y, p, m)
    for accumulator_type in (GlobalAccumulator, utils.GlobalAccumulator):
        with pytest.raises(ValueError, match="expected"):
            accumulator_type().add(y, p, m)


def test_wp_sums_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="equal"):
        wp_sums(np.ones(4), np.ones(3))
    with pytest.raises(ValueError, match="mask shape"):
        wp_sums(np.ones(4), np.ones(4), np.ones((1, 4)))
    with pytest.raises(ValueError, match="dimension 6"):
        wp_from_sums(np.ones(5))
