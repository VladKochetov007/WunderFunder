import numpy as np

from wnn.gmetric import GlobalAccumulator, global_wp, reference_global_wp, wp_from_sums, wp_sums


def _data(seed=0, b=5, t=400):
    rng = np.random.default_rng(seed)
    y = np.round(rng.standard_t(3, (b, t, 2)) * 2) * 0.25
    p = 0.4 * y + rng.normal(0, 1, y.shape) * rng.uniform(0.1, 3, (b, 1, 1))
    m = rng.random((b, t)) < 0.3
    return y.astype(np.float32), p.astype(np.float32), m


def test_matches_reference():
    y, p, m = _data()
    assert abs(global_wp(y, p, m)["wp"] - reference_global_wp(y, p, m)) < 1e-10


def test_sums_merge_equals_pooled():
    y, p, m = _data(1)
    acc = GlobalAccumulator()
    for b in range(len(y)):
        acc.add(y[b], p[b], m[b], group=b % 2)
    assert abs(acc.result()["wp"] - global_wp(y, p, m)["wp"]) < 1e-12
    merged = sum(acc.groups.values())
    assert np.allclose(merged, acc.sums)


def test_clip_and_degenerate():
    y = np.array([3.0, -3.0, 1.0, 0.0])
    s = wp_sums(y, np.full(4, 5.0))
    assert wp_from_sums(s) == 0.0  # constant prediction after clipping
    assert s[0] == 2 + 2 + 1  # weights use the clipped target
