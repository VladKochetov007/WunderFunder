from __future__ import annotations

import numpy as np

from wnn.viz import counts, envelope, heat, pixel_bins


def _oracle_bins(n, width):
    starts = pixel_bins(n, width)
    ends = np.r_[starts[1:], n]
    return list(zip(starts, ends))


def test_envelope_matches_per_bin_extrema():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20_000).cumsum()
    _, lo, hi = envelope(y, 1600)
    for i, (a, b) in enumerate(_oracle_bins(len(y), 1600)):
        assert lo[i] == y[a:b].min() and hi[i] == y[a:b].max()


def test_counts_and_heat_conserve_mass():
    rng = np.random.default_rng(1)
    flags = rng.random(20_000) < 0.1
    _, n = counts(flags, 777)
    assert n.sum() == flags.sum()
    block = rng.normal(size=(20_000, 22))
    h = heat(block, 500)
    assert h.shape == (22, 500)
    sizes = np.array([b - a for a, b in _oracle_bins(20_000, 500)])
    np.testing.assert_allclose((h * sizes).sum(axis=1), block.sum(axis=0))


def test_width_larger_than_series_is_clamped():
    _, lo, hi = envelope(np.arange(5.0), 1600)
    assert len(lo) == 5 and np.array_equal(lo, hi)
