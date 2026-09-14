from __future__ import annotations

import numpy as np
import pytest

from wnn.evaluation import paired_bootstrap


def test_detects_true_improvement_and_null():
    rng = np.random.default_rng(0)
    base = rng.normal(0.6, 0.15, size=800)
    better = base + 0.01 + rng.normal(0, 0.01, size=800)
    res = paired_bootstrap(better, base)
    assert res["ci95"][0] > 0 and res["mean_diff"] == pytest.approx(0.01, abs=0.002)
    null = base + rng.normal(0, 0.01, size=800)
    res0 = paired_bootstrap(null, base)
    assert res0["ci95"][0] < 0 < res0["ci95"][1]


def test_ignores_non_common_blocks_and_keeps_strata():
    a = np.array([0.5, np.nan, 0.7, 0.9])
    b = np.array([0.4, 0.3, np.nan, 0.8])
    res = paired_bootstrap(a, b, strata=np.array([0, 0, 1, 1]), n_boot=200)
    assert res["n"] == 2 and res["mean_diff"] == pytest.approx(0.1)
    # With one element per stratum every resample is identical.
    assert res["ci95"][0] == pytest.approx(res["ci95"][1])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])
