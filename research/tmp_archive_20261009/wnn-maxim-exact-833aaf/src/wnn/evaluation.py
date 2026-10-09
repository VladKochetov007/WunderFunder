"""Validation statistics: paired, class-stratified bootstrap over sequences.

Competition WP is an equal mean over eligible sequences, so the resampling unit
is the sequence. Two runs are compared only on blocks eligible in both.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap(a, b, strata=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of (a - b) over common finite entries with a percentile 95% CI.

    ``a``/``b``: per-sequence scores aligned by sequence. ``strata``: optional
    labels (e.g. market class); resampling is done within each stratum so the
    class mix stays fixed.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("score arrays must align by sequence")
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    s = np.zeros(len(d), dtype=np.int64) if strata is None else np.asarray(strata)[ok]
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(s == g) for g in np.unique(s)]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        boots[i] = d[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "n": int(len(d)),
            "p_le_0": float((boots <= 0).mean())}
