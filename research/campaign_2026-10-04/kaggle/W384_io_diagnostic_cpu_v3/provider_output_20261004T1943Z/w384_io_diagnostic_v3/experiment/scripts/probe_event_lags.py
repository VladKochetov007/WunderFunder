"""Linear probe: targets from prices at the last K i0 price events, scored with the real metric."""

from __future__ import annotations

import numpy as np

from wnn.data import SequenceFile, market_class
from wnn.metric import block_scores

K = 6


def event_lag_design(x: np.ndarray, k: int = K) -> np.ndarray:
    """(T, 22*k+1): i0 prices at the current and previous k-1 price events, held between events."""
    t = len(x)
    ev = np.r_[True, (np.diff(x[:, 0]) != 0) & (np.diff(x[:, 11]) != 0)]
    idx = np.flatnonzero(ev)
    p = x[idx, 0:22].astype(np.float64)
    lags = np.stack([p[np.maximum(np.arange(len(idx)) - o, 0)] for o in range(k)], axis=1)
    lags = lags.reshape(len(idx), -1)
    held = lags[np.searchsorted(idx, np.arange(t), side="right") - 1]
    return np.hstack([held, np.ones((t, 1))])


def main():
    v = SequenceFile("datasets/valid.parquet")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(v))
    train_ids, test_ids = order[:300], order[300:600]
    acc = {c: [np.zeros((22 * K + 1,) * 2), np.zeros((22 * K + 1, 2))] for c in range(4)}
    for i in train_ids:
        s = v.read(i)
        c = market_class(s.targets)
        if c < 0:
            continue
        d = event_lag_design(s.features)[99:]
        w = np.abs(np.clip(s.targets[99:], -2, 2)).mean(1, keepdims=True) + 0.05
        acc[c][0] += (d * w).T @ d
        acc[c][1] += (d * w).T @ s.targets[99:]
    coef = {c: np.linalg.solve(a + 10 * np.eye(len(a)), b) for c, (a, b) in acc.items()}
    ys, ps, ms, cs = [], [], [], []
    for i in test_ids:
        s = v.read(i)
        c = market_class(s.targets)
        if c < 0:
            continue
        p = event_lag_design(s.features) @ coef[c]
        p[:99] = 0
        ys.append(s.targets)
        ps.append(p.astype(np.float32))
        ms.append(s.scored)
        cs.append(c)
    per = block_scores(np.stack(ys), np.stack(ps), np.stack(ms)).mean(1)
    cs = np.array(cs)
    print(
        "overall",
        np.nanmean(per),
        {c: round(float(np.nanmean(per[cs == c])), 3) for c in range(4)},
        "counts",
        np.bincount(cs),
    )


if __name__ == "__main__":
    main()
