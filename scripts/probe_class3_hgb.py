"""Nonlinear probe for large-tick (class 3) targets from queue/flow features, metric-true."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from wnn.data import SequenceFile, market_class
from wnn.metric import block_scores


def design(x: np.ndarray, groups) -> np.ndarray:
    t = len(x)
    ev = np.r_[True, (np.diff(x[:, 0]) != 0) & (np.diff(x[:, 11]) != 0)]
    last = np.maximum.accumulate(np.where(ev, np.arange(t), 0))
    parts = []
    if "vol" in groups:
        parts.append(x[:, 22:44])
    if "dvol" in groups:
        parts.append(x[:, 22:44] - x[last, 22:44])
    if "px" in groups:
        parts.append(x[:, 0:22])
    if "since" in groups:
        parts.append(np.log1p(np.arange(t) - last)[:, None])
    if "trade" in groups:
        tr = x[:, 44:52]
        chg = np.r_[False, np.diff(x[:, 48]) != 0].astype(np.float32)
        csum = np.cumsum(chg)
        parts += [tr, (csum - csum[last])[:, None]]
    if "i1" in groups:
        parts.append(x[:, 52:104])
        ev1 = np.r_[True, (np.diff(x[:, 52]) != 0) | (np.diff(x[:, 63]) != 0)]
        last1 = np.maximum.accumulate(np.where(ev1, np.arange(t), 0))
        parts.append(np.log1p(np.arange(t) - last1)[:, None])
        parts.append(x[:, 52:74] - x[last, 52:74])
    if "a" in groups:
        parts.append(x[:, 104:112])
    return np.hstack(parts).astype(np.float32)


def main():
    v = SequenceFile("datasets/valid.parquet")
    order = np.random.default_rng(0).permutation(len(v))
    train = [v.read(i) for i in order[:500]]
    test = [v.read(i) for i in order[500:800]]
    train = [s for s in train if market_class(s.targets) == 3]
    test = [s for s in test if market_class(s.targets) == 3]
    print("class3 train", len(train), "test", len(test), flush=True)
    for groups in [
        ("vol",),
        ("vol", "dvol", "since"),
        ("vol", "dvol", "since", "trade"),
        ("vol", "dvol", "since", "trade", "i1"),
        ("vol", "dvol", "since", "trade", "i1", "a", "px"),
    ]:
        xs = np.vstack([design(s.features, groups)[99::4] for s in train])
        preds = []
        for k in range(2):
            ys = np.concatenate([s.targets[99::4, k] for s in train])
            w = np.abs(np.clip(ys, -2, 2)) + 0.05
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63)
            m.fit(xs, ys, sample_weight=w)
            preds.append([m.predict(design(s.features, groups)) for s in test])
        p = np.stack([np.stack([preds[0][i], preds[1][i]], 1) for i in range(len(test))]).astype(
            np.float32
        )
        y = np.stack([s.targets for s in test])
        mask = np.stack([s.scored for s in test])
        print(groups, round(float(np.nanmean(block_scores(y, p, mask).mean(1))), 3), flush=True)


if __name__ == "__main__":
    main()
