"""H-016: learned level reconstruction from price jumps + deviation from EMA equilibrium.

Decoder: HGB on train events, (dP, prev prices, cur prices) -> target jump in quanta.
Q_t = cumulative decoded jump over events (causal, label-free at inference).
Prediction = Q_t - EMA_span(Q)_t (row clock). Metric on valid dev split.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from wnn.data import TARGET_QUANTA, SequenceFile, market_class
from wnn.metric import block_scores
from wnn.splits import valid_row_groups

SPANS = [50, 200, 1000, 4000]


def event_design(x):
    x = x.astype(np.float64)
    ev = np.r_[False, (np.diff(x[:, 0]) != 0) & (np.diff(x[:, 11]) != 0)]
    i = np.flatnonzero(ev)
    return i, np.hstack([x[i, 0:22] - x[i - 1, 0:22], x[i - 1, 0:22], x[i, 0:22]])


def ema(q, span):
    a = 2.0 / (span + 1.0)
    out = np.empty_like(q)
    acc = q[0]
    for t in range(len(q)):  # plain loop is fine for a probe; numba/Rust for production
        acc += a * (q[t] - acc)
        out[t] = acc
    return out


def main():
    cls = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    train = SequenceFile("datasets/train.parquet")
    valid = SequenceFile("datasets/valid.parquet")
    dev, _ = valid_row_groups("datasets/valid.parquet")
    xs, ys = [], []
    for g in range(1500):
        s = train.read(g)
        if market_class(s.targets) != cls:
            continue
        i, d = event_design(s.features)
        xs.append(d)
        ys.append((s.targets[i] - s.targets[i - 1]) / TARGET_QUANTA[cls])
        if sum(len(a) for a in xs) > 600_000:
            break
    x, y = np.vstack(xs), np.vstack(ys)
    models = [
        HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=63).fit(x, y[:, k])
        for k in range(2)
    ]
    print("decoder fit on", len(x), "train events", flush=True)
    seqs = [s for s in (valid.read(g) for g in dev[:700]) if market_class(s.targets) == cls][:150]
    preds = {sp: [] for sp in SPANS}
    for s in seqs:
        t = len(s.features)
        i, d = event_design(s.features)
        q = np.zeros((t, 2))
        for k in range(2):
            jump = np.zeros(t)
            jump[i] = models[k].predict(d)
            q[:, k] = np.cumsum(jump)
        for sp in SPANS:
            preds[sp].append(
                np.stack([q[:, k] - ema(q[:, k], sp) for k in range(2)], 1).astype(np.float32)
            )
    y = np.stack([s.targets for s in seqs])
    m = np.stack([s.scored for s in seqs])
    res = {
        f"span{sp}": round(float(np.nanmean(block_scores(y, np.stack(preds[sp]), m).mean(1))), 3)
        for sp in SPANS
    }
    print(json.dumps({"class": cls, "n_seqs": len(seqs), **res}))


if __name__ == "__main__":
    main()
