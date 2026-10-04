"""Non-recurrent ensemble member: gradient boosting on causal heuristic + raw features.

    scripts/limited .venv/bin/python scripts/gbm_member.py fit
    scripts/limited .venv/bin/python scripts/gbm_member.py predict
``fit`` trains one HistGradientBoostingRegressor per target on cache/heur_train.npz with
weights |clip(y)| * p(scored)^2 (the GRU's weighting), early-stopped on held-out train
sequences; saves cache/gbm_member.pkl. ``predict`` writes cache/pred_gbm_member_valid.npy for
every validation row (features are causal, so predictions are streaming-compatible).
"""

from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_MODELS = None


def fit():
    from sklearn.ensemble import HistGradientBoostingRegressor

    d = np.load(ROOT / "cache" / "heur_train.npz")
    X = np.concatenate([d["X"], d["raw"]], 1)
    y = np.clip(d["y"].astype(np.float64), -2, 2)
    p = d["prop"].astype(np.float64)
    groups = d["group"]
    held = np.isin(groups, np.unique(groups)[::10])
    models = []
    for k in range(2):
        w = np.abs(y[:, k]) * (p ** 2 + 1e-3)
        m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=400,
                                          l2_regularization=1.0, early_stopping=False)
        m.fit(X[~held], y[~held, k], sample_weight=w[~held])
        # pick the iteration count on held-out sequences by weighted Pearson (the metric's core)
        best, best_it = -1.0, m.n_iter_
        for it, pred in enumerate(m.staged_predict(X[held]), 1):
            if it % 25:
                continue
            ww = np.abs(y[held, k])
            my, mp = (ww * y[held, k]).sum() / ww.sum(), (ww * pred).sum() / ww.sum()
            c = (ww * (y[held, k] - my) * (pred - mp)).sum() / np.sqrt((ww * (y[held, k] - my) ** 2).sum() * (ww * (pred - mp) ** 2).sum())
            if c > best:
                best, best_it = c, it
        print(f"target {k}: best iterations {best_it} held-out weighted corr {best:.4f}", flush=True)
        m.set_params(max_iter=best_it)
        m.fit(X, y[:, k], sample_weight=w)
        models.append(m)
    pickle.dump(models, open(ROOT / "cache" / "gbm_member.pkl", "wb"))


def _init():
    global _MODELS
    _MODELS = pickle.load(open(ROOT / "cache" / "gbm_member.pkl", "rb"))


def _predict(g):
    from wnn.data import SequenceFile
    from wnn.heuristics import heuristic_features

    x = SequenceFile(ROOT / "datasets" / "valid.parquet").read(g, targets=False, mask=False).features
    F = np.concatenate([heuristic_features(x), x], 1)
    return np.stack([m.predict(F) for m in _MODELS], 1).astype(np.float32)


def predict():
    groups = np.load(ROOT / "cache" / "valid_targets.npz")["row_group"]
    out = np.empty((len(groups), 20_000, 2), np.float32)
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "8")), initializer=_init) as ex:
        for i, p in enumerate(ex.map(_predict, [int(g) for g in groups], chunksize=4)):
            out[i] = p
    np.save(ROOT / "cache" / "pred_gbm_member_valid.npy", out)


if __name__ == "__main__":
    {"fit": fit, "predict": predict}[sys.argv[1]]()
