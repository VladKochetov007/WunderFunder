"""Q6: is the model slow relative to its own inputs? (efficiency / revision-predictability probe)

If the head-FT prediction p_t were the efficient conditional expectation, its future in-segment revision
r_t = p[min(t+8, seg_end)] - p_t would be unpredictable from anything known at t. A cross-fitted gradient-boosted
model of r_t on causal inputs at t (112 raw features, p_t and parent p_t of both targets, elapsed rows in the
segment, class) measures how much of the in-segment "information timing" value (lead_in_seg bound, Q3) is reachable
causally. The readout p' = p + g * r_hat is then scored on held-out sequences (2-way cross-fit by fold pair). For
comparison the same learner predicts the residual y - p_t (classic stacking). r_t is a training-time target that
uses future rows of the model's own output; inference inputs stay causal.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

from cr_dtw_readout_core import OUT, ROOT, load_panel, wp

NSEQ_PER_FOLD = 80
D = 8


def main():
    y, mask, ft, par, E, fold, cls = load_panel()
    f = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    names = [n for n in f.schema_arrow.names if n not in ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")]
    rng = np.random.default_rng(7)
    pick = np.concatenate([rng.choice(np.flatnonzero(fold == k), NSEQ_PER_FOLD, replace=False) for k in range(4)])
    X, R, Y, P, G, C = [], [], [], [], [], []
    for i in pick:
        rows = np.flatnonzero(mask[i])
        t = f.read_row_group(int(i), columns=names, use_threads=False)
        raw = np.column_stack([t.column(n).to_numpy() for n in names]).astype(np.float32)[rows]
        start = np.maximum.accumulate(np.where(E[i] | (np.arange(E.shape[1]) == 0), np.arange(E.shape[1]), 0))
        nxt = np.empty(E.shape[1], np.int64)
        n = E.shape[1]
        for tt in range(E.shape[1] - 1, -1, -1):
            nxt[tt] = n
            if E[i, tt]:
                n = tt
        end = nxt - 1
        lead = np.minimum(rows + D, end[rows])
        p = ft[i, rows].astype(np.float32)
        feats = np.column_stack([raw, p, par[i, rows], ft[i, rows - 1], np.log1p(rows - start[rows]),
                                 np.full(len(rows), cls[i], np.float32)])
        X.append(feats)
        R.append(ft[i, lead] - p)
        Y.append(np.clip(y[i, rows], -2, 2))
        P.append(p)
        G.append(np.full(len(rows), fold[i]))
        C.append(np.full(len(rows), cls[i]))
    X, R, Y, P, G, C = (np.concatenate(a) for a in (X, R, Y, P, G, C))
    print("rows", len(X), "features", X.shape[1])
    res = {"rows": int(len(X))}
    gains = (0.25, 0.5, 1.0)
    preds = {k: np.zeros_like(P) for k in ("rev", "res")}
    for half in (0, 1):
        tr = (G // 2) != half
        te = ~tr
        for j in range(2):
            w = np.abs(Y[tr, j])
            for kind, tgt in (("rev", R[:, j]), ("res", Y[:, j] - P[:, j])):
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=63,
                                                  min_samples_leaf=200, l2_regularization=1.0, random_state=0)
                m.fit(X[tr], tgt[tr], sample_weight=w)
                preds[kind][te, j] = m.predict(X[te])
    for kind in ("rev", "res"):
        tgt = R if kind == "rev" else Y - P
        w = np.abs(Y)
        r2 = 1 - (w * (tgt - preds[kind]) ** 2).sum(0) / (w * (tgt - (w * tgt).sum(0) / w.sum(0)) ** 2).sum(0)
        res[f"{kind}_weighted_r2"] = r2.round(4).tolist()
    def wps(pp, sel):
        yy, qq = Y[sel], np.clip(pp[sel], -2, 2)
        w = np.abs(yy)
        s = np.stack([w.sum(0), (w * yy).sum(0), (w * qq).sum(0), (w * yy * yy).sum(0), (w * qq * qq).sum(0),
                      (w * yy * qq).sum(0)], -1)
        return float(wp(s))
    groups = {"all": np.ones(len(Y), bool), **{f"c{c}": C == c for c in range(4)},
              **{f"half{h}": (G // 2) == h for h in (0, 1)}}
    base = {g: wps(P, s) for g, s in groups.items()}
    res["base_wp"] = base
    for kind in ("rev", "res"):
        for g_ in gains if kind == "rev" else (1.0,):
            pp = P + g_ * preds[kind]
            res[f"{kind}_g{g_}"] = {g: round(wps(pp, s) - base[g], 5) for g, s in groups.items()}
    print(json.dumps(res, indent=1))
    (OUT / "q6_revision_probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
