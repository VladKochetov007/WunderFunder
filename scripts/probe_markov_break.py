"""H-011: do in-context Markov features predict class-3 break direction early? Dev split only."""

from __future__ import annotations

import json

import numpy as np
from probe_class3_hgb import design
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from wnn.data import SequenceFile, market_class
from wnn.markov import markov_features
from wnn.metric import block_scores
from wnn.splits import valid_row_groups

G = ("vol", "dvol", "since", "trade", "i1", "a")
BUCKETS = [(1, 1), (2, 5), (6, 20), (21, 100), (101, 1000), (1001, 20000)]


def break_rows(seq, feats):
    x, y = seq.features, seq.targets
    t = len(x)
    ev = np.r_[True, (np.diff(x[:, 0]) != 0) & (np.diff(x[:, 11]) != 0)]
    idx = np.flatnonzero(ev)
    nxt_i = np.searchsorted(idx, np.arange(t), side="right")
    valid = nxt_i < len(idx)
    nxt = idx[np.minimum(nxt_i, len(idx) - 1)]
    dy = y[nxt, 0] - y[np.maximum(nxt - 1, 0), 0]
    ok = valid & (dy != 0) & (np.arange(t) >= 99)
    return feats[ok], dy[ok] > 0, (nxt - np.arange(t))[ok]


def main():
    path = "datasets/valid.parquet"
    dev, _ = valid_row_groups(path)
    v = SequenceFile(path)
    order = np.random.default_rng(0).permutation(dev)
    seqs = [s for s in (v.read(i) for i in order[:700]) if market_class(s.targets) == 3]
    tr, te = seqs[: len(seqs) * 2 // 3], seqs[len(seqs) * 2 // 3 :]
    print("class3 dev train", len(tr), "test", len(te), flush=True)
    cache = {id(s): (design(s.features, G), markov_features(s.features)) for s in tr + te}
    out = {}
    for name, use_markov in [("snapshot", False), ("snapshot+markov", True)]:

        def feats(s):
            a, m = cache[id(s)]
            return np.hstack([a, m]) if use_markov else a

        rtr = [break_rows(s, feats(s)) for s in tr]
        rte = [break_rows(s, feats(s)) for s in te]
        xtr, ytr = np.vstack([r[0] for r in rtr]), np.concatenate([r[1] for r in rtr])
        xte, yte = np.vstack([r[0] for r in rte]), np.concatenate([r[1] for r in rte])
        rem = np.concatenate([r[2] for r in rte])
        sub = np.random.default_rng(1).choice(len(xtr), min(len(xtr), 600_000), replace=False)
        clf = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63).fit(
            xtr[sub], ytr[sub]
        )
        p = clf.predict_proba(xte)[:, 1]
        auc = {
            f"{lo}-{hi}": round(
                float(roc_auc_score(yte[(rem >= lo) & (rem <= hi)], p[(rem >= lo) & (rem <= hi)])),
                3,
            )
            for lo, hi in BUCKETS
            if ((rem >= lo) & (rem <= hi)).sum() > 100
        }
        # metric-true regression on all rows
        preds = []
        for k in range(2):
            xs = np.vstack([feats(s)[99::4] for s in tr])
            ys = np.concatenate([s.targets[99::4, k] for s in tr])
            reg = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=63)
            reg.fit(xs, ys, sample_weight=np.abs(np.clip(ys, -2, 2)) + 0.05)
            preds.append([reg.predict(feats(s)) for s in te])
        pr = np.stack([np.stack([preds[0][i], preds[1][i]], 1) for i in range(len(te))]).astype(
            np.float32
        )
        wp = float(
            np.nanmean(
                block_scores(
                    np.stack([s.targets for s in te]), pr, np.stack([s.scored for s in te])
                ).mean(1)
            )
        )
        out[name] = {
            "auc_overall": round(float(roc_auc_score(yte, p)), 3),
            "auc_by_rows_to_break": auc,
            "class3_wp": round(wp, 3),
        }
        print(name, json.dumps(out[name]), flush=True)


if __name__ == "__main__":
    main()
