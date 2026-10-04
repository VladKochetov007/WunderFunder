"""H-031: how much of is_scored is explained by near-future events (non-causal probe)?

Fit on valid devA, AUC on devB. Features: event flags at lags -L..+K rows.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from wnn.data import SequenceFile
from wnn.features import event_features
from wnn.splits import dev_halves

FLAGS = slice(66, 74)


def design(seq, lags, leads):
    f = event_features(seq.features)[:, FLAGS]
    t = len(f)
    cols = [f]
    for k in range(1, lags + 1):
        cols.append(np.vstack([np.zeros((k, f.shape[1])), f[:-k]]))
    for k in range(1, leads + 1):
        cols.append(np.vstack([f[k:], np.zeros((k, f.shape[1]))]))
    x = np.hstack(cols)
    keep = np.arange(t) >= 99
    return x[keep], seq.scored[keep]


def main():
    v = SequenceFile("datasets/valid.parquet")
    a, b = dev_halves("datasets/valid.parquet")
    ta = [v.read(int(g)) for g in a[:60]]
    tb = [v.read(int(g)) for g in b[:40]]
    out = {}
    for leads in (0, 1, 2, 5, 10, 20):
        xa, ya = map(np.concatenate, zip(*[design(s, 5, leads) for s in ta]))
        xa = np.vstack([design(s, 5, leads)[0] for s in ta])
        xb = np.vstack([design(s, 5, leads)[0] for s in tb])
        yb = np.concatenate([design(s, 5, leads)[1] for s in tb])
        clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=63).fit(xa, ya)
        out[f"leads{leads}"] = round(float(roc_auc_score(yb, clf.predict_proba(xb)[:, 1])), 4)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
