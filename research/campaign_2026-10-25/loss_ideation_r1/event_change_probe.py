"""Zero-fit gate for H-LOSS25-EVENT-HAZARD-R1 (H-362).

On i0 book-event rows (any of feature columns 0-21 changes), predict from causal row
features (a) whether the dense label pair changes on that row and (b) the sign of the
difference-mode change given a change. Chain-disjoint train/test split over valid
sequences. HistGradientBoosting is a lower bound on what a recurrent state could do.
Usage: event_change_probe.py VALID_PARQUET OUT_JSON [N_SEQ] [SEED]
"""

import json
import sys

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

NON_FEATURES = {"seq_ix", "step_in_seq", "need_prediction", "t0", "t1", "is_scored"}


def sequence_rows(x: np.ndarray, y: np.ndarray, scored: np.ndarray):
    """Return causal features, change label, direction label and metadata on event rows."""
    ev = np.zeros(len(x), bool)
    ev[1:] = (x[1:, :22] != x[:-1, :22]).any(1)
    ev[:100] = False
    idx = np.flatnonzero(ev)
    prev_ev = np.full(len(x), -1)
    last = -1
    for t in range(len(x)):
        prev_ev[t] = last
        if ev[t]:
            last = t
    since = (idx - prev_ev[idx]).astype(np.float32)
    csum = np.concatenate([[0], np.cumsum(ev)])
    rate64 = (csum[idx] - csum[np.maximum(idx - 64, 0)]).astype(np.float32)
    dx = x[idx, :52] - x[idx - 1, :52]
    feats = np.hstack([x[idx], dx, since[:, None], rate64[:, None]]).astype(np.float32)
    dy = y[idx] - y[idx - 1]
    change = (np.abs(dy) > 1e-6).any(1)
    du = (dy[:, 0] - dy[:, 1]) / 2
    weight = np.abs(np.clip(y[idx], -2, 2)).sum(1)
    return feats, change, du > 0, weight, scored[idx]


def main() -> None:
    path, out = sys.argv[1], sys.argv[2]
    n_seq = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 20261025)
    pf = pq.ParquetFile(path)
    names = [n for n in pf.schema_arrow.names if n not in NON_FEATURES]
    with np.load("cache/chain_folds.npz") as z:
        keep = z["split"] == "valid"
        chain_of = dict(zip(z["row_group"][keep].tolist(), z["chain"][keep].tolist()))
    with np.load("cache/meta_valid.npz") as z:
        klass = dict(zip(z["row_group"].astype(int).tolist(), z["market_class"].astype(int).tolist()))
    groups = rng.choice(pf.num_row_groups, n_seq, replace=False)
    chains = np.array(sorted({chain_of[int(g)] for g in groups}))
    test_chains = set(rng.choice(chains, len(chains) // 3, replace=False).tolist())
    parts = {"train": [], "test": []}
    for g in groups:
        t = pf.read_row_group(int(g), columns=names + ["t0", "t1", "is_scored"])
        x = np.column_stack([t[n].to_numpy() for n in names]).astype(np.float32)
        y = np.column_stack([t["t0"].to_numpy(), t["t1"].to_numpy()]).astype(np.float32)
        rows = sequence_rows(x, y, t["is_scored"].to_numpy(zero_copy_only=False))
        cls = np.full(len(rows[0]), klass[int(g)])
        parts["test" if chain_of[int(g)] in test_chains else "train"].append((*rows, cls))
    cat = {k: [np.concatenate(c) for c in zip(*v)] for k, v in parts.items()}
    xtr, ctr, dtr, _, _, _ = cat["train"]
    xte, cte, dte, wte, ste, kte = cat["test"]
    report = {"n_seq": n_seq, "n_train_rows": int(len(xtr)), "n_test_rows": int(len(xte)),
              "test_chains": len(test_chains), "change_rate_test": float(cte.mean())}
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                         early_stopping=False, random_state=0)
    clf.fit(xtr, ctr)
    p = clf.predict_proba(xte)[:, 1]
    report["change_auc"] = float(roc_auc_score(cte, p))
    report["change_auc_label_weighted"] = float(roc_auc_score(cte, p, sample_weight=wte + 1e-6))
    report["change_auc_by_class"] = {int(k): float(roc_auc_score(cte[kte == k], p[kte == k]))
                                     for k in np.unique(kte) if len(np.unique(cte[kte == k])) == 2}
    dclf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                          early_stopping=False, random_state=0)
    dclf.fit(xtr[ctr], dtr[ctr])
    q = dclf.predict_proba(xte[cte])[:, 1]
    report["direction_auc_given_change"] = float(roc_auc_score(dte[cte], q))
    report["direction_auc_by_class"] = {
        int(k): float(roc_auc_score(dte[cte][kte[cte] == k], q[kte[cte] == k]))
        for k in np.unique(kte[cte]) if len(np.unique(dte[cte][kte[cte] == k])) == 2}
    report["direction_auc_scored_rows"] = float(roc_auc_score(dte[cte & ste], dclf.predict_proba(xte[cte & ste])[:, 1]))
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
