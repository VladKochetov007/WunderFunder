"""Zero-fit residual-information audit (class-3 reframe gate).

Question: do simple causal microstructure features (i1 lead deltas, per-column change
counts as depletion/refill proxies, event-rate surprises) carry information about the
strong GRU control's residual on held-out F0 evaluation sequences?

Method: F0 evaluation groups only (never fitted by the control). Causal features per scored
row. Component-grouped 3-fold cross-fitting: HistGradientBoosting predicts the |y|-weighted
residual y - p per target from features (+ p itself); out-of-fold g is stacked as p + b*g
with b fit on the training folds. Report official WP (all classes and class 3) for p and
p + b*g. A gain means extractable information the GRU missed; no gain means these
mechanisms do not add information at probe capacity.
Usage: residual_info_audit.py PRED_NAME OUT_JSON
"""

import json
import sys

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "src")
from wnn.gmetric import global_wp  # noqa: E402

NON_FEATURES = {"seq_ix", "step_in_seq", "need_prediction", "t0", "t1", "is_scored"}
WINDOWS = (4, 16, 64)


def windowed_counts(events: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([np.zeros((1, events.shape[1])), np.cumsum(events, 0)])
    t = np.arange(len(events))
    return c[t + 1] - c[np.maximum(t + 1 - w, 0)]


def causal_features(x: np.ndarray) -> np.ndarray:
    """Per-row causal features from the raw 112-column stream."""
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    sign = np.sign(d[:, :52])  # i0 + early columns: per-column change direction
    i0_ev = (d[:, :22] != 0).any(1, keepdims=True).astype(np.float32)
    i1_ev = (d[:, 52:74] != 0).any(1, keepdims=True).astype(np.float32)
    trade_ev = (d[:, 22:44] != 0).any(1, keepdims=True).astype(np.float32)
    feats = [x[:, :112]]
    for w in WINDOWS:
        feats.append(windowed_counts(sign, w))
        ev = np.hstack([i0_ev, i1_ev, trade_ev])
        feats.append(windowed_counts(ev, w))
        feats.append(windowed_counts(d[:, 52:53], w))  # i1 price-code drift
        feats.append(windowed_counts(d[:, 63:64], w))
    # rate surprise: short minus long event counts
    ev = np.hstack([i0_ev, i1_ev, trade_ev])
    feats.append(windowed_counts(ev, 8) / 8 - windowed_counts(ev, 256) / 256)
    # i1 move since last i0 book event, rows since last i0 book event
    last = np.zeros(len(x), np.int64)
    idx = np.flatnonzero(i0_ev[:, 0])
    pos = np.searchsorted(idx, np.arange(len(x)), side="right") - 1
    last = np.where(pos >= 0, idx[np.maximum(pos, 0)], 0)
    feats.append((x[:, 52] - x[last, 52])[:, None])
    feats.append((np.arange(len(x)) - last)[:, None].astype(np.float32))
    return np.hstack(feats).astype(np.float32)


def wp(y: np.ndarray, p: np.ndarray) -> float:
    return float(global_wp(y, p, np.ones(y.shape[:1], bool))["wp"])


def main() -> None:
    name, out = sys.argv[1], sys.argv[2]
    side = json.load(open(f"cache/pred_{name}_valid.json"))
    pred = np.load(f"cache/pred_{name}_valid.npy", mmap_mode="r")
    order = [int(g) for g in side["row_groups"]]
    split = json.load(open("research/campaign_2026-09-29/artifacts/splits/fold0.json"))
    groups = [int(g) for g in split["evaluation"]["valid"]]
    comp_of = {int(r["row_group"]): int(r["component"]) for r in split["sequences"]["valid"]}
    with np.load("cache/valid_targets.npz") as z:
        tpos = {int(g): i for i, g in enumerate(z["row_group"])}
        klass_all = z["market_class"]
        Y, M = z["y"], z["mask"]
        pf = pq.ParquetFile("datasets/valid.parquet")
        names = [n for n in pf.schema_arrow.names if n not in NON_FEATURES]
        rows = []
        for g in groups:
            i = tpos[g]
            m = M[i].copy()
            m[:99] = False
            sel = np.flatnonzero(m)
            t = pf.read_row_group(g, columns=names)
            x = np.column_stack([t[n].to_numpy() for n in names]).astype(np.float32)
            f = causal_features(x)[sel]
            p = np.asarray(pred[order.index(g)][sel], np.float32)
            rows.append((f, p, Y[i][sel].astype(np.float32), np.full(len(sel), comp_of[g]),
                         np.full(len(sel), klass_all[i])))
    F, P, Yc, C, K = (np.concatenate(c) for c in zip(*rows))
    Yc = np.clip(Yc, -2, 2)
    X = np.hstack([F, P])
    comps = np.unique(C)
    rng = np.random.default_rng(20261025)
    fold_of = dict(zip(comps, rng.permutation(len(comps)) % 3))
    fid = np.array([fold_of[c] for c in C])
    G = np.zeros_like(P)
    betas = []
    for k in range(3):
        tr, te = fid != k, fid == k
        for j in range(2):
            w = np.abs(Yc[tr, j])
            reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                                min_samples_leaf=200, l2_regularization=1.0, random_state=0)
            reg.fit(X[tr], Yc[tr, j] - P[tr, j], sample_weight=w)
            g_tr = reg.predict(X[tr])
            r = Yc[tr, j] - P[tr, j]
            b = float((w * g_tr * r).sum() / max((w * g_tr * g_tr).sum(), 1e-12))
            betas.append(b)
            G[te, j] = b * reg.predict(X[te])
    report = {"pred": name, "n_rows": int(len(P)), "n_components": int(len(comps)), "betas": betas}
    for label, sel in (("all", np.ones(len(P), bool)), ("class3", K == 3), ("class012", K != 3)):
        report[f"wp_{label}_control"] = wp(Yc[sel], P[sel])
        report[f"wp_{label}_stacked"] = wp(Yc[sel], P[sel] + G[sel])
        report[f"delta_{label}"] = report[f"wp_{label}_stacked"] - report[f"wp_{label}_control"]
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
