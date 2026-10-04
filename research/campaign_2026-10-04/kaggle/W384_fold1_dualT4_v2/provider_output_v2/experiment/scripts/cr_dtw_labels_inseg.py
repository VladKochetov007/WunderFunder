"""Card H-L3: time-warped analogs of the partial path INSIDE a quiet class-c segment -> rest-of-segment residual.

At checkpoint c = e_k + TAU of every segment that is still quiet at c (causal: no quote move in (e_k, c]), the query
is the last M within-segment micro-events (rows in (e_k, c] where a trade field, an i1 code or an i0 volume code
changed): channels  cumulative bid-side / ask-side volume-code change since e_k, trade flag, i1 flag,
log elapsed rows. Fewer than M micro-events are left-padded with null events. Label = |y|-weighted residual of the
parent OOF p over scored rows in [c, segment end). DTW-kNN (band B, LB_Keogh) and lock-step kNN against the other
three folds' segments; the correction (applied only to rows >= c) is scored with exact sums, lambda cross-fitted.

    scripts/limited .venv/bin/python scripts/cr_dtw_labels_inseg.py CLASS TAU M B
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cr_dtw_labels_common import OUT, cells, corrected, load, report_cells, wp6  # noqa: E402
from cr_dtw_labels_knn import knn_dtw, knn_euclid  # noqa: E402
from cr_mature_common import ENDPOINT  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402

TRADE = [f"i0_dp{i}" for i in range(4)] + [f"i0_dv{i}" for i in range(4)]
I1 = [f"i1_p{i}" for i in range(22)]
VOL = [f"i0_v{i}" for i in range(22)]
MODEL = "c20261001_d256_s3_f{k}"
RNG = np.random.default_rng(11)
K = 64
NLIB = 20000
GRID = np.round(np.arange(0.0, 3.0001, 0.1), 3)
_G: dict = {}


def _init(fold_of, tau, m):
    z = np.load(ROOT / "cache/valid_targets.npz")
    _G.update(y=z["y"], mask=z["mask"], fold=fold_of, tau=tau, m=m,
              pred={k: np.load(ROOT / f"cache/pred_{MODEL.format(k=k)}_valid.npy", mmap_mode="r") for k in range(4)},
              pf=pq.ParquetFile(ROOT / "datasets/valid.parquet"))


def seq_paths(i):
    tau, M = _G["tau"], _G["m"]
    t = _G["pf"].read_row_group(i, columns=TRADE + I1 + VOL + ["a6"], use_threads=False)
    col = lambda names: np.column_stack([t.column(c).to_numpy() for c in names]).astype(np.float32)
    tr, i1, V = col(TRADE), col(I1), col(VOL)
    a6 = t.column("a6").to_numpy().astype(np.float32)
    n = len(a6)
    e = a6 == ENDPOINT
    e[0] = False
    ev = np.flatnonzero(e)
    ends = np.r_[ev[1:], n]
    ok = ends - ev > tau
    ev, ends = ev[ok], ends[ok]
    dV = np.zeros_like(V)
    dV[1:] = V[1:] - V[:-1]
    trf = np.zeros(n, bool)
    trf[1:] = (tr[1:] != tr[:-1]).any(1)
    i1f = np.zeros(n, bool)
    i1f[1:] = (i1[1:] != i1[:-1]).any(1)
    vf = (dV != 0).any(1)
    micro = trf | i1f | vf
    cvb, cva = np.cumsum(dV[:, :11].sum(1)), np.cumsum(dV[:, 11:].sum(1))
    y = np.clip(_G["y"][i].astype(np.float64), -2, 2)
    p = np.clip(np.asarray(_G["pred"][_G["fold"][i]][i], dtype=np.float64), -2, 2)
    msk = _G["mask"][i].copy()
    msk[:99] = False
    X, S = [], []
    for a, b in zip(ev.tolist(), ends.tolist()):
        c = a + tau
        rows = np.arange(c, b)
        rows = rows[msk[rows]]
        if len(rows) == 0:
            continue
        mi = np.flatnonzero(micro[a + 1:c + 1]) + a + 1
        mi = mi[-M:]
        P = np.zeros((M, 5), np.float32)
        if len(mi):
            P[M - len(mi):] = np.column_stack([cvb[mi] - cvb[a], cva[mi] - cva[a], trf[mi], i1f[mi], np.log1p(mi - a)])
        w = np.abs(y[rows])
        yy, pp = y[rows], p[rows]
        s = np.stack([w.sum(0), (w * yy).sum(0), (w * pp).sum(0), (w * yy * yy).sum(0), (w * pp * pp).sum(0),
                      (w * yy * pp).sum(0)], -1)  # (2, 6)
        X.append(P)
        S.append(s)
    if not X:
        return None
    return np.stack(X), np.stack(S), np.full(len(X), _G["fold"][i], np.int8)


def main():
    cls, tau, M, B = (int(a) for a in sys.argv[1:5])
    fold_of = np.full(1873, -1, np.int64)
    for k in range(4):
        s = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json", ROOT)
        fold_of[np.asarray(sorted(s.evaluation_groups))] = k
    cls_of = np.load(ROOT / "cache/valid_targets.npz")["market_class"]
    seqs = np.flatnonzero(cls_of == cls).tolist()
    parts = []
    with Pool(2, initializer=_init, initargs=(fold_of, tau, M)) as pool:
        for r in pool.imap(seq_paths, seqs, chunksize=4):
            if r is not None:
                parts.append(r)
    X = np.concatenate([p[0] for p in parts])
    S = np.concatenate([p[1] for p in parts])
    fold = np.concatenate([p[2] for p in parts]).astype(np.int64)
    # cumulative volume channels: signed log compression, then standardise
    X[..., :2] = np.sign(X[..., :2]) * np.log1p(np.abs(X[..., :2]))
    mu, sd = X.reshape(-1, 5).mean(0), X.reshape(-1, 5).std(0) + 1e-6
    X = (X - mu) / sd
    tot = S.sum(0)
    W, WY, WP, WYY, WPP, WYP = tot.T
    beta = (WYP / W - WY * WP / W**2) / (WPP / W - (WP / W) ** 2)
    alpha = WY / W - beta * WP / W
    r = (S[..., 1] - alpha * S[..., 0] - beta * S[..., 2]) / np.maximum(S[..., 0], 1e-12)
    wseg = S[..., 0].mean(1)
    res = {"class": cls, "tau": tau, "M": M, "band": B, "n_segments": int(len(X)),
           "share_of_class_weight_covered": None}
    print(res, flush=True)

    def wavg(nbr, lab):
        w = wseg[nbr]
        return (lab[nbr] * w[..., None]).sum(1) / np.maximum(w.sum(1), 1e-12)[:, None]

    def apply(c):
        s = S.copy()
        s[..., 2] += c * S[..., 0]
        s[..., 4] += 2 * c * S[..., 2] + c * c * S[..., 0]
        s[..., 5] += c * S[..., 1]
        return s

    tb = load()
    ref_cells = cells(tb, corrected(tb["sg_sums"][:, 0].astype(np.float64)))
    preds = {}
    libs = {f: RNG.choice(np.flatnonzero(fold != f), min(NLIB, int((fold != f).sum())), replace=False) for f in range(4)}
    for name, band in (("euclid", 0), ("dtw", B)):
        pc = np.zeros((len(X), 2))
        for f in range(4):
            q = np.flatnonzero(fold == f)
            lib = libs[f]
            if band == 0:
                nbr = knn_euclid(X[q], X[lib], K)
            else:
                nbr = knn_dtw(X[q].astype(np.float64), X[lib].astype(np.float64), band, K, 3000)[0]
            pc[q] = wavg(lib[nbr], r)
        preds[name] = pc
    out = {}
    for name, pc in preds.items():
        lams, new = [], S.copy()
        for f in range(4):
            tr = fold != f
            lam = GRID[int(np.argmax([wp6(apply(g * pc * tr[:, None])[tr].sum(0)).mean() for g in GRID]))]
            lams.append(float(lam))
            new[fold == f] = apply(lam * pc * (fold == f)[:, None])[fold == f]
        o = {"lambda": lams,
             "class_rows_ge_tau_delta": float(wp6(new.sum(0)).mean() - wp6(S.sum(0)).mean()),
             "fold_delta": [float(wp6(new[fold == f].sum(0)).mean() - wp6(S[fold == f].sum(0)).mean()) for f in range(4)],
             "wcorr_pred_resid": [float(np.cov(pc[:, j], r[:, j], aweights=wseg)[0, 1]
                                        / np.sqrt(np.cov(pc[:, j], aweights=wseg) * np.cov(r[:, j], aweights=wseg))) for j in range(2)]}
        diff = np.zeros((4, 4, 2, 6))
        for f in range(4):
            diff[cls, f] = (new[fold == f] - S[fold == f]).sum(0)
        o["panel"] = report_cells(tb, ref_cells, ref_cells + diff)
        out[name] = o
        print(name, o, flush=True)
    res.update(out)
    res["ref_class_rows_ge_tau_wp"] = float(wp6(S.sum(0)).mean())
    res["covered_weight"] = float(S[..., 0].sum())
    (OUT / f"inseg_c{cls}_tau{tau}_M{M}_B{B}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
