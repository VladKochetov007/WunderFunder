"""DTW streams H4: does the GRU forget sequence-level drift? Long-memory summaries of its own output.

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited \
        .venv/bin/python scripts/cr_dtw_streams_level.py

At every 3rd scored row t (all classes): trailing means of the OOF direction pD = (p0 - p1)/2 and
level pS = (p0 + p1)/2 over windows of 500 / 2000 / 8000 rows and since row 99 (all rows <= t:
causal, the model's own past outputs are available online). Per class: |y|-weighted partial
correlation of each summary with the direction residual rD = yD - beta*pD (feature projected on pD),
by fold; then a cross-fitted ridge (3 folds -> 4th, per class) scored as p' = p + lam*g*(1, -1)/beta.
"""
from __future__ import annotations

import json

import numpy as np

from cr_dtw_streams_common import OUT, ROOT, panel, report

WINS = (500, 2000, 8000)


def trailing_mean(x, w):
    c = np.r_[0.0, np.cumsum(x)]
    t = np.arange(1, len(x) + 1)
    lo = np.maximum(t - w, 99)
    return (c[t] - c[lo]) / np.maximum(t - lo, 1)


def main():
    fold, cls, S_full, n_scored = panel()
    z = np.load(ROOT / "cache/valid_targets.npz")
    Y, M = z["y"], z["mask"].copy()
    M[:, :99] = False
    par = [np.load(ROOT / f"cache/pred_c20261001_d256_s3_f{f}_valid.npy", mmap_mode="r") for f in range(4)]
    hft = [np.load(ROOT / f"cache/pred_c20261001_ft_d256_s3_f{f}h_valid.npy", mmap_mode="r") for f in range(4)]
    rng = np.random.default_rng(2)
    names = [f"mD_{w}" for w in WINS] + ["mD_all"] + [f"mS_{w}" for w in WINS] + ["mS_all", "absD_2000"]
    feats, ys, ps, seqs = [], [], [], []
    for i in range(len(cls)):
        f = fold[i]
        p = 0.5 * (np.asarray(par[f][i], np.float64) + np.asarray(hft[f][i], np.float64))
        p = np.clip(p, -2, 2)
        pD, pS = (p[:, 0] - p[:, 1]) / 2, (p[:, 0] + p[:, 1]) / 2
        rows = np.flatnonzero(M[i])
        rows = rows[rng.integers(3)::3]
        cols = []
        for x in (pD, pS):
            for w in WINS:
                cols.append(trailing_mean(x, w)[rows])
            cols.append(trailing_mean(x, 20000)[rows])
        cols.append(trailing_mean(np.abs(pD), 2000)[rows])
        feats.append(np.stack(cols, 1))
        ys.append(Y[i][rows].astype(np.float64))
        ps.append(p[rows])
        seqs.append(np.full(len(rows), i))
    X, y, p, seq = np.concatenate(feats), np.concatenate(ys), np.concatenate(ps), np.concatenate(seqs)
    rc, rf = cls[seq], fold[seq]
    yc = np.clip(y, -2, 2)
    yD, pD = (yc[:, 0] - yc[:, 1]) / 2, (p[:, 0] - p[:, 1]) / 2
    w = np.abs(yc).sum(1)
    out = {"partial": {}, "ridge": {}}

    def wc(a, b, ww):
        a = a - np.average(a, weights=ww)
        b = b - np.average(b, weights=ww)
        return float((ww * a * b).sum() / np.sqrt((ww * a * a).sum() * (ww * b * b).sum()))

    for c in range(4):
        s = rc == c
        b = (w[s] * yD[s] * pD[s]).sum() / (w[s] * pD[s] ** 2).sum()
        r = yD[s] - b * pD[s]
        tab = {}
        for j, nm in enumerate(names):
            x = X[s, j]
            xr = x - pD[s] * (w[s] * x * pD[s]).sum() / (w[s] * pD[s] ** 2).sum()
            tab[nm] = {"partial": round(wc(xr, r, w[s]), 4),
                       "by_fold": [round(wc(xr[rf[s] == k], r[rf[s] == k], w[s][rf[s] == k]), 4) for k in range(4)]}
        out["partial"][f"class{c}"] = tab
        print("class", c, json.dumps(tab), flush=True)
    g = np.zeros(len(y))
    for c in range(4):
        for k in range(4):
            tr, te = (rc == c) & (rf != k), (rc == c) & (rf == k)
            b = (w[tr] * yD[tr] * pD[tr]).sum() / (w[tr] * pD[tr] ** 2).sum()
            A = np.c_[X[tr], pD[tr]]
            mu = np.average(A, axis=0, weights=w[tr])
            sd = np.sqrt(np.average((A - mu) ** 2, axis=0, weights=w[tr])) + 1e-9
            An = np.c_[np.ones(tr.sum()), (A - mu) / sd]
            coef = np.linalg.solve(An.T @ (An * w[tr, None]) + 1e-2 * w[tr].sum() * np.eye(An.shape[1]),
                                   An.T @ (w[tr] * (yD[tr] - b * pD[tr])))
            At = np.c_[np.ones(te.sum()), (np.c_[X[te], pD[te]] - mu) / sd]
            g[te] = At @ coef / b
    nseq = len(cls)
    n_samp = np.bincount(seq, minlength=nseq)
    fac = np.where(n_samp > 0, n_scored / np.maximum(n_samp, 1), 0.0)
    for lam in (0.5, 1.0):
        pn = p + lam * g[:, None] * np.array([1.0, -1.0])
        S_c = S_full.copy()
        for k in range(2):
            for arr, sign in ((p, -1.0), (pn, 1.0)):
                yy, pp = yc[:, k], np.clip(arr[:, k], -2, 2)
                ww = np.abs(yy)
                for j, v in enumerate((ww, ww * yy, ww * pp, ww * yy * yy, ww * pp * pp, ww * yy * pp)):
                    S_c[:, k, j] += sign * fac * np.bincount(seq, weights=v, minlength=nseq)
        out["ridge"][f"lam{lam}"] = report(S_full, S_c, fold, cls)
        print("ridge", lam, json.dumps(out["ridge"][f"lam{lam}"]), flush=True)
    (OUT / "h4_level.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
