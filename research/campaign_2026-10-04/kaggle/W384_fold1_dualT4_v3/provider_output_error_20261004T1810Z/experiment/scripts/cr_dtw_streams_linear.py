"""DTW streams H1 step 3b: linear partial correlations and cross-fitted ridge on the residual.

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited \
        .venv/bin/python scripts/cr_dtw_streams_linear.py [STRIDE]

Per class and fold: |y|-weighted correlation of each feature with the direction residual
rD = yD - beta*pD (yD = (t0 - t1)/2, w = |t0| + |t1|), after projecting the feature on pD.
Raw (unconditional) correlation with yD is shown next to it: a large raw and ~0 partial correlation
means the GRU already uses the signal. Then a cross-fitted ridge (3 folds -> 4th) on
signed-sqrt features, scored as p' = p + lam * g * (+1, -1) / beta.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from cr_dtw_streams_common import OUT, panel, report

FEATS = ["r0_20", "r0_100", "r0_500", "r1_20", "r1_100", "r1_500", "r1_2000", "r1_since0", "ec_200", "ec_2000",
         "dtw_pend", "dtw_lag", "dtw_cc"]


def wcorr(a, b, w):
    a = a - np.average(a, weights=w)
    b = b - np.average(b, weights=w)
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    z = np.load(OUT / f"h1_rows_s{stride}.npz")
    X, y, p, seq = z["X"].astype(np.float64), z["y"].astype(np.float64), z["p"].astype(np.float64), z["seq"]
    names = list(z["names"])
    fold, cls, S_full, n_scored = panel()
    rc, rf = cls[seq], fold[seq]
    yc = np.clip(y, -2, 2)
    yD, pD = (yc[:, 0] - yc[:, 1]) / 2, (np.clip(p[:, 0], -2, 2) - np.clip(p[:, 1], -2, 2)) / 2
    w = np.abs(yc).sum(1)
    out = {"partial": {}, "ridge": {}}
    for c in (1, 2, 3):
        s = rc == c
        b = (w[s] * yD[s] * pD[s]).sum() / (w[s] * pD[s] ** 2).sum()
        r = yD[s] - b * pD[s]
        tab = {}
        for f in FEATS:
            x = X[s, names.index(f)]
            xr = x - pD[s] * (w[s] * x * pD[s]).sum() / (w[s] * pD[s] ** 2).sum()
            per_fold = [round(wcorr(xr[rf[s] == k], r[rf[s] == k], w[s][rf[s] == k]), 4) for k in range(4)]
            tab[f] = {"raw_corr_yD": round(wcorr(x, yD[s], w[s]), 4), "partial_corr_resid": round(wcorr(xr, r, w[s]), 4),
                      "partial_by_fold": per_fold}
        out["partial"][f"class{c}"] = tab
        print("class", c, json.dumps(tab), flush=True)
    cols = [names.index(f) for f in FEATS]
    Z = np.sign(X[:, cols]) * np.sqrt(np.abs(X[:, cols]))
    g = np.zeros(len(y))
    for c in (1, 2, 3):
        for k in range(4):
            tr, te = (rc == c) & (rf != k), (rc == c) & (rf == k)
            b = (w[tr] * yD[tr] * pD[tr]).sum() / (w[tr] * pD[tr] ** 2).sum()
            A = np.c_[np.ones(tr.sum()), Z[tr], pD[tr]]
            mu = np.average(A, axis=0, weights=w[tr])
            sd = np.sqrt(np.average((A - mu) ** 2, axis=0, weights=w[tr])) + 1e-9
            mu[0], sd[0] = 0, 1
            An = (A - mu) / sd
            coef = np.linalg.solve(An.T @ (An * w[tr, None]) + 1e-2 * w[tr].sum() * np.eye(An.shape[1]),
                                   An.T @ (w[tr] * (yD[tr] - b * pD[tr])))
            At = (np.c_[np.ones(te.sum()), Z[te], pD[te]] - mu) / sd
            g[te] = At @ coef / b
    nseq = len(cls)
    n_samp = np.bincount(seq, minlength=nseq)
    fac = np.where(n_samp > 0, n_scored / np.maximum(n_samp, 1), 0.0)
    for lam in (0.5, 1.0, 2.0):
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
    (OUT / f"h1_linear_s{stride}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
