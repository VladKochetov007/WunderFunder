"""DTW streams H1 ceiling: IN-SAMPLE linear fit of the direction residual on i1/DTW features.

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited \
        .venv/bin/python scripts/cr_dtw_streams_ceiling.py [STRIDE]

Same features and scoring as cr_dtw_streams_linear.py, but the per-class ridge is fitted on ALL
folds and evaluated on the same rows (optimistic, not deployable): if even this is small, the
linear i1 lead-lag signal left in the residual cannot move the score.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from cr_dtw_streams_common import OUT, panel, report
from cr_dtw_streams_linear import FEATS


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    z = np.load(OUT / f"h1_rows_s{stride}.npz")
    X, y, p, seq = z["X"].astype(np.float64), z["y"].astype(np.float64), z["p"].astype(np.float64), z["seq"]
    names = list(z["names"])
    fold, cls, S_full, n_scored = panel()
    rc = cls[seq]
    yc = np.clip(y, -2, 2)
    yD, pD = (yc[:, 0] - yc[:, 1]) / 2, (np.clip(p[:, 0], -2, 2) - np.clip(p[:, 1], -2, 2)) / 2
    w = np.abs(yc).sum(1)
    cols = [names.index(f) for f in FEATS]
    Z = np.sign(X[:, cols]) * np.sqrt(np.abs(X[:, cols]))
    g = np.zeros(len(y))
    for c in (1, 2, 3):
        s = rc == c
        b = (w[s] * yD[s] * pD[s]).sum() / (w[s] * pD[s] ** 2).sum()
        A = np.c_[np.ones(s.sum()), Z[s], pD[s]]
        coef = np.linalg.lstsq(A * np.sqrt(w[s, None]), np.sqrt(w[s]) * (yD[s] - b * pD[s]), rcond=None)[0]
        g[s] = A @ coef / b
    nseq = len(cls)
    n_samp = np.bincount(seq, minlength=nseq)
    fac = np.where(n_samp > 0, n_scored / np.maximum(n_samp, 1), 0.0)
    out = {}
    for lam in (0.5, 1.0):
        pn = p + lam * g[:, None] * np.array([1.0, -1.0])
        S_c = S_full.copy()
        for k in range(2):
            for arr, sign in ((p, -1.0), (pn, 1.0)):
                yy, pp = yc[:, k], np.clip(arr[:, k], -2, 2)
                ww = np.abs(yy)
                for j, v in enumerate((ww, ww * yy, ww * pp, ww * yy * yy, ww * pp * pp, ww * yy * pp)):
                    S_c[:, k, j] += sign * fac * np.bincount(seq, weights=v, minlength=nseq)
        out[f"lam{lam}"] = report(S_full, S_c, fold, cls)
        print(lam, json.dumps({k: out[f"lam{lam}"][k] for k in ("pooled", "public_mix", "class1", "class2", "class3")}))
    (OUT / f"h1_ceiling_s{stride}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
