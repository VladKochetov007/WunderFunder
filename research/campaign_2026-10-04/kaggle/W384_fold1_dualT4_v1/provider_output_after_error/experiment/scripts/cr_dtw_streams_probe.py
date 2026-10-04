"""DTW streams H1 step 3: cross-fitted nonlinear residual probe for i1 lead-lag features.

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited \
        .venv/bin/python scripts/cr_dtw_streams_probe.py [STRIDE]

Per class (1-3) and target, a HistGradientBoosting regressor fits r = y - beta*p (|y|-weighted,
clip +-2; beta = weighted OLS scale on the training folds) on three folds and predicts the fourth
(fold = chain-excluded campaign fold; never the evaluated rows). Candidate p' = p + lam*g/beta
keeps the class scale. Feature sets: A = p + i0-only path features (control for "a probe on
residuals finds anything"), B = A + i1 fixed-lag/EC features, C = B + DTW alignment features,
D = A + DTW features only. Training uses future labels (that is what labels are); inference-time
inputs are causal rows <= t.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from cr_dtw_streams_common import OUT, panel, report

SETS = {
    "A": ["p0", "p1", "since0", "r0_20", "r0_100", "r0_500", "r0_2000"],
}
I1 = ["since1", "r1_20", "r1_100", "r1_500", "r1_2000", "r1_since0", "ec_200", "ec_2000", "unk1"]
DTW = ["dtw_pend", "dtw_lag", "dtw_ratio", "dtw_cc"]
SETS["B"] = SETS["A"] + I1
SETS["C"] = SETS["B"] + DTW
SETS["D"] = SETS["A"] + DTW
LAMS = (0.5, 1.0)


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sets = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SETS)
    z = np.load(OUT / f"h1_rows_s{stride}.npz")
    X, y, p, seq = z["X"], z["y"].astype(np.float64), z["p"].astype(np.float64), z["seq"]
    names = list(z["names"])
    fold, cls, S_full, n_scored = panel()
    rcls, rfold = cls[seq], fold[seq]
    yc, pc = np.clip(y, -2, 2), np.clip(p, -2, 2)
    w = np.abs(yc)
    nseq = len(cls)
    n_samp = np.bincount(seq, minlength=nseq)
    fac = np.where(n_samp > 0, n_scored / np.maximum(n_samp, 1), 0.0)
    results = {}
    for sname in sets:
        cols = [names.index(c) for c in SETS[sname]]
        g = np.zeros_like(y)
        beta = np.ones((4, 2, 4))
        for c in (1, 2, 3):
            for k in range(2):
                for f in range(4):
                    tr = (rcls == c) & (rfold != f)
                    te = (rcls == c) & (rfold == f)
                    b = (w[tr, k] * yc[tr, k] * pc[tr, k]).sum() / (w[tr, k] * pc[tr, k] ** 2).sum()
                    beta[c, k, f] = b
                    m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=31,
                                                      min_samples_leaf=400, l2_regularization=1.0, random_state=0)
                    m.fit(X[tr][:, cols], yc[tr, k] - b * pc[tr, k], sample_weight=w[tr, k])
                    g[te, k] = m.predict(X[te][:, cols]) / b
            print(sname, "class", c, "fitted", flush=True)
        for lam in LAMS:
            pn = p + lam * g
            S_c = S_full.copy()
            # full sums + expansion factor * (sampled candidate sums - sampled baseline sums), per sequence
            for k in range(2):
                for arr, sign in ((p, -1.0), (pn, 1.0)):
                    yy = np.clip(y[:, k], -2, 2)
                    pp = np.clip(arr[:, k], -2, 2)
                    ww = np.abs(yy)
                    comps = [ww, ww * yy, ww * pp, ww * yy * yy, ww * pp * pp, ww * yy * pp]
                    for j, v in enumerate(comps):
                        S_c[:, k, j] += sign * fac * np.bincount(seq, weights=v, minlength=nseq)
            results[f"{sname}_lam{lam}"] = report(S_full, S_c, fold, cls)
            print(sname, lam, json.dumps(results[f"{sname}_lam{lam}"]), flush=True)
    tag = "_".join(sets)
    (OUT / f"h1_probe_s{stride}_{tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
