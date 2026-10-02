"""Card H-L2: DTW k-medoid prototypes (motifs) of the pre-segment event path; does membership explain residuals?

Medoids are fit without labels on a feature-only sample of class-c event paths (DTW, Sakoe-Chiba band B). Every
scored segment is assigned to its nearest medoid (causal: path up to the segment start). Per held-out fold, the
|y|-weighted mean residual of each prototype is computed on the other three folds and added with a cross-fitted
scale. Control: lock-step (Euclidean) k-medoids on the same sample.

    scripts/limited .venv/bin/python scripts/cr_dtw_labels_proto.py CLASS L B M
"""

from __future__ import annotations

import json
import sys

import numba as nb
import numpy as np
from cr_dtw_labels_common import OUT, cells, corrected, load, report_cells, wp6
from cr_dtw_labels_knn import dtw, windows

RNG = np.random.default_rng(7)
NFIT = 3000
GRID = np.round(np.arange(0.0, 3.0001, 0.1), 3)


@nb.njit(parallel=True, cache=True)
def pairwise(A, Bm, band):
    out = np.empty((A.shape[0], Bm.shape[0]))
    for i in nb.prange(A.shape[0]):
        for j in range(Bm.shape[0]):
            out[i, j] = dtw(A[i], Bm[j], band, 1e300)
    return out


def kmedoids(D, M, iters=30):
    med = RNG.choice(len(D), M, replace=False)
    for _ in range(iters):
        lab = np.argmin(D[:, med], 1)
        new = med.copy()
        for c in range(M):
            mem = np.flatnonzero(lab == c)
            if len(mem):
                new[c] = mem[np.argmin(D[np.ix_(mem, mem)].sum(1))]
        if (new == med).all():
            break
        med = new
    return med


def main():
    cls, L, B, M = (int(a) for a in sys.argv[1:5])
    t = load()
    S = t["sg_sums"][:, 0].astype(np.float64)
    tot = S[t["sg_cls"] == cls].sum(0)
    W, WY, WP, WYY, WPP, WYP = tot[:, :6].T
    beta = (WYP / W - WY * WP / W**2) / (WPP / W - (WP / W) ** 2)
    alpha = WY / W - beta * WP / W
    r = (S[:, :, 1] - alpha * S[:, :, 0] - beta * S[:, :, 2]) / np.maximum(S[:, :, 0], 1e-12)
    sel = np.flatnonzero(t["sg_cls"] == cls)
    X, ok = windows(t, t["sg_sidx"][sel], L)
    sel, X = sel[ok], X[ok]
    fold = t["sg_fold"][sel]
    mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    X = ((X - mu) / sd).astype(np.float64)
    fit = RNG.choice(len(X), NFIT, replace=False)
    res = {"class": cls, "L": L, "band": B, "M": M}
    for name, band in (("dtw", B), ("euclid", 0)):
        D = pairwise(X[fit], X[fit], band)
        med = kmedoids(D, M)
        lab = np.argmin(pairwise(X, X[fit[med]], band), 1)
        w = S[sel][:, :, 0]
        c = np.zeros((len(S), 2))
        pc = np.zeros((len(sel), 2))
        for f in range(4):
            tr = fold != f
            for j in range(2):
                num = np.bincount(lab[tr], weights=w[tr, j] * r[sel][tr, j], minlength=M)
                den = np.bincount(lab[tr], weights=w[tr, j], minlength=M)
                pc[fold == f, j] = (num / np.maximum(den, 1e-12))[lab[fold == f]]
        lams = []
        for f in range(4):
            oth = sel[fold != f]
            lam = GRID[int(np.argmax([wp6(corrected(S[oth], g * pc[fold != f]).sum(0)).mean() for g in GRID]))]
            lams.append(float(lam))
            c[sel[fold == f]] = lam * pc[fold == f]
        rep = report_cells(t, cells(t, corrected(S)), cells(t, corrected(S, c)))
        rep["lambda"] = lams
        share = np.bincount(lab, weights=w.mean(1), minlength=M) / w.mean(1).sum()
        rD = (r[sel][:, 0] - r[sel][:, 1]) / 2
        mD = np.bincount(lab, weights=w.mean(1) * rD, minlength=M) / np.maximum(np.bincount(lab, weights=w.mean(1), minlength=M), 1e-12)
        rep["proto_weight_share"] = np.round(share, 4).tolist()
        rep["proto_mean_residual_D"] = np.round(mD, 4).tolist()
        # between-prototype share of residual variance (in-sample, all folds)
        vw = w.mean(1)
        rep["between_R2_D"] = float((vw * mD[lab] ** 2).sum() / (vw * rD**2).sum())
        res[name] = rep
        print(name, json.dumps(rep, default=float), flush=True)
    (OUT / f"proto_c{cls}_L{L}_B{B}_M{M}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
