"""Card H-L1: DTW-kNN analogs of the pre-segment event path -> segment residual of the OOF GRU256 prediction.

Query = scored segment k of class c; its causal representation is the path of the last L quote-move events
(e_{k-L+1} .. e_k, all <= segment start): channels  mid level relative to e_k (ticks, clip +-6), spread ticks,
log rows between events, log trade-update rows, log i1 code changes; standardised on the library.
Label of a library segment = |y|-weighted mean residual of y on the class-level affine fit of p over the segment's
scored rows (per target). Library = segments of the other three folds only (cross-fitted), so every query's
neighbours are disjoint from its fold. Prediction = |y|-weight-averaged neighbour residual; a scale lambda per
class is fit on the other folds' predictions and the correction p + lambda * c is scored with the exact segment
sums (training-time labels of OTHER folds are used to build the library; inference uses only rows <= e_k).

  euclid: lock-step kNN (band 0) for every scored segment -> exact cross-fitted WP delta.
  dtw:    Sakoe-Chiba band B with LB_Keogh pruning, on a weight-proportional query subsample, compared with
          euclid on the same subsample and with a shuffled-label control.

    scripts/limited .venv/bin/python scripts/cr_dtw_labels_knn.py CLASS L B
"""

from __future__ import annotations

import json
import sys
import time

import numba as nb
import numpy as np
from cr_dtw_labels_common import OUT, cells, corrected, load, report_cells, wp6

RNG = np.random.default_rng(20261002)
K = 64
NLIB = 12000  # library segments per fold
NQ = 2500  # DTW query subsample per fold
GRID = np.round(np.arange(0.0, 3.0001, 0.1), 3)


@nb.njit(cache=True)
def dtw(q, c, band, cutoff):
    n = q.shape[0]
    INF = 1e300
    D = np.full((n + 1, n + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(n, i + band)
        rowmin = INF
        for j in range(lo, hi + 1):
            d = 0.0
            for h in range(q.shape[1]):
                x = q[i - 1, h] - c[j - 1, h]
                d += x * x
            m = D[i - 1, j - 1]
            if D[i - 1, j] < m:
                m = D[i - 1, j]
            if D[i, j - 1] < m:
                m = D[i, j - 1]
            D[i, j] = d + m
            if D[i, j] < rowmin:
                rowmin = D[i, j]
        if rowmin > cutoff:
            return INF
    return D[n, n]


@nb.njit(parallel=True, cache=True)
def knn_dtw(Q, lib, band, k, m_cand):
    nq, n, C = Q.shape
    nl = lib.shape[0]
    idx = np.zeros((nq, k), np.int64)
    dist = np.zeros((nq, k))
    exact = np.zeros(nq, np.bool_)
    for qi in nb.prange(nq):
        q = Q[qi]
        U = np.empty((n, C))
        Lo = np.empty((n, C))
        for i in range(n):
            for h in range(C):
                u, l = -1e300, 1e300
                for j in range(max(0, i - band), min(n, i + band + 1)):
                    v = q[j, h]
                    if v > u:
                        u = v
                    if v < l:
                        l = v
                U[i, h], Lo[i, h] = u, l
        lb = np.empty(nl)
        for j in range(nl):
            s = 0.0
            for i in range(n):
                for h in range(C):
                    v = lib[j, i, h]
                    if v > U[i, h]:
                        s += (v - U[i, h]) ** 2
                    elif v < Lo[i, h]:
                        s += (Lo[i, h] - v) ** 2
            lb[j] = s
        cand = np.argsort(lb)[:m_cand]
        bd = np.full(k, 1e300)
        bi = np.zeros(k, np.int64)
        worst = 0
        done = True
        for t in range(m_cand):
            j = cand[t]
            if lb[j] >= bd[worst]:
                break
            if t == m_cand - 1:
                done = False
            d = dtw(q, lib[j], band, bd[worst])
            if d < bd[worst]:
                bd[worst], bi[worst] = d, j
                for r in range(k):
                    if bd[r] > bd[worst]:
                        worst = r
        idx[qi], dist[qi], exact[qi] = bi, bd, done
    return idx, dist, exact


def knn_euclid(Q, lib, k, chunk=2048):
    import torch

    torch.set_num_threads(int(__import__("os").environ.get("WNN_THREADS", "1")))
    Qf = torch.from_numpy(Q.reshape(len(Q), -1).astype(np.float32))
    Lf = torch.from_numpy(lib.reshape(len(lib), -1).astype(np.float32))
    ln = (Lf * Lf).sum(1)
    out = np.zeros((len(Q), k), np.int64)
    for a in range(0, len(Q), chunk):
        d = ln[None] - 2 * Qf[a:a + chunk] @ Lf.T
        out[a:a + chunk] = torch.topk(d, k, dim=1, largest=False).indices.numpy()
    return out


def windows(t, sidx, L):
    f = t["ev_feat"]
    mid = np.cumsum((f[:, 0] + f[:, 1]) / 2.0)
    offs = np.arange(-L + 1, 1)
    ix = sidx[:, None] + offs[None]
    ok = (ix[:, 0] >= 0) & (t["ev_seq"][np.maximum(ix[:, 0], 0)] == t["ev_seq"][sidx])
    ix = np.maximum(ix, 0)
    lvl = np.clip(mid[ix] - mid[sidx][:, None], -6, 6)
    X = np.stack([lvl, np.clip(f[ix, 2], 0, 6), f[ix, 3], f[ix, 4], f[ix, 5]], -1).astype(np.float32)
    return X, ok


def wcorr(a, b, w):
    c = np.cov(a, b, aweights=w)
    return float(c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]))


def wavg(nbr, lab, wt):
    w = wt[nbr]
    return (lab[nbr] * w[..., None]).sum(1) / np.maximum(w.sum(1), 1e-12)[:, None]


def main():
    cls, L, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    t = load()
    m = 0  # parent GRU256 (head-FT gives the same picture, see drift.json)
    S = t["sg_sums"][:, m].astype(np.float64)
    tot = S[t["sg_cls"] == cls].sum(0)  # (2, 10)
    W, WY, WP, WYY, WPP, WYP = tot[:, :6].T
    beta = (WYP / W - WY * WP / W**2) / (WPP / W - (WP / W) ** 2)
    alpha = WY / W - beta * WP / W
    Wseg = S[:, :, 0].mean(1)
    r = (S[:, :, 1] - alpha * S[:, :, 0] - beta * S[:, :, 2]) / np.maximum(S[:, :, 0], 1e-12)  # (n, 2)
    sel = np.flatnonzero(t["sg_cls"] == cls)
    X, ok = windows(t, t["sg_sidx"][sel], L)
    sel, X = sel[ok], X[ok]
    fold = t["sg_fold"][sel]
    libs = {f: RNG.choice(np.flatnonzero(fold == f), min(NLIB, int((fold == f).sum())), replace=False) for f in range(4)}
    allib = np.concatenate([libs[f] for f in range(4)])
    mu, sd = X[allib].reshape(-1, X.shape[-1]).mean(0), X[allib].reshape(-1, X.shape[-1]).std(0) + 1e-6
    X = (X - mu) / sd
    res = {"class": cls, "L": L, "band": B, "K": K, "n_query_segments": int(len(sel)), "beta": beta.tolist()}
    print(res, flush=True)

    # euclid kNN for every segment, cross-fitted by fold
    t0 = time.time()
    pred = np.zeros((len(sel), 2))
    pred_y = np.zeros((len(sel), 2))
    ys = t["sg_ystart"][sel].astype(np.float64)
    perm = RNG.permutation(len(sel))  # shuffled-label control
    pred_sh = np.zeros((len(sel), 2))
    for f in range(4):
        q = np.flatnonzero(fold == f)
        lib = np.concatenate([libs[g] for g in range(4) if g != f])
        nbr = lib[knn_euclid(X[q], X[lib], K)]
        pred[q] = wavg(nbr, r[sel], Wseg[sel])
        pred_y[q] = wavg(nbr, np.clip(ys, -2, 2), Wseg[sel])
        pred_sh[q] = wavg(nbr, r[sel][perm], Wseg[sel][perm])
    res["euclid_seconds"] = round(time.time() - t0, 1)

    def crossfit(pc):
        """lambda per held-out fold fit on the other folds (this class only); return full-panel sums."""
        c = np.zeros((len(S), 2))
        lams = []
        for f in range(4):
            oth = sel[fold != f]
            cs = lambda lam: wp6(corrected(S[oth], lam * pc[fold != f]).sum(0)).mean()
            lam = GRID[int(np.argmax([cs(g) for g in GRID]))]
            lams.append(float(lam))
            c[sel[fold == f]] = lam * pc[fold == f]
        ref = cells(t, corrected(S))
        new = cells(t, corrected(S, c))
        out = report_cells(t, ref, new)
        out["lambda"] = lams
        w = Wseg[sel]
        out["wcorr_pred_resid"] = [wcorr(pc[:, j], r[sel][:, j], w) for j in range(2)]
        return out

    res["euclid_resid"] = crossfit(pred)
    res["euclid_shuffled"] = crossfit(pred_sh)
    w = Wseg[sel]
    pseg = S[sel][:, :, 2] / np.maximum(S[sel][:, :, 0], 1e-12)
    res["analog_y_skill"] = {"wcorr(knn y, y)": [wcorr(pred_y[:, j], ys[:, j], w) for j in range(2)],
                             "wcorr(p seg mean, y)": [wcorr(pseg[:, j], ys[:, j], w) for j in range(2)],
                             "wcorr(knn y, resid)": [wcorr(pred_y[:, j], r[sel][:, j], w) for j in range(2)]}
    print(json.dumps(res, default=float), flush=True)

    # DTW kNN on a weight-proportional query subsample vs euclid on the same queries
    if B > 0:
        qs, dt = [], {}
        pdtw = np.zeros((0, 2))
        t0 = time.time()
        exact = []
        for f in range(4):
            cand = np.flatnonzero((fold == f) & (w > 0))
            q = RNG.choice(cand, min(NQ, len(cand)), replace=False, p=w[cand] / w[cand].sum())
            lib = np.concatenate([libs[g] for g in range(4) if g != f])
            nbr, _, ex = knn_dtw(X[q].astype(np.float64), X[lib].astype(np.float64), B, K, 4000)
            exact.append(ex.mean())
            pdtw = np.concatenate([pdtw, wavg(lib[nbr], r[sel], Wseg[sel])])
            qs.append(q)
        q = np.concatenate(qs)
        rq = r[sel][q]
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # sample is weight-proportional: unweighted stats
        res["dtw_subsample"] = {"n": int(len(q)), "seconds": round(time.time() - t0, 1), "exact_share": [float(e) for e in exact],
                                "corr(dtw pred, resid)": [corr(pdtw[:, j], rq[:, j]) for j in range(2)],
                                "corr(euclid pred, resid)": [corr(pred[q, j], rq[:, j]) for j in range(2)],
                                "corr(shuffled pred, resid)": [corr(pred_sh[q, j], rq[:, j]) for j in range(2)],
                                "corr(dtw, euclid)": [corr(pdtw[:, j], pred[q, j]) for j in range(2)]}
        # per-fold consistency of the DTW signal
        fq = fold[q]
        res["dtw_subsample"]["corr_by_fold_D"] = [corr((pdtw[fq == f, 0] - pdtw[fq == f, 1]) / 2, (rq[fq == f, 0] - rq[fq == f, 1]) / 2)
                                                  for f in range(4)]
        res["dtw_subsample"]["euclid_corr_by_fold_D"] = [corr((pred[q][fq == f, 0] - pred[q][fq == f, 1]) / 2,
                                                              (rq[fq == f, 0] - rq[fq == f, 1]) / 2) for f in range(4)]
    (OUT / f"knn_c{cls}_L{L}_B{B}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
