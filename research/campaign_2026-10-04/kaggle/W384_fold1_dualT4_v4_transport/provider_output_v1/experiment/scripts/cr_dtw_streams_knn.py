"""DTW streams H2: DTW-kNN retrieval of per-sequence calibration (scale) from so-far stream paths.

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited \
        .venv/bin/python scripts/cr_dtw_streams_knn.py

Query for sequence i at checkpoint K: rows [0, K) only (causal), C channels binned in 100-row bins:
i0 price-code event rate, i1 price-code event rate, i0 trade rate, i1 trade rate, a2, a3, mean |pD|
and mean |pS| of the OOF prediction itself (available online). Channels are standardised with
global constants. Library: same-class sequences of the other three folds (never the evaluated
fold); their label = realised per-sequence scale beta_j = sum w y p / sum w p^2 on rows >= K
(both targets pooled), relative to their class-fold scale. Prediction: median log ratio of the k
nearest neighbours under multivariate DTW (Sakoe-Chiba band 3 bins) or Euclidean distance.
Candidate p' = p * exp(lam * pred) on rows >= K. Also reports the oracle (realised beta_i) and
corr(pred, realised) across sequences.
"""
from __future__ import annotations

import json
import sys

import numba as nb
import numpy as np
import pyarrow.parquet as pq

from cr_dtw_streams_common import OUT, ROOT, panel, report

KS = (2000, 5000)
BIN = 100
NN = (10, 30, 100)
LAMS = (0.5, 1.0)


@nb.njit(cache=True)
def mdtw(a, b, band):
    n = a.shape[0]
    D = np.full((n + 1, n + 1), 1e18)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(n, i + band) + 1):
            c = 0.0
            for q in range(a.shape[1]):
                c += (a[i - 1, q] - b[j - 1, q]) ** 2
            D[i, j] = c + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[n, n]


@nb.njit(cache=True)
def all_pairs(Q, band, use_dtw):
    n = Q.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if use_dtw:
                d = mdtw(Q[i], Q[j], band)
            else:
                d = ((Q[i] - Q[j]) ** 2).sum()
            out[i, j] = d
            out[j, i] = d
    return out


def channels(Kmax):
    """(N, Kmax//BIN, 8) binned stream channels for all valid sequences, rows < Kmax."""
    cache = OUT / f"knn_channels_{Kmax}.npy"
    if cache.exists():
        return np.load(cache)
    z = np.load(ROOT / "cache/valid_targets.npz")
    rg = z["row_group"].astype(int)
    fold, cls, _, _ = panel()
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    cols = ([f"i0_p{k}" for k in range(22)] + [f"i1_p{k}" for k in range(22)] + [f"i0_dv{k}" for k in range(4)]
            + [f"i1_dv{k}" for k in range(4)] + ["a2", "a3"])
    par = [np.load(ROOT / f"cache/pred_c20261001_d256_s3_f{f}_valid.npy", mmap_mode="r") for f in range(4)]
    hft = [np.load(ROOT / f"cache/pred_c20261001_ft_d256_s3_f{f}h_valid.npy", mmap_mode="r") for f in range(4)]
    nb_ = Kmax // BIN
    C = np.zeros((len(rg), nb_, 8), np.float32)
    for i in range(len(rg)):
        t = pf.read_row_group(int(rg[i]), columns=cols, use_threads=False)
        x = np.stack([t.column(c).to_numpy()[:Kmax] for c in cols], 1)
        ev0 = np.r_[0, (x[1:, 0:22] != x[:-1, 0:22]).any(1)]
        ev1 = np.r_[0, (x[1:, 22:44] != x[:-1, 22:44]).any(1)]
        tr0 = np.r_[0, (x[1:, 44:48] != x[:-1, 44:48]).any(1)]
        tr1 = np.r_[0, (x[1:, 48:52] != x[:-1, 48:52]).any(1)]
        f = fold[i]
        p = 0.5 * (np.asarray(par[f][i][:Kmax], np.float32) + np.asarray(hft[f][i][:Kmax], np.float32))
        pd_ = np.abs(p[:, 0] - p[:, 1]) / 2
        ps_ = np.abs(p[:, 0] + p[:, 1]) / 2
        pd_[:99] = np.nan
        ps_[:99] = np.nan
        for q, v in enumerate((ev0, ev1, tr0, tr1, x[:, 52], x[:, 53], pd_, ps_)):
            C[i, :, q] = np.nanmean(np.asarray(v, np.float64).reshape(nb_, BIN), 1) if q < 6 else \
                np.array([np.nanmean(s) if np.isfinite(s).any() else np.nan for s in np.asarray(v).reshape(nb_, BIN)])
    # first bin of |p| channels is warm-up only (NaN): fill with the next bin
    for q in (6, 7):
        C[:, 0, q] = C[:, 1, q]
    np.save(cache, C)
    return C


def tail_sums(K):
    """Per-sequence sums (N, 2, 6) of baseline p on scored rows >= K and rows < K."""
    cache = OUT / f"knn_tail_{K}.npz"
    if cache.exists():
        c = np.load(cache)
        return c["hi"], c["lo"]
    from cr_dtw_streams_common import sums6
    z = np.load(ROOT / "cache/valid_targets.npz")
    Y, M = z["y"], z["mask"].copy()
    M[:, :99] = False
    fold, cls, _, _ = panel()
    par = [np.load(ROOT / f"cache/pred_c20261001_d256_s3_f{f}_valid.npy", mmap_mode="r") for f in range(4)]
    hft = [np.load(ROOT / f"cache/pred_c20261001_ft_d256_s3_f{f}h_valid.npy", mmap_mode="r") for f in range(4)]
    hi = np.zeros((len(cls), 2, 6))
    lo = np.zeros((len(cls), 2, 6))
    for i in range(len(cls)):
        f = fold[i]
        r = np.flatnonzero(M[i])
        p = 0.5 * (np.asarray(par[f][i][r], np.float32) + np.asarray(hft[f][i][r], np.float32))
        sel = r >= K
        for k in range(2):
            hi[i, k] = sums6(Y[i][r[sel], k], p[sel, k])
            lo[i, k] = sums6(Y[i][r[~sel], k], p[~sel, k])
    np.savez(cache, hi=hi, lo=lo)
    return hi, lo


def scaled(S, s):
    """Sums of (y, s*p) from sums of (y, p): wp -> s wp, wpp -> s^2 wpp, wyp -> s wyp (no clipping change)."""
    out = S.copy()
    out[..., 2] *= s[:, None]
    out[..., 4] *= (s ** 2)[:, None]
    out[..., 5] *= s[:, None]
    return out


def main():
    fold, cls, S_full, _ = panel()
    C = channels(max(KS))
    flat = C.reshape(-1, C.shape[2])
    mu, sd = np.nanmean(flat, 0), np.nanstd(flat, 0) + 1e-9  # global constants (no labels)
    C = np.nan_to_num((C - mu) / sd)
    res = {}
    for K in KS:
        hi, lo = tail_sums(K)
        # uncentred |y|-weighted LS slope of y on p (both targets pooled), relative to the class slope:
        # the WP-relevant per-sequence multiplier (centred slopes drop the sequence's own D level)
        num, den = hi[:, :, 5].sum(1), hi[:, :, 4].sum(1)
        ok = (den > 1e-6) & (hi[:, :, 0].sum(1) > 0)
        cb = np.array([num[(cls == c) & ok].sum() / den[(cls == c) & ok].sum() for c in range(4)])
        beta = np.where(ok, num / np.maximum(den, 1e-9) / cb[cls], 1.0)
        lb = np.log(np.clip(beta, 0.05, 20))
        Q = np.ascontiguousarray(C[:, : K // BIN])
        Qm = np.ascontiguousarray(Q.mean(1, keepdims=True))
        D = {"dtw": all_pairs(Q, 3, True), "euc": all_pairs(Q, 3, False), "mean": all_pairs(Qm, 0, False)}
        S_base = lo + hi
        # oracle: realised per-sequence scale on rows >= K
        res[f"K{K}_oracle"] = report(S_base, lo + scaled(hi, np.clip(beta, 0.0, 5.0)), fold, cls)
        print(K, "oracle", res[f"K{K}_oracle"]["pooled"], res[f"K{K}_oracle"]["public_mix"], flush=True)
        for dname, Dm in D.items():
            for nn in NN:
                pred = np.zeros(len(cls))
                for i in range(len(cls)):
                    lib = np.flatnonzero((cls == cls[i]) & (fold != fold[i]) & ok)
                    # relative to the library's class median (the class scale is already calibrated)
                    nbr = lib[np.argsort(Dm[i, lib])[:nn]]
                    pred[i] = np.median(lb[nbr]) - np.median(lb[lib])
                rel = lb - np.array([np.median(lb[(cls == cls[i]) & ok]) for i in range(len(cls))])
                cc = [float(np.corrcoef(pred[(cls == c) & ok], rel[(cls == c) & ok])[0, 1]) for c in range(4)]
                for lam in LAMS:
                    s = np.exp(lam * pred)
                    r = report(S_base, lo + scaled(hi, s), fold, cls)
                    r["corr_pred_realised_by_class"] = [round(v, 4) for v in cc]
                    res[f"K{K}_{dname}_nn{nn}_lam{lam}"] = r
                    print(K, dname, nn, lam, json.dumps(r), flush=True)
    (OUT / "h2_knn.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
