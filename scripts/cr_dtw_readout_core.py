"""Shared panel loading, segment features and WP bookkeeping for the dtw/readout probes.

Segments are causal: a new segment starts at row 0 and at every row with E (a6 == lower endpoint, observed at that
row). Feature columns (per row t, per target) use only rows of the current segment <= t.
"""

from __future__ import annotations

import os
from pathlib import Path

import numba as nb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANEL = Path(os.environ.get("CR_DTW_PANEL", "/tmp/claude-1000/-home-vlad-development-wunder-wnn-connectome-starterpack/"
                            "7ea48a59-c833-4cf5-b1fe-467c6d7c1ac5/scratchpad/panel"))
OUT = ROOT / "research/campaign_2026-10-02/dtw/readout"
PUBLIC_MIX = np.array([0.183, 0.160, 0.198, 0.459])
NB_BUCKET = 13

FEATS = ["p", "runmean", "hold", "sema.5", "sema.2", "sema.05", "sema.01", "p_lag1", "p_lag8", "par", "par_runmean", "one"]
NF = len(FEATS)


def load_panel():
    with np.load(ROOT / "cache/valid_targets.npz") as z:
        y, mask = z["y"], z["mask"]
    meta = np.load(PANEL / "meta.npz")
    ft = np.asarray(np.load(PANEL / "ft.npy", mmap_mode="r"))
    par = np.asarray(np.load(PANEL / "par.npy", mmap_mode="r"))
    E = np.asarray(np.load(PANEL / "E.npy", mmap_mode="r"))
    return y, mask, ft, par, E, meta["fold"].astype(np.int64), meta["cls"].astype(np.int64)


@nb.njit(cache=True)
def bucket_of(k):
    b = 0
    k1 = k + 1
    while k1 > 1 and b < NB_BUCKET - 1:
        k1 >>= 1
        b += 1
    return b


@nb.njit(cache=True)
def seq_features(p, q, E, F, B):
    """p, q: (T,) head-FT and parent predictions of one target. Fills F (T, NF) and elapsed bucket B (T,)."""
    T = p.shape[0]
    start = 0
    s_p = 0.0
    s_q = 0.0
    h = 0.0
    e1 = e2 = e3 = e4 = 0.0
    for t in range(T):
        x = p[t]
        if t == 0 or E[t]:
            start = t
            s_p = 0.0
            s_q = 0.0
            h = x
            e1 = e2 = e3 = e4 = x
        else:
            e1 = 0.5 * x + 0.5 * e1
            e2 = 0.2 * x + 0.8 * e2
            e3 = 0.05 * x + 0.95 * e3
            e4 = 0.01 * x + 0.99 * e4
        s_p += x
        s_q += q[t]
        n = t - start + 1
        F[t, 0] = x
        F[t, 1] = s_p / n
        F[t, 2] = h
        F[t, 3] = e1
        F[t, 4] = e2
        F[t, 5] = e3
        F[t, 6] = e4
        F[t, 7] = p[t - 1] if t - 1 >= start else x
        F[t, 8] = p[max(t - 8, start)]
        F[t, 9] = q[t]
        F[t, 10] = s_q / n
        F[t, 11] = 1.0
        B[t] = bucket_of(t - start)


@nb.njit(parallel=True, cache=True)
def accumulate_stats(y, mask, ft, par, E, fold, cls):
    """|y|-weighted normal equations per (fold, class, bucket, target): XtWX (NF, NF), XtWy (NF)."""
    S, T = mask.shape
    xtx = np.zeros((S, NB_BUCKET, 2, NF, NF))
    xty = np.zeros((S, NB_BUCKET, 2, NF))
    for i in nb.prange(S):
        F = np.empty((T, NF))
        B = np.empty(T, np.int64)
        for j in range(2):
            seq_features(ft[i, :, j], par[i, :, j], E[i], F, B)
            for t in range(T):
                if not mask[i, t]:
                    continue
                yy = min(max(y[i, t, j], -2.0), 2.0)
                w = abs(yy)
                if w == 0.0:
                    continue
                b = B[t]
                for a in range(NF):
                    fa = w * F[t, a]
                    xty[i, b, j, a] += fa * yy
                    for c in range(NF):
                        xtx[i, b, j, a, c] += fa * F[t, c]
    nfold = fold.max() + 1
    XtX = np.zeros((nfold, 4, NB_BUCKET, 2, NF, NF))
    Xty = np.zeros((nfold, 4, NB_BUCKET, 2, NF))
    for i in range(S):
        XtX[fold[i], cls[i]] += xtx[i]
        Xty[fold[i], cls[i]] += xty[i]
    return XtX, Xty


@nb.njit(parallel=True, cache=True)
def apply_models(y, mask, ft, par, E, fold, cls, coef):
    """coef (nfold, M, 4, NB_BUCKET, 2, NF). Returns per-sequence raw WP sums (S, M, 2, 6) of clipped outputs."""
    S, T = mask.shape
    M = coef.shape[1]
    out = np.zeros((S, M, 2, 6))
    for i in nb.prange(S):
        F = np.empty((T, NF))
        B = np.empty(T, np.int64)
        f = fold[i]
        c = cls[i]
        for j in range(2):
            seq_features(ft[i, :, j], par[i, :, j], E[i], F, B)
            for t in range(T):
                if not mask[i, t]:
                    continue
                yy = min(max(y[i, t, j], -2.0), 2.0)
                w = abs(yy)
                b = B[t]
                for m in range(M):
                    v = 0.0
                    for a in range(NF):
                        v += coef[f, m, c, b, j, a] * F[t, a]
                    v = min(max(v, -2.0), 2.0)
                    o = out[i, m, j]
                    o[0] += w
                    o[1] += w * yy
                    o[2] += w * v
                    o[3] += w * yy * yy
                    o[4] += w * v * v
                    o[5] += w * yy * v
    return out


def wp(s):
    """s (..., 2, 6) raw sums -> mean over targets of weighted Pearson."""
    n, sy, sp, syy, spp, syp = np.moveaxis(np.asarray(s, np.float64), -1, 0)
    r = (syp / n - sy * sp / n**2) / np.sqrt((syy / n - (sy / n) ** 2) * (spp / n - (sp / n) ** 2))
    return r.mean(-1)


def summarize(seq_sums, fold, cls):
    """seq_sums (S, M, 2, 6) -> dict of WP arrays (M,): pooled, public mix, per class, per fold."""
    share = np.bincount(cls, minlength=4) / len(cls)
    wseq = (PUBLIC_MIX / share)[cls]
    res = {"pooled": wp(seq_sums.sum(0)), "public": wp((seq_sums * wseq[:, None, None, None]).sum(0))}
    for c in range(4):
        res[f"c{c}"] = wp(seq_sums[cls == c].sum(0))
    for k in range(fold.max() + 1):
        res[f"f{k}"] = wp(seq_sums[fold == k].sum(0))
        res[f"f{k}pub"] = wp((seq_sums * wseq[:, None, None, None])[fold == k].sum(0))
    return res
