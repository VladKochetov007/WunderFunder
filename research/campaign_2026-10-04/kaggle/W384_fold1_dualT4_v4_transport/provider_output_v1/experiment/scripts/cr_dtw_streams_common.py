"""Shared helpers for the DTW streams probes: OOF panel sums and delta scoring.

Scoring convention: per-sequence raw |y|-weighted sums (clip +-2) of the unmodified pooled OOF
D256 pair p = (parent + head-FT)/2 on all scored rows; a candidate evaluated on a row subsample is
scored as full sums + (n_scored / n_sampled) * (sampled candidate sums - sampled baseline sums).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "research/campaign_2026-10-02/dtw/streams"
PUBLIC_MIX = np.array([0.183, 0.160, 0.198, 0.459])

from wnn.gmetric import wp_from_sums  # noqa: E402


def sums6(y, p):
    """(..., 6) raw sums over the row axis (axis -1) of clipped y, p."""
    y = np.clip(np.asarray(y, np.float64), -2, 2)
    p = np.clip(np.asarray(p, np.float64), -2, 2)
    w = np.abs(y)
    wy, wp = w * y, w * p
    return np.stack([w.sum(-1), wy.sum(-1), wp.sum(-1), (wy * y).sum(-1), (wp * p).sum(-1), (wy * p).sum(-1)], -1)


def panel():
    """fold (N,), cls (N,), per-sequence full baseline sums (N, 2, 6), n_scored (N,)."""
    from wnn.splits import load_campaign_split
    cache = OUT / "panel_sums.npz"
    z = np.load(ROOT / "cache/valid_targets.npz")
    cls, rg = z["market_class"].astype(int), z["row_group"].astype(int)
    if cache.exists():
        c = np.load(cache)
        return c["fold"], cls, c["S"], c["n"]
    Y, M = z["y"], z["mask"].copy()
    M[:, :99] = False
    pos = {int(g): k for k, g in enumerate(rg)}
    fold = np.full(len(rg), -1)
    for f in range(4):
        sp = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT)
        for g in sp.evaluation_groups:
            fold[pos[g]] = f
    par = [np.load(ROOT / f"cache/pred_c20261001_d256_s3_f{f}_valid.npy", mmap_mode="r") for f in range(4)]
    hft = [np.load(ROOT / f"cache/pred_c20261001_ft_d256_s3_f{f}h_valid.npy", mmap_mode="r") for f in range(4)]
    S = np.zeros((len(rg), 2, 6))
    n = M.sum(1)
    for i in range(len(rg)):
        f = fold[i]
        r = np.flatnonzero(M[i])
        p = 0.5 * (np.asarray(par[f][i][r], np.float32) + np.asarray(hft[f][i][r], np.float32))
        for k in range(2):
            S[i, k] = sums6(Y[i][r, k], p[:, k])
    np.savez(cache, fold=fold, S=S, n=n)
    return fold, cls, S, n


def wp(S, weights=None):
    """Two-target pooled WP of per-sequence sums (N, 2, 6), optional per-sequence weights."""
    t = S if weights is None else S * weights[:, None, None]
    return float(np.mean([wp_from_sums(t[:, k].sum(0)) for k in range(2)]))


def mix_weights(cls, sel):
    share = np.bincount(cls[sel], minlength=4) / max(sel.sum(), 1)
    return (PUBLIC_MIX / np.maximum(share, 1e-12))[cls]


def report(S_base, S_cand, fold, cls):
    """Deltas pooled / per class / public mix / per fold for full-panel sum arrays."""
    allsel = fold >= 0
    out = {"pooled": wp(S_cand[allsel]) - wp(S_base[allsel])}
    w = mix_weights(cls, allsel)
    out["public_mix"] = wp(S_cand[allsel], w[allsel]) - wp(S_base[allsel], w[allsel])
    for c in range(4):
        s = allsel & (cls == c)
        out[f"class{c}"] = wp(S_cand[s]) - wp(S_base[s])
    out["base_pooled"] = wp(S_base[allsel])
    out["base_public_mix"] = wp(S_base[allsel], w[allsel])
    for c in range(4):
        out[f"base_class{c}"] = wp(S_base[allsel & (cls == c)])
    for f in range(4):
        s = fold == f
        wf = mix_weights(cls, s)
        out[f"fold{f}"] = wp(S_cand[s]) - wp(S_base[s])
        out[f"fold{f}_mix"] = wp(S_cand[s], wf[s]) - wp(S_base[s], wf[s])
    return {k: round(v, 6) for k, v in out.items()}
