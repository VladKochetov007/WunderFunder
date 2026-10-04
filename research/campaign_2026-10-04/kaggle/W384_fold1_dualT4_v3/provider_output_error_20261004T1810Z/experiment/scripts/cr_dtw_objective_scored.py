"""DTW round / objective agent, second pass: are model outputs on UNSCORED rows shrunk relative to scored rows, and do
forward-in-segment teacher targets stay unbiased when restricted to scored (or propensity-weighted) rows?

Training-side quantities only (they read future rows of the label segment and the scored mask / propensity).
Writes research/campaign_2026-10-02/dtw/objective/scored_sums.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-02/dtw/objective"
T = 20000
WARMUP = 99
FEATURES = ["one", "y", "P", "TE", "TE_fwd", "TE_fwd_sc", "TE_td_sc", "TE_fwd_prop", "TE_fwd_prop_sc", "TE_td"]


@numba.njit(cache=True)
def forward_weighted(x, wt, start, fallback):
    """Label segments: wt-weighted mean of x over [t, seg end] and x at the last row of [t, end] with wt > 0;
    rows whose remaining segment has zero weight fall back to fallback[t]."""
    n = x.shape[0]
    fwd = np.empty_like(x)
    last = np.empty_like(x)
    s = 0.0
    c = 0.0
    lv = 0.0
    have = False
    for t in range(n - 1, -1, -1):
        if t == n - 1 or start[t + 1]:
            s = 0.0
            c = 0.0
            have = False
        s += wt[t] * x[t]
        c += wt[t]
        if wt[t] > 0 and not have:
            lv = x[t]
            have = True
        fwd[t] = s / c if c > 0 else fallback[t]
        last[t] = lv if have else fallback[t]
    return fwd, last


def main():
    z = np.load(ROOT / "cache/valid_targets.npz")
    Y, MASK = z["y"], z["mask"].copy()
    MASK[:, :WARMUP] = False
    CLS = z["market_class"].astype(int)
    rg = z["row_group"].astype(int)
    pos = {int(g): i for i, g in enumerate(rg)}
    FOLD = np.full(len(rg), -1)
    for f in range(4):
        sp = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT)
        for g in sp.evaluation_groups:
            FOLD[pos[g]] = f

    def load(name):
        return np.load(ROOT / f"cache/pred_{name}_valid.npy", mmap_mode="r")

    par = [load(f"c20261001_d256_s3_f{f}") for f in range(4)]
    hft = [load(f"c20261001_ft_d256_s3_f{f}h") for f in range(4)]
    ts3 = [load(f"c20260929_since_s3_f{f}") for f in range(4)]
    ts4 = [load(f"c20260929_since_s4_f{f}") for f in range(4)]
    prop = [np.load(ROOT / f"cache/propcache_c20260929_prop_f{f}_valid.npy", mmap_mode="r") for f in range(4)]
    n, d = len(Y), len(FEATURES)
    M = np.zeros((n, 2, d, d))
    # scored (1) vs unscored (0) rows >= warm-up, |y| weights: [class, scored, k, model(TE,P), stats w wy wx wxx wyx wyy]
    SU = np.zeros((4, 2, 2, 2, 6))
    for i in range(n):
        f, c = FOLD[i], CLS[i]
        y = np.clip(np.asarray(Y[i], np.float64), -2, 2)
        m = MASK[i]
        live = np.arange(T) >= WARMUP
        P = np.clip(0.5 * (np.asarray(par[f][i], np.float64) + np.asarray(hft[f][i], np.float64)), -2, 2)
        TE = np.clip(0.5 * (np.asarray(ts3[f][i], np.float64) + np.asarray(ts4[f][i], np.float64)), -2, 2)
        pi2 = np.asarray(prop[f][rg[i]], np.float64) ** 2 + 0.03  # the student's main-loss row weight
        pi2[~live] = 0.0
        lab_start = np.zeros(T, bool)
        lab_start[0] = True
        lab_start[1:] = (Y[i][1:] != Y[i][:-1]).any(-1)
        for k in range(2):
            yk, Pk, Tk = y[:, k], P[:, k], TE[:, k]
            fwd, td = forward_weighted(Tk, live.astype(np.float64), lab_start, Tk)
            fwd_sc, td_sc = forward_weighted(Tk, m.astype(np.float64), lab_start, Tk)
            fwd_pr, _ = forward_weighted(Tk, pi2, lab_start, Tk)
            fwd_prs, _ = forward_weighted(Tk, pi2 * m + 1e-12 * live, lab_start, Tk)
            feats = np.stack([np.ones(T), yk, Pk, Tk, fwd, fwd_sc, td_sc, fwd_pr, fwd_prs, td], 1)[m]
            w = np.abs(yk[m])
            M[i, k] = (feats * w[:, None]).T @ feats
            for sc in (0, 1):
                sel = live & (m if sc else ~m)
                ww, yy = np.abs(yk[sel]), yk[sel]
                for j, x in enumerate((Tk[sel], Pk[sel])):
                    SU[c, sc, k, j] += [ww.sum(), (ww * yy).sum(), (ww * x).sum(), (ww * x * x).sum(),
                                        (ww * yy * x).sum(), (ww * yy * yy).sum()]
        if i % 300 == 0:
            print(i, flush=True)
    np.savez(OUT / "scored_sums.npz", M=M, SU=SU, CLS=CLS, FOLD=FOLD, features=np.array(FEATURES))
    print("done")


if __name__ == "__main__":
    main()
