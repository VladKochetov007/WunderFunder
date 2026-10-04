"""DTW streams H1 step 1: are i0 and i1 decoded mid paths related, and does i1 lead?

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited \
        .venv/bin/python scripts/cr_dtw_streams_leadlag.py [N_PER_CLASS]

Classes 1-3 (train-fitted code->tick move tables exist; class 0 has none). i1 is decoded with the
i0 table (codes share one global encoding; unknown i1 transitions contribute 0 ticks and are counted).
Per class: cross-correlation of B-row binned mid returns corr(r0[t], r1[t-k]) for k = -K..K bins
(k > 0: i1 leads), and windowed DTW lag: within 4000-row windows, z-normalised 40-point paths
(100-row bins) are aligned with a Sakoe-Chiba band of 8 bins; the mean (j - i) along the warping
path is the DTW lead of i1 in bins (positive = i1 earlier), next to the Euclidean/DTW distance ratio.
No targets, no predictions: inputs only.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numba as nb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "research/campaign_2026-10-02/dtw/streams"

from wnn.data import SequenceFile  # noqa: E402
from wnn.ticks import reconstruct  # noqa: E402


@nb.njit(cache=True)
def dtw_band(a, b, band):
    """DTW on 1-D series with |i-j| <= band; returns (cost, mean(j-i) along the optimal path)."""
    n, m = len(a), len(b)
    inf = 1e18
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            best = D[i - 1, j - 1]
            if D[i - 1, j] < best:
                best = D[i - 1, j]
            if D[i, j - 1] < best:
                best = D[i, j - 1]
            D[i, j] = c + best
    # backtrack
    i, j = n, m
    s = 0.0
    k = 0
    while i > 0 and j > 0:
        s += j - i
        k += 1
        d0 = D[i - 1, j - 1]
        d1 = D[i - 1, j]
        d2 = D[i, j - 1]
        if d0 <= d1 and d0 <= d2:
            i -= 1
            j -= 1
        elif d1 <= d2:
            i -= 1
        else:
            j -= 1
    return D[n, m], s / max(k, 1)


def decode(x, mv):
    a, b, u0 = reconstruct(x, *mv)
    x1 = x.copy()
    x1[:, 0:22] = x[:, 52:74]
    a1, b1, u1 = reconstruct(x1, *mv)
    ev0 = int((x[1:, :22] != x[:-1, :22]).any(1).sum())
    ev1 = int((x1[1:, :22] != x1[:-1, :22]).any(1).sum())
    return 0.5 * (a + b).astype(float), 0.5 * (a1 + b1).astype(float), u0 / max(ev0, 1), u1 / max(ev1, 1)


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    z = np.load(ROOT / "cache/valid_targets.npz")
    cls, rg = z["market_class"], z["row_group"]
    sf = SequenceFile(ROOT / "datasets/valid.parquet")
    rng = np.random.default_rng(0)
    bins = (20, 100, 500)
    K = 5
    res = {}
    for c in (1, 2, 3):
        mv = pickle.load(open(ROOT / f"cache/class{c}_moves_full.pkl", "rb"))
        idx = rng.choice(np.flatnonzero(cls == c), n_per, replace=False)
        xc = {B: np.zeros((2 * K + 1, 3)) for B in bins}  # sums of r0*r1, r0^2, r1^2 per lag
        unk0, unk1, lags, ratios, ccs = [], [], [], [], []
        for i in idx:
            x = sf.read(int(rg[i]), targets=False, mask=False).features
            m0, m1, u0, u1 = decode(x, mv)
            unk0.append(u0)
            unk1.append(u1)
            for B in bins:
                r0 = np.diff(m0[::B])
                r1 = np.diff(m1[::B])
                r0 = r0 - r0.mean()
                r1 = r1 - r1.mean()
                for li, k in enumerate(range(-K, K + 1)):
                    if k >= 0:
                        a, b = r0[k:], r1[: len(r1) - k]
                    else:
                        a, b = r0[: len(r0) + k], r1[-k:]
                    xc[B][li] += [np.dot(a, b), np.dot(a, a), np.dot(b, b)]
            for w0 in range(0, 20000 - 4000 + 1, 4000):
                p0 = m0[w0:w0 + 4000:100]
                p1 = m1[w0:w0 + 4000:100]
                if p0.std() < 1e-9 or p1.std() < 1e-9:
                    continue
                p0 = (p0 - p0.mean()) / p0.std()
                p1 = (p1 - p1.mean()) / p1.std()
                cost, lag = dtw_band(p0, p1, 8)
                eu = float(((p0 - p1) ** 2).sum())
                lags.append(lag)
                ratios.append(cost / max(eu, 1e-9))
                ccs.append(float(np.corrcoef(p0, p1)[0, 1]))
        out = {"unknown_i0": float(np.mean(unk0)), "unknown_i1": float(np.mean(unk1))}
        for B in bins:
            s = xc[B]
            out[f"xcorr_B{B}"] = {int(k): round(float(s[li, 0] / np.sqrt(s[li, 1] * s[li, 2])), 4)
                                  for li, k in enumerate(range(-K, K + 1))}
        lags = np.array(lags)
        ccs = np.array(ccs)
        out["dtw_window"] = {"n": len(lags), "level_corr_median": round(float(np.median(ccs)), 3),
                             "level_corr_mean": round(float(np.mean(ccs)), 3),
                             "lag_bins_mean": round(float(lags.mean()), 3), "lag_bins_median": round(float(np.median(lags)), 3),
                             "lag_bins_q10_q90": [round(float(q), 2) for q in np.quantile(lags, [0.1, 0.9])],
                             "dtw_over_euclid_median": round(float(np.median(ratios)), 3)}
        res[f"class{c}"] = out
        print(c, json.dumps(out), flush=True)
    (OUT / "leadlag_step1.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
