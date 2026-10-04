"""DTW streams H3: are class-3 quiet runs time-warped copies of each other? (TRAIN, descriptive)

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited \
        .venv/bin/python scripts/cr_dtw_streams_warp.py [N_SEQ]

N_SEQ (<= 300) class-3 TRAIN sequences. Runs = rows between consecutive target changes (training-time
use of labels for segmentation only; equivalently a6 endpoint rows), 64-1024 rows long, row >= 99.
Per run: cumulative-fraction curves of i0 book-volume events and i0 trade events on 32 points of
normalised row time. (a) mean curve vs the homogeneous-Poisson diagonal; (b) pairwise DTW (band 4)
/ Euclidean distance ratio for real runs vs Poisson runs with the same event counts; (c) does
early intensity predict the run length (time-warp = duration ~ 1/speed)? corr(log len, log events
in first 32 rows); (d) does the run's warp (front-loading of book events, causal after 64 rows)
relate to the run's target direction? weighted corr with yD.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numba as nb
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/campaign_2026-10-02/dtw/streams"
NP = 32


@nb.njit(cache=True)
def dtw1(a, b, band):
    n = len(a)
    D = np.full((n + 1, n + 1), 1e18)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(n, i + band) + 1):
            D[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[n, n]


def curve(ev):
    c = np.cumsum(ev).astype(float)
    if c[-1] <= 0:
        return None
    u = np.linspace(0, len(ev) - 1, NP).astype(int)
    return c[u] / c[-1]


def main():
    n_seq = min(int(sys.argv[1]) if len(sys.argv) > 1 else 150, 300)
    cls = np.load(ROOT / "research/artifacts/market-classes-v1.npz")["train"]
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(cls == 3), n_seq, replace=False)
    pf = pq.ParquetFile(ROOT / "datasets/train.parquet")
    cols = [f"i0_v{k}" for k in range(22)] + [f"i0_dv{k}" for k in range(4)] + ["t0", "t1"]
    cb, ct, lens, early, front, yd = [], [], [], [], [], []
    nb_ev, nt_ev = [], []
    for g in idx:
        t = pf.read_row_group(int(g), columns=cols, use_threads=False)
        x = np.stack([t.column(c).to_numpy() for c in cols], 1)
        y = x[:, 26:28]
        evb = np.r_[0, (x[1:, 0:22] != x[:-1, 0:22]).any(1)]
        evt = np.r_[0, (x[1:, 22:26] != x[:-1, 22:26]).any(1)]
        chg = np.flatnonzero(np.r_[False, (y[1:] != y[:-1]).any(1)])
        for s, e in zip(chg[:-1], chg[1:]):
            L = e - s
            if s < 99 or L < 64 or L > 1024:
                continue
            c1, c2 = curve(evb[s + 1:e + 1]), curve(evt[s + 1:e + 1])
            if c1 is None or c2 is None:
                continue
            cb.append(c1)
            ct.append(c2)
            nb_ev.append(int(evb[s + 1:e + 1].sum()))
            nt_ev.append(int(evt[s + 1:e + 1].sum()))
            lens.append(L)
            early.append(evb[s + 1:s + 33].sum())
            # causal warp statistic at row s+64: share of the first 64 rows' book events in rows 1-16
            front.append(evb[s + 1:s + 17].sum() / max(evb[s + 1:s + 65].sum(), 1))
            yd.append((np.clip(y[s + 1, 0], -2, 2) - np.clip(y[s + 1, 1], -2, 2)) / 2)
    cb, ct = np.array(cb), np.array(ct)
    lens, early, front, yd = map(np.array, (lens, early, front, yd))
    u = np.linspace(0, 1, NP)
    res = {"n_runs": int(len(lens)), "median_len": float(np.median(lens)),
           "book_curve_mean_minus_diagonal_at_u": {f"{v:.2f}": round(float(m), 3) for v, m in
                                                    zip(u[::6], (cb.mean(0) - u)[::6])},
           "trade_curve_mean_minus_diagonal_at_u": {f"{v:.2f}": round(float(m), 3) for v, m in
                                                     zip(u[::6], (ct.mean(0) - u)[::6])}}
    # pairwise DTW/Euclid ratios: real vs Poisson-uniform event times with the same counts
    pairs = rng.integers(0, len(lens), size=(3000, 2))
    def ratios(C):
        r, eu = [], []
        for a, b in pairs:
            if a == b:
                continue
            e = float(((C[a] - C[b]) ** 2).sum())
            if e < 1e-9:
                continue
            r.append(dtw1(C[a], C[b], 4) / e)
            eu.append(e)
        return np.array(r), np.array(eu)
    sim = []
    for L, n in zip(lens, nb_ev):
        ev = np.zeros(L)
        ev[rng.choice(L, size=min(n, L), replace=False)] = 1
        sim.append(curve(ev))
    sim = np.array(sim)
    rr, er = ratios(cb)
    rs, es = ratios(sim)
    res["book_dtw_over_euclid_median_real"] = round(float(np.median(rr)), 3)
    res["book_dtw_over_euclid_median_poisson"] = round(float(np.median(rs)), 3)
    res["book_euclid_median_real_vs_poisson"] = [round(float(np.median(er)), 3), round(float(np.median(es)), 3)]
    res["corr_loglen_logearly32"] = round(float(np.corrcoef(np.log(lens), np.log1p(early))[0, 1]), 3)
    res["corr_loglen_log_book_events_total"] = round(float(np.corrcoef(np.log(lens), np.log1p(nb_ev))[0, 1]), 3)
    w = np.abs(yd)
    nz = w > 0
    def wc(a, b, ww):
        a = a - np.average(a, weights=ww)
        b = b - np.average(b, weights=ww)
        return float((ww * a * b).sum() / np.sqrt((ww * a * a).sum() * (ww * b * b).sum()))
    res["wcorr_front_loading_yD"] = round(wc(front[nz], yd[nz], w[nz]), 4)
    res["wcorr_front_loading_absyD"] = round(float(np.corrcoef(front, np.abs(yd))[0, 1]), 4)
    res["share_runs_yD_zero"] = round(float((~nz).mean()), 3)
    print(json.dumps(res, indent=1))
    (OUT / "h3_warp.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
