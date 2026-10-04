"""DTW streams H1 step 2: causal i0/i1 decoded-path features on scored OOF rows (classes 1-3).

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited \
        .venv/bin/python scripts/cr_dtw_streams_build.py [STRIDE]

Every STRIDE-th scored row (row >= 99, random per-sequence offset) of every class-1..3 validation
sequence. Inputs only for features (rows <= t of the same sequence; move tables were fitted on
TRAIN labels); y and the pooled out-of-fold D256 pair prediction p = (parent + head-FT)/2 are stored
for the residual probe. DTW feature: at row t the last W=20 bins (B=20 rows) of the i0 and i1 mid
paths, each z-normalised within the window, are aligned with open-end DTW (band 6); j* is the i1
bin matched to i0's last bin; dtw_pend = m1(t) - m1(end of bin j*) is the i1 move not yet echoed
by i0, dtw_lag = W-1-j* (bins, > 0 = i1 leads).
"""
from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numba as nb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "research/campaign_2026-10-02/dtw/streams"
KS = (20, 100, 500, 2000)
HLS = (200, 2000)
B, W, BAND = 20, 20, 6
FEATS = (["p0", "p1", "since0", "since1"] + [f"r0_{k}" for k in KS] + [f"r1_{k}" for k in KS]
         + ["r1_since0"] + [f"ec_{h}" for h in HLS] + ["unk1", "dtw_pend", "dtw_lag", "dtw_ratio", "dtw_cc"])

_SF = _MV = None


@nb.njit(cache=True)
def open_end_dtw(a, b, band):
    """Align all of a to a prefix of b (a's last point free to match any j); returns (j*, cost, eu)."""
    n, m = len(a), len(b)
    inf = 1e18
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            best = min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            D[i, j] = c + best
    jb = n
    cb = inf
    for j in range(max(1, n - band), min(m, n + band) + 1):
        if D[n, j] < cb:
            cb = D[n, j]
            jb = j
    eu = 0.0
    for i in range(n):
        eu += (a[i] - b[i]) ** 2
    return jb - 1, cb, eu


@nb.njit(cache=True)
def dtw_feats(m0, m1, rows, B, W, band):
    out = np.zeros((len(rows), 4))
    for r in range(len(rows)):
        t = rows[r]
        if t < B * W:
            continue
        # bins end at t, t-B, ...: bin value = mid at the bin end
        a = np.empty(W)
        b = np.empty(W)
        for k in range(W):
            a[k] = m0[t - (W - 1 - k) * B]
            b[k] = m1[t - (W - 1 - k) * B]
        sa = a.std()
        sb = b.std()
        if sa < 1e-9 or sb < 1e-9:
            continue
        az = (a - a.mean()) / sa
        bz = (b - b.mean()) / sb
        j, cost, eu = open_end_dtw(az, bz, band)
        out[r, 0] = m1[t] - b[j]
        out[r, 1] = W - 1 - j
        out[r, 2] = cost / max(eu, 1e-9)
        out[r, 3] = ((az * bz).mean())
    return out


def _ema(x, hl):
    a = 1.0 - 0.5 ** (1.0 / hl)
    out = np.empty_like(x)
    s = x[0]
    for i in range(len(x)):
        s += a * (x[i] - s)
        out[i] = s
    return out


_ema_nb = nb.njit(cache=True)(_ema)


def _init():
    global _SF, _MV
    from wnn.data import SequenceFile
    _SF = SequenceFile(ROOT / "datasets/valid.parquet")
    _MV = {c: pickle.load(open(ROOT / f"cache/class{c}_moves_full.pkl", "rb")) for c in (1, 2, 3)}


def _one(args):
    from wnn.ticks import reconstruct
    i, g, c, rows = args
    x = _SF.read(int(g), targets=False, mask=False).features
    a, b, _ = reconstruct(x, *_MV[c])
    x1 = x.copy()
    x1[:, 0:22] = x[:, 52:74]
    a1, b1, _ = reconstruct(x1, *_MV[c])
    m0 = 0.5 * (a + b).astype(np.float64)
    m1 = 0.5 * (a1 + b1).astype(np.float64)
    T = len(x)
    t = np.arange(T)
    ev0 = np.r_[False, (x[1:, :22] != x[:-1, :22]).any(1)]
    ev1 = np.r_[False, (x1[1:, :22] != x1[:-1, :22]).any(1)]
    last0 = np.maximum.accumulate(np.where(ev0, t, 0))
    last1 = np.maximum.accumulate(np.where(ev1, t, 0))
    # unknown i1 transitions so far (quality of the i1 path)
    e1 = np.flatnonzero(ev1)
    unk = np.zeros(T)
    if len(e1):
        keys = [tuple(r) for r in np.round(x1[np.r_[e1 - 1, e1], :22], 4)]
        n = len(e1)
        unk[e1] = [(keys[k], keys[n + k]) not in _MV[c][0] for k in range(n)]
    unk = np.cumsum(unk) / np.maximum(np.cumsum(ev1), 1)
    mch = np.r_[False, np.diff(m0) != 0]
    lastm = np.maximum.accumulate(np.where(mch, t, 0))
    d = m0 - m1
    F = np.zeros((len(rows), len(FEATS)), np.float32)
    F[:, 2] = rows - last0[rows]
    F[:, 3] = rows - last1[rows]
    col = 4
    for m in (m0, m1):
        for k in KS:
            F[:, col] = m[rows] - m[np.maximum(rows - k, 0)]
            col += 1
    F[:, col] = m1[rows] - m1[lastm[rows]]
    col += 1
    for h in HLS:
        F[:, col] = (d - _ema_nb(d, float(h)))[rows]
        col += 1
    F[:, col] = unk[rows]
    col += 1
    F[:, col:col + 4] = dtw_feats(m0, m1, rows, B, W, BAND)
    return i, F


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    from wnn.splits import load_campaign_split
    z = np.load(ROOT / "cache/valid_targets.npz")
    Y, MASK, CLS, RG = z["y"], z["mask"].copy(), z["market_class"].astype(int), z["row_group"].astype(int)
    MASK[:, :99] = False
    pos = {int(g): k for k, g in enumerate(RG)}
    fold = np.full(len(RG), -1)
    for f in range(4):
        sp = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT)
        for g in sp.evaluation_groups:
            fold[pos[g]] = f
    par = [np.load(ROOT / f"cache/pred_c20261001_d256_s3_f{f}_valid.npy", mmap_mode="r") for f in range(4)]
    hft = [np.load(ROOT / f"cache/pred_c20261001_ft_d256_s3_f{f}h_valid.npy", mmap_mode="r") for f in range(4)]
    rng = np.random.default_rng(1)
    jobs, meta = [], []
    for i in np.flatnonzero((CLS >= 1) & (fold >= 0)):
        rows = np.flatnonzero(MASK[i])
        rows = rows[rng.integers(stride)::stride]
        if len(rows) == 0:
            continue
        jobs.append((int(i), int(RG[i]), int(CLS[i]), rows))
    print("sequences", len(jobs), "rows", sum(len(j[3]) for j in jobs), "unassigned", int((fold < 0).sum()), flush=True)
    parts = {}
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "2")), initializer=_init) as ex:
        for n, (i, F) in enumerate(ex.map(_one, jobs, chunksize=4)):
            parts[i] = F
            if n % 200 == 0:
                print("done", n, flush=True)
    feats, ys, ps, seq, rr = [], [], [], [], []
    for i, g, c, rows in jobs:
        f = fold[i]
        p = 0.5 * (np.asarray(par[f][i][rows], np.float32) + np.asarray(hft[f][i][rows], np.float32))
        F = parts[i]
        F[:, 0:2] = p
        feats.append(F)
        ys.append(Y[i][rows])
        ps.append(p)
        seq.append(np.full(len(rows), i, np.int32))
        rr.append(rows.astype(np.int32))
    np.savez(OUT / f"h1_rows_s{stride}.npz", X=np.concatenate(feats), y=np.concatenate(ys), p=np.concatenate(ps),
             seq=np.concatenate(seq), row=np.concatenate(rr), cls=CLS, fold=fold, names=np.array(FEATS))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
