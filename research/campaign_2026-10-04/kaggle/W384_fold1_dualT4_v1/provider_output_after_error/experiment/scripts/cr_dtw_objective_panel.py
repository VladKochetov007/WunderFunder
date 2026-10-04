"""DTW round / objective agent: one pass over the pooled OOF panel (all 1873 valid sequences, fold k's evaluation
sequences scored with fold k's models) that stores everything the objective probes need as additive sums.

Per sequence and target it accumulates the |y|-weighted second-moment matrix of a feature vector z (scored rows,
step >= 99, clip +-2), so the pooled WP of ANY linear combination of features (fixed or cross-fitted per fold/class)
is a quadratic form of these matrices. Features mix causal inference-time variants of the model output P (event-reset
hold / running mean / reset EMA) with training-target candidates built from the OOF teacher TE and the label's own
future (segment-end / forward-mean / whole-segment-mean teacher values: training-side only, they use future rows).

Also accumulated: binned sums for the H-059 bias (class x since-bin x teacher-bin, with g = exp(-u/32), u = rows until
the label changes), WP sums by rows-since-event (response profile), the teachers' in-sample vs out-of-fold WP gap,
and a fold-0 comparison of the since parent vs distilled alpha 0.5 / 0.7 students.

    WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice scripts/limited .venv/bin/python \
        scripts/cr_dtw_objective_panel.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numba
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-02/dtw/objective"
T = 20000
WARMUP = 99
SINCE_EDGES = np.array([0, 2, 8, 32, 128, 512, 1 << 30])
U_EDGES = np.array([0, 2, 8, 32, 128, 512, 1 << 30])
TB_MULT = np.array([-np.inf, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, np.inf])
FEATURES = ["one", "y", "P", "P_hold", "P_run", "P_ema3", "P_ema6", "P_ema9", "P_segmean_hs", "P_fwd", "P_td",
            "TE", "TE_fwd", "TE_td", "TE_seg", "P_prev_seg_end", "P_segmean_sc", "TE_segmean_sc"]


@numba.njit(cache=True)
def reset_ema(x, start, beta):
    out = np.empty_like(x)
    for t in range(x.shape[0]):
        out[t] = x[t] if start[t] else beta * out[t - 1] + (1 - beta) * x[t]
    return out


@numba.njit(cache=True)
def hold_run(x, start):
    """Value at the segment's first row and running mean since it (both causal for event segments)."""
    hold = np.empty_like(x)
    run = np.empty_like(x)
    s = 0.0
    n = 0
    for t in range(x.shape[0]):
        if start[t]:
            s = 0.0
            n = 0
            hold[t] = x[t]
        else:
            hold[t] = hold[t - 1]
        s += x[t]
        n += 1
        run[t] = s / n
    return hold, run


@numba.njit(cache=True)
def forward_stats(x, start):
    """Label segments (training side, uses future rows): mean over [t, seg end], value at seg end, whole-seg mean."""
    n = x.shape[0]
    fwd = np.empty_like(x)
    end = np.empty_like(x)
    seg = np.empty_like(x)
    s = 0.0
    c = 0
    last = x[n - 1]
    total = 0.0
    tot_n = 0
    t = n - 1
    seg_hi = n - 1
    while t >= 0:
        if t == n - 1 or start[t + 1]:
            s = 0.0
            c = 0
            last = x[t]
        s += x[t]
        c += 1
        fwd[t] = s / c
        end[t] = last
        if start[t]:
            m = s / c
            for v in range(t, seg_hi + 1):
                seg[v] = m
            seg_hi = t - 1
        t -= 1
    return fwd, end, seg


@numba.njit(cache=True)
def prev_seg_end(x, start):
    """Event segments: the model's value at the last row of the previous segment (causal), x[0] before any event."""
    out = np.empty_like(x)
    cur = x[0]
    for t in range(x.shape[0]):
        if start[t] and t > 0:
            cur = x[t - 1]
        out[t] = cur
    return out


def scored_seg_mean(x, ids, m):
    """Hindsight: mean of x over the scored rows of each label segment (the metric's own segment mean)."""
    num = np.bincount(ids, x * m)
    den = np.bincount(ids, m.astype(np.float64))
    return (num / np.maximum(den, 1))[ids]


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
    assert (FOLD >= 0).all()
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    a6 = pf.read(columns=["a6"]).column(0).to_numpy().reshape(-1, T).astype(np.float32)
    EVENT = np.isclose(a6, float(a6.min()), atol=1e-6)
    del a6

    def load(name):
        return np.load(ROOT / f"cache/pred_{name}_valid.npy", mmap_mode="r")

    par = [load(f"c20261001_d256_s3_f{f}") for f in range(4)]
    hft = [load(f"c20261001_ft_d256_s3_f{f}h") for f in range(4)]
    ts3 = [load(f"c20260929_since_s3_f{f}") for f in range(4)]
    ts4 = [load(f"c20260929_since_s4_f{f}") for f in range(4)]
    a07p, a07h = load("c20261001_d256a07_s3_f0"), load("c20261001_ft_d256a07_s3_f0h")

    # per-class median |TE| for teacher bins (first 60 sequences of each class)
    med = np.zeros(4)
    for c in range(4):
        idx = np.flatnonzero(CLS == c)[:60]
        v = [np.abs(0.5 * (np.asarray(ts3[FOLD[i]][i]) + np.asarray(ts4[FOLD[i]][i])))[MASK[i]] for i in idx]
        med[c] = float(np.median(np.concatenate(v)))
    print("median |TE| by class", med.round(4), flush=True)

    n, d = (int(sys.argv[1]) if len(sys.argv) > 1 else len(Y)), len(FEATURES)
    M = np.zeros((n, 2, d, d))
    # H-059 binned sums: [fold, class, k, since-bin, te-bin, stat]
    NS = 11  # w, wg, wgg, wr, wgr, wy, wP, wPP, wyP, wyy, wrr
    HB = np.zeros((4, 4, 2, 6, 12, NS))
    # u-bin calibration table: [fold, class, ubin, stat] stats w, wy, wTE, wTE2, wyTE, wP, wP2, wyP, wyy
    UB = np.zeros((4, 4, 6, 9))
    # response profile: [class, since 0..15 and 16+, wsums 6] for P, per target summed
    RP = np.zeros((4, 17, 2, 6))
    # in-sample vs OOF: [model 0 teacher/1 d256 parent, g fold-model, in(0)/oof(1), class, k, 6]
    MEM = np.zeros((2, 4, 2, 4, 2, 6))
    d0 = ["one", "y", "Psince", "Pa05", "Pa07", "Psince_seg", "Pa05_seg", "Pa07_seg"]
    M0 = np.zeros((2, 4, len(d0), len(d0)))  # [k, class]
    t_start = time.time()
    for i in range(n):
        f, c = FOLD[i], CLS[i]
        y = np.clip(np.asarray(Y[i], np.float64), -2, 2)
        m = MASK[i]
        P = np.clip(0.5 * (np.asarray(par[f][i], np.float64) + np.asarray(hft[f][i], np.float64)), -2, 2)
        TE = np.clip(0.5 * (np.asarray(ts3[f][i], np.float64) + np.asarray(ts4[f][i], np.float64)), -2, 2)
        ev_start = EVENT[i].copy()
        ev_start[0] = True
        lab_start = np.zeros(T, bool)
        lab_start[0] = True
        lab_start[1:] = (Y[i][1:] != Y[i][:-1]).any(-1)
        ev_seg = np.cumsum(ev_start) - 1
        since = np.arange(T) - np.flatnonzero(ev_start)[ev_seg]
        lab_ids = np.cumsum(lab_start) - 1
        lab_end = np.append(np.flatnonzero(lab_start)[1:], T)  # first row of the next label segment
        u = (lab_end[lab_ids] - np.arange(T)).astype(np.float64)  # rows until the label changes (>= 1)
        g = np.exp(-u / 32.0)
        sb = np.searchsorted(SINCE_EDGES, since, side="right") - 1
        ub = np.searchsorted(U_EDGES, u, side="right") - 1
        for k in range(2):
            yk, Pk, Tk = y[:, k], P[:, k], TE[:, k]
            hold, run = hold_run(Pk, ev_start)
            fwdP, endP, segP = forward_stats(Pk, lab_start)
            fwdT, endT, segT = forward_stats(Tk, lab_start)
            feats = np.stack([np.ones(T), yk, Pk, hold, run, reset_ema(Pk, ev_start, 0.3),
                              reset_ema(Pk, ev_start, 0.6), reset_ema(Pk, ev_start, 0.9), segP, fwdP, endP,
                              Tk, fwdT, endT, segT, prev_seg_end(Pk, ev_start),
                              scored_seg_mean(Pk, lab_ids, m), scored_seg_mean(Tk, lab_ids, m)], 1)[m]
            w = np.abs(yk[m])
            M[i, k] = (feats * w[:, None]).T @ feats
            # H-059 bins
            tb = np.clip(np.searchsorted(TB_MULT * med[c], Tk[m], side="right") - 1, 0, 11)
            r = yk[m] - Tk[m]
            gm = g[m]
            stats = np.stack([w, w * gm, w * gm * gm, w * r, w * gm * r, w * yk[m], w * Pk[m], w * Pk[m] ** 2,
                              w * yk[m] * Pk[m], w * yk[m] ** 2, w * r * r], 1)
            flat = sb[m] * 12 + tb
            HB[f, c, k] += np.stack([np.bincount(flat, stats[:, j], minlength=72) for j in range(NS)], 1).reshape(6, 12, NS)
            ustats = np.stack([w, w * yk[m], w * Tk[m], w * Tk[m] ** 2, w * yk[m] * Tk[m], w * Pk[m], w * Pk[m] ** 2,
                               w * yk[m] * Pk[m], w * yk[m] ** 2], 1)
            UB[f, c] += np.stack([np.bincount(ub[m], ustats[:, j], minlength=6) for j in range(9)], 1)
            sk = np.minimum(since[m], 16)
            ws = np.stack([w, w * yk[m], w * Pk[m], w * yk[m] ** 2, w * Pk[m] ** 2, w * yk[m] * Pk[m]], 1)
            RP[c, :, k] += np.stack([np.bincount(sk, ws[:, j], minlength=17) for j in range(6)], 1)
            for gfold in range(4):
                io = 1 if gfold == f else 0
                for mi, arrs in enumerate(((ts3, ts4), (par,))):
                    pg = np.clip(sum(np.asarray(a[gfold][i][:, k], np.float64) for a in arrs) / len(arrs), -2, 2)[m]
                    MEM[mi, gfold, io, c, k] += [w.sum(), (w * yk[m]).sum(), (w * pg).sum(), (w * yk[m] ** 2).sum(),
                                                 (w * pg * pg).sum(), (w * yk[m] * pg).sum()]
            if f == 0:
                ps = TE[:, k] * 0 + np.clip(np.asarray(ts3[0][i][:, k], np.float64), -2, 2)
                p5 = Pk
                p7 = np.clip(0.5 * (np.asarray(a07p[i][:, k], np.float64) + np.asarray(a07h[i][:, k], np.float64)), -2, 2)
                f0 = np.stack([np.ones(T), yk, ps, p5, p7, forward_stats(ps, lab_start)[2],
                               segP, forward_stats(p7, lab_start)[2]], 1)[m]
                M0[k, c] += (f0 * w[:, None]).T @ f0
        if i % 200 == 0:
            print(i, f"{time.time() - t_start:.0f}s", flush=True)
    np.savez(OUT / ("panel_sums.npz" if n == len(Y) else "panel_smoke.npz"), M=M, HB=HB, UB=UB, RP=RP, MEM=MEM, M0=M0, CLS=CLS[:n], FOLD=FOLD[:n],
             features=np.array(FEATURES), d0=np.array(d0), med=med)
    print("done", f"{time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
