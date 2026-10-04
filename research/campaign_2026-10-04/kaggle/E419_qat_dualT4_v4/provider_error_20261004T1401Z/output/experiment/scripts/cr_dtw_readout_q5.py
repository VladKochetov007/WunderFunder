"""Q5: toy-student test of the training targets of Q4 (bias vs label-noise trade-off).

Student = |y|-weighted cell mean of a target T over cells (class x elapsed-row bucket x decile of an independent
causal feature h = GRU128 OOF prediction of the same target). Trained on n random sequences of three folds, scored
(Global WP against y on scored rows) on the fourth fold; n in {8, 32, 128, all}, 8 draws for small n. A target that
is unbiased but less noisy wins at small n and ties at n = all; a biased target loses at n = all.
Teacher M = OOF head-FT, optionally per-(class, target) slope-calibrated to y (calibrated variants marked 'cal').
"""

from __future__ import annotations

import json
import sys

import numba as nb
import numpy as np

from cr_dtw_readout_core import NB_BUCKET, OUT, ROOT, bucket_of, load_panel, wp

LAMS = np.array([0.0, 0.5, 0.8, 0.9, 0.97])
NAMES = (["y", "distill_a0.5", "distill_a0.7", "h059", "segend_E174", "teacher_M"]
         + [f"lambda_{l:g}" for l in LAMS] + ["cal_distill_a0.7"] + [f"cal_lambda_{l:g}" for l in (0.5, 0.8, 0.9)] + ["distill_a0.85"])
NC = len(NAMES)
NQ = 10
NCELL = 4 * NB_BUCKET * NQ


@nb.njit(parallel=True, cache=True)
def kernel(y, mask, M, H, E, cls, edges, beta, t0):
    S, T, _ = y.shape
    tr = np.zeros((S, NC, 2, NCELL, 2), np.float32)
    ev_ = np.zeros((S, 2, NCELL, 3))
    for i in nb.prange(S):
        nxt = np.empty(T, np.int64)
        n = T
        for t in range(T - 1, -1, -1):
            nxt[t] = n
            if E[i, t]:
                n = t
        st = 0
        cell = np.empty(T, np.int64)
        C = np.empty((T, NC))
        G = np.empty(T)
        c = cls[i]
        for j in range(2):
            st = 0
            for t in range(T):
                if t > 0 and E[i, t]:
                    st = t
                q = 0
                hv = H[i, t, j]
                while q < NQ - 1 and hv > edges[c, j, q]:
                    q += 1
                cell[t] = (c * NB_BUCKET + bucket_of(t - st)) * NQ + q
            for t in range(T):
                yy = min(max(y[i, t, j], -2.0), 2.0)
                m = min(max(M[i, t, j], -2.0), 2.0)
                u = nxt[t] - t
                al = 0.5 + 0.4 * (1.0 - np.exp(-u / 32.0))
                C[t, 0] = yy
                C[t, 1] = 0.5 * m + 0.5 * yy
                C[t, 2] = 0.7 * m + 0.3 * yy
                C[t, 3] = al * m + (1 - al) * yy
                C[t, 4] = min(max(M[i, nxt[t] - 1, j], -2.0), 2.0)
                C[t, 5] = m
                C[t, 11] = 0.7 * m * beta[c, j] + 0.3 * yy
                C[t, 15] = 0.85 * m + 0.15 * yy
            for q in range(LAMS_N):
                lam = LAMS_V[q]
                for t in range(T - 1, -1, -1):
                    if nxt[t] == t + 1 or t == T - 1:
                        G[t] = C[t, 0]
                    else:
                        G[t] = (1 - lam) * min(max(M[i, t + 1, j], -2.0), 2.0) + lam * G[t + 1]
                    C[t, 6 + q] = G[t]
            for q in range(3):
                lam = (0.5, 0.8, 0.9)[q]
                for t in range(T - 1, -1, -1):
                    if nxt[t] == t + 1 or t == T - 1:
                        G[t] = C[t, 0]
                    else:
                        G[t] = (1 - lam) * beta[c, j] * min(max(M[i, t + 1, j], -2.0), 2.0) + lam * G[t + 1]
                    C[t, 12 + q] = G[t]
            for t in range(t0, T):
                yy = C[t, 0]
                w = abs(yy)
                if w == 0.0:
                    continue
                k = cell[t]
                for a in range(NC):
                    tr[i, a, j, k, 0] += w
                    tr[i, a, j, k, 1] += w * C[t, a]
                if mask[i, t]:
                    ev_[i, j, k, 0] += w
                    ev_[i, j, k, 1] += w * yy
                    ev_[i, j, k, 2] += w * yy * yy
    return tr, ev_


LAMS_V = LAMS
LAMS_N = len(LAMS)


def score(cellpred, ev):
    """cellpred (2, NCELL) student values; ev (2, NCELL, 3) scored-row sums [w, wy, wyy] -> WP."""
    w, wy, wyy = ev[..., 0], ev[..., 1], ev[..., 2]
    s = np.stack([w.sum(-1), wy.sum(-1), (w * cellpred).sum(-1), wyy.sum(-1), (w * cellpred**2).sum(-1),
                  (wy * cellpred).sum(-1)], -1)
    return float(wp(s))


def main():
    swap = "--swap" in sys.argv  # teacher = GRU128 since OOF (the real, weaker distillation teacher), cells on head-FT
    y, mask, ft, par, E, fold, cls = load_panel()
    H = np.empty_like(ft)
    for k in range(4):
        src = np.load(ROOT / f"cache/pred_c20260929_since_s3_f{k}_valid.npy", mmap_mode="r")
        for i in np.flatnonzero(fold == k):
            H[i] = src[i]
    if swap:
        H, ft = ft, H
    edges = np.zeros((4, 2, NQ - 1))
    beta = np.zeros((4, 2))
    rng = np.random.default_rng(0)
    for c in range(4):
        idx = rng.choice(np.flatnonzero(cls == c), 60, replace=False)
        for j in range(2):
            hv = H[idx, 99:, j].ravel()
            edges[c, j] = np.quantile(hv, np.linspace(0, 1, NQ + 1)[1:-1])
            yy = np.clip(y[cls == c, 99:, j], -2, 2)
            mm = np.clip(ft[cls == c, 99:, j], -2, 2)
            w = np.abs(yy)
            beta[c, j] = ((w * yy * mm).sum() / w.sum() - (w * yy).sum() * (w * mm).sum() / w.sum() ** 2) / (
                (w * mm * mm).sum() / w.sum() - ((w * mm).sum() / w.sum()) ** 2)
            del yy, mm, w
    print("teacher slope calibration beta (class x target):", beta.round(3).tolist())
    tr, ev = kernel(y, mask, ft, H, E, cls, edges, beta, 99)
    res = {}
    sizes = [8, 32, 128, 0]
    for n in sizes:
        draws = 8 if n else 1
        out = np.zeros((draws, 4, NC))
        for d in range(draws):
            for f in range(4):
                pool = np.flatnonzero(fold != f)
                pick = pool if n == 0 else np.random.default_rng(100 * d + f).choice(pool, n, replace=False)
                tsum = tr[pick].sum(0, dtype=np.float64)  # (NC, 2, NCELL, 2)
                evf = ev[fold == f].sum(0)
                # empty cells fall back to the class mean of the target
                for a in range(NC):
                    num, den = tsum[a, :, :, 1], tsum[a, :, :, 0]
                    cm = np.zeros_like(num)
                    for c in range(4):
                        sl = slice(c * NB_BUCKET * NQ, (c + 1) * NB_BUCKET * NQ)
                        cm[:, sl] = (num[:, sl].sum(-1) / np.maximum(den[:, sl].sum(-1), 1e-12))[:, None]
                    pred = np.where(den > 0, num / np.maximum(den, 1e-12), cm)
                    out[d, f, a] = score(pred, evf)
        mean = out.mean(1)  # fold-mean WP per draw
        res[f"n{n or 'all'}"] = {a: {"wp": float(mean[:, k].mean()), "delta_vs_y": float((mean[:, k] - mean[:, 0]).mean()),
                                     "delta_sd": float((mean[:, k] - mean[:, 0]).std()),
                                     "folds_delta": (out[:, :, k] - out[:, :, 0]).mean(0).round(5).tolist()}
                                 for k, a in enumerate(NAMES)}
        print(f"n = {n or 'all'} sequences: WP(y-trained) {mean[:, 0].mean():.4f}")
        for k, a in enumerate(NAMES):
            r = res[f"n{n or 'all'}"][a]
            print(f"  {a:18s} delta {r['delta_vs_y']:+.4f} (sd {r['delta_sd']:.4f}) folds {r['folds_delta']}")
    res["beta"] = beta.tolist()
    (OUT / ("q5_toy_student_swap.json" if swap else "q5_toy_student.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
