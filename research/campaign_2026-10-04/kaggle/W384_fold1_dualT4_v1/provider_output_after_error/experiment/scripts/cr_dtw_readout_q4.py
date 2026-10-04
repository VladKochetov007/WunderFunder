"""Q4: training-target diagnostics for piecewise-constant labels (training-time targets MAY use future labels).

Teacher M = pooled OOF head-FT prediction (stand-in for a frozen distillation teacher). Candidate targets per row:
  y; distill a*M_t + (1-a)*y (a = 0.5, 0.7); H-059 outcome-weighted a(u) (u = rows until the label changes);
  E-174-style seg-end bootstrap M_{e-1}; in-segment lead M_{min(t+8, e-1)}; DTW-warped teacher (band 16, aligned
  to the label path); and the optional-stopping lambda-return G_t = (1-l) M_{t+1} + l G_{t+1}, G_{e-1} = y
  (e = next event row). Only the lambda-return is unbiased for E_Q[y | x_t] when M is a calibrated teacher.
Diagnostics (|y|-weighted = the wMSE measure Q, rows >= 99, y != 0): for test functions h(x_t) — an independent
GRU128 OOF prediction, the teacher M_t itself, and (class x elapsed-bucket) indicators — the covariance ratio
Cov(T, h) / Cov(y, h) (1 = unbiased along h) and corr(T, h) / corr(y, h) (target signal-to-noise gain).
"""

from __future__ import annotations

import json
import sys

import numba as nb
import numpy as np

from cr_dtw_readout_core import NB_BUCKET, OUT, ROOT, bucket_of, load_panel
from cr_dtw_readout_q3 import dtw_band

LAMBDAS = np.array([0.0, 0.5, 0.8, 0.9, 0.97])
NAMES = (["y", "distill_a0.5", "distill_a0.7", "h059_a0.5to0.9", "segend_E174", "lead8_clamped", "dtw_warped_w16"]
         + [f"lambda_return_{l:g}" for l in LAMBDAS] + ["0.5M+0.5G0.9", "0.3M+0.7G0.9"])
NC = len(NAMES)
NH = 2


@nb.njit(parallel=True, cache=True)
def kernel(y, M, H, E, lambdas, t0, band):
    S, T, _ = y.shape
    nl = lambdas.shape[0]
    sums = np.zeros((S, NC, NH, 2, 6))
    bsum = np.zeros((S, NC, 2, NB_BUCKET, 2))
    for i in nb.prange(S):
        L = T - t0
        D = np.empty((L, 2 * band + 1))
        pi = np.empty(2 * L, np.int64)
        pj = np.empty(2 * L, np.int64)
        ev = np.empty(T, np.int64)  # next event row strictly after t (T if none)
        nxt = T
        for t in range(T - 1, -1, -1):
            ev[t] = nxt
            if E[i, t]:
                nxt = t
        st = np.empty(T, np.int64)
        s = 0
        for t in range(T):
            if t > 0 and E[i, t]:
                s = t
            st[t] = s
        C = np.empty((T, NC))
        G = np.empty(T)
        a = np.empty(L)
        b = np.empty(L)
        for j in range(2):
            for t in range(T):
                yy = min(max(y[i, t, j], -2.0), 2.0)
                m = min(max(M[i, t, j], -2.0), 2.0)
                e = ev[t]
                C[t, 0] = yy
                C[t, 1] = 0.5 * m + 0.5 * yy
                C[t, 2] = 0.7 * m + 0.3 * yy
                u = e - t
                al = 0.5 + 0.4 * (1.0 - np.exp(-u / 32.0))
                C[t, 3] = al * m + (1 - al) * yy
                C[t, 4] = min(max(M[i, e - 1, j], -2.0), 2.0)
                C[t, 5] = min(max(M[i, min(t + 8, e - 1), j], -2.0), 2.0)
            # DTW-warped teacher (rows >= t0), aligned onto the label path
            for t in range(L):
                a[t] = C[t0 + t, 0]
                b[t] = min(max(M[i, t0 + t, j], -2.0), 2.0)
            n = dtw_band(a, b, band, D, pi, pj)
            for t in range(T):
                C[t, 6] = 0.0
            cnt = np.zeros(T)
            for q in range(n):
                C[t0 + pi[q], 6] += b[pj[q]]
                cnt[t0 + pi[q]] += 1
            for t in range(t0, T):
                C[t, 6] /= cnt[t]
            for t in range(t0):
                C[t, 6] = C[t, 0]
            # lambda-returns: backward within segments, terminal = y at the last row before the event
            for q in range(nl):
                lam = lambdas[q]
                for t in range(T - 1, -1, -1):
                    if ev[t] == t + 1 or t == T - 1:
                        G[t] = C[t, 0]
                    else:
                        G[t] = (1 - lam) * min(max(M[i, t + 1, j], -2.0), 2.0) + lam * G[t + 1]
                    C[t, 7 + q] = G[t]
            for t in range(T):
                m = min(max(M[i, t, j], -2.0), 2.0)
                C[t, 7 + nl] = 0.5 * m + 0.5 * C[t, 7 + 3]
                C[t, 8 + nl] = 0.3 * m + 0.7 * C[t, 7 + 3]
            for t in range(t0, T):
                yy = C[t, 0]
                w = abs(yy)
                if w == 0.0:
                    continue
                bk = bucket_of(t - st[t])
                for k in range(NC):
                    v = C[t, k]
                    bsum[i, k, j, bk, 0] += w
                    bsum[i, k, j, bk, 1] += w * (v - yy)
                    for h in range(NH):
                        hv = H[i, t, j] if h == 0 else min(max(M[i, t, j], -2.0), 2.0)
                        o = sums[i, k, h, j]
                        o[0] += w
                        o[1] += w * v
                        o[2] += w * hv
                        o[3] += w * v * hv
                        o[4] += w * v * v
                        o[5] += w * hv * hv
    return sums, bsum


def stats(s):
    n, sv, sh, svh, svv, shh = np.moveaxis(s, -1, 0)
    cov = svh / n - sv * sh / n**2
    vv = svv / n - (sv / n) ** 2
    vh = shh / n - (sh / n) ** 2
    return cov, vv, cov / np.sqrt(vv * vh)


def main():
    y, mask, ft, par, E, fold, cls = load_panel()
    H = np.empty_like(ft)
    for k in range(4):
        src = np.load(ROOT / f"cache/pred_c20260929_since_s3_f{k}_valid.npy", mmap_mode="r")
        for i in np.flatnonzero(fold == k):
            H[i] = src[i]
    sums, bsum = kernel(y, ft, H, E, LAMBDAS, 99, 16)
    res = {}
    for c in list(range(4)) + ["all"]:
        sel = np.ones(len(cls), bool) if c == "all" else cls == c
        cov, vv, cor = stats(sums[sel].sum(0))  # (NC, NH, 2)
        b = bsum[sel].sum(0)  # (NC, 2, NB, 2)
        sd_y = np.sqrt(vv[0, 0])  # (2,)
        dev = b[..., 1] / np.maximum(b[..., 0], 1e-12) / sd_y[None, :, None]  # standardized E_Q[T - y | bucket]
        wsh = b[..., 0] / b[..., 0].sum(-1, keepdims=True)
        res[f"c{c}"] = {n: {"cov_ratio_vs_gru128": float((cov[k, 0] / cov[0, 0]).mean()),
                            "cov_ratio_vs_teacher": float((cov[k, 1] / cov[0, 1]).mean()),
                            "corr_gain_vs_gru128": float((cor[k, 0] / cor[0, 0]).mean()),
                            "sd_ratio": float(np.sqrt(vv[k, 0] / vv[0, 0]).mean()),
                            "bucket_bias_wmean_abs_sd": float((np.abs(dev[k]) * wsh[k]).sum(-1).mean()),
                            "bucket_bias_max_abs_sd": float(np.abs(dev[k])[wsh[k] > 0.01].max())}
                        for k, n in enumerate(NAMES)}
    (OUT / "q4_targets.json").write_text(json.dumps(res, indent=1))
    hdr = f"{'target':18s} {'covR_gru':>8s} {'covR_M':>8s} {'corrGain':>8s} {'sdRatio':>8s} {'bktBias':>8s} {'bktMax':>8s}"
    for c in ["all", 3, 0]:
        print(f"class {c}\n{hdr}")
        for n in NAMES:
            r = res[f"c{c}"][n]
            print(f"{n:18s} {r['cov_ratio_vs_gru128']:8.3f} {r['cov_ratio_vs_teacher']:8.3f} {r['corr_gain_vs_gru128']:8.3f} "
                  f"{r['sd_ratio']:8.3f} {r['bucket_bias_wmean_abs_sd']:8.4f} {r['bucket_bias_max_abs_sd']:8.4f}")


if __name__ == "__main__":
    sys.exit(main())
