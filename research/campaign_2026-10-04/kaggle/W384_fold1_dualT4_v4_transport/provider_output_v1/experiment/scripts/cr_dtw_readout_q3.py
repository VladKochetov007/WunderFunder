"""Q3: lag / anticipation / alignment diagnostics of the pooled OOF panel (head-FT unless stated).

(a) Shift experiments: causal delay p[t-d]; non-causal in-segment lead p[min(t+d, seg_end)]; post-event "settle"
    bound (rows in the first d rows of a segment get the value the model reaches d rows later).
(b) Information accrual: |y|-weighted Pearson of p with its own segment target by elapsed-row bucket and by
    rows-to-segment-end bucket (the latter is non-causal bookkeeping), per class; persistence of the OLD segment's
    target in p right after an event (lag signature) via per-k least squares p[s+k] ~ y_new + y_old.
(c) Constrained DTW (Sakoe-Chiba band) of the prediction path onto the target path per sequence/target: warping
    offset (aligned p index - y index) by position relative to events, and WP of the warped (non-causal) path.
(d) Parent vs head-FT: lagged correlation of first differences and banded DTW offset.
"""

from __future__ import annotations

import json
import sys

import numba as nb
import numpy as np

from cr_dtw_readout_core import OUT, load_panel, summarize, wp

SHIFTS = np.array([1, 2, 4, 8, 16, 32, 64])
NPOS = 13  # log2 buckets for elapsed / remaining rows
KMAX = 16  # post-event rows for the old/new regression


@nb.njit(cache=True)
def lb(k):
    b = 0
    k1 = k + 1
    while k1 > 1 and b < NPOS - 1:
        k1 >>= 1
        b += 1
    return b


@nb.njit(cache=True)
def acc(o, w, yy, v):
    o[0] += w
    o[1] += w * yy
    o[2] += w * v
    o[3] += w * yy * yy
    o[4] += w * v * v
    o[5] += w * yy * v


@nb.njit(parallel=True, cache=True)
def shift_kernel(y, mask, P, E, shifts):
    """Per-seq raw sums (S, 3*len(shifts), 2, 6) for delay / in-segment lead / settle; plus accrual sums."""
    S, T = mask.shape
    nd = shifts.shape[0]
    out = np.zeros((S, 3 * nd, 2, 6))
    acc_el = np.zeros((S, NPOS, 2, 6))
    acc_rm = np.zeros((S, NPOS, 2, 6))
    for i in nb.prange(S):
        start = np.zeros(T, np.int64)
        end = np.zeros(T, np.int64)  # last row of the segment
        s = 0
        for t in range(T):
            if t > 0 and E[i, t]:
                s = t
            start[t] = s
        e = T - 1
        for t in range(T - 1, -1, -1):
            end[t] = e
            if E[i, t]:
                e = t - 1
        for j in range(2):
            for t in range(T):
                if not mask[i, t]:
                    continue
                yy = min(max(y[i, t, j], -2.0), 2.0)
                w = abs(yy)
                pv = min(max(P[i, t, j], -2.0), 2.0)
                acc(acc_el[i, lb(t - start[t]), j], w, yy, pv)
                acc(acc_rm[i, lb(end[t] - t), j], w, yy, pv)
                for q in range(nd):
                    d = shifts[q]
                    v = P[i, max(t - d, 0), j]
                    acc(out[i, q, j], w, yy, min(max(v, -2.0), 2.0))
                    v = P[i, min(t + d, end[t]), j]
                    acc(out[i, nd + q, j], w, yy, min(max(v, -2.0), 2.0))
                    v = P[i, min(start[t] + d, end[t]), j] if t - start[t] < d else P[i, t, j]
                    acc(out[i, 2 * nd + q, j], w, yy, min(max(v, -2.0), 2.0))
    return out, acc_el, acc_rm


@nb.njit(parallel=True, cache=True)
def oldnew_kernel(y, P, E, cls, minlen):
    """Normal equations of p[s+k] ~ [1, y_new, y_old] over events s whose old and new segments are >= minlen rows."""
    S, T = E.shape
    xtx = np.zeros((S, KMAX, 2, 3, 3))
    xty = np.zeros((S, KMAX, 2, 3))
    for i in nb.prange(S):
        prev = -1
        for s in range(100, T):
            if not E[i, s]:
                continue
            nxt = s + 1
            while nxt < T and not E[i, nxt]:
                nxt += 1
            if prev >= 0 and s - prev >= minlen and nxt - s >= minlen:
                for j in range(2):
                    yn = min(max(y[i, s, j], -2.0), 2.0)
                    yo = min(max(y[i, s - 1, j], -2.0), 2.0)
                    x = (1.0, yn, yo)
                    for k in range(KMAX):
                        pv = min(max(P[i, s + k, j], -2.0), 2.0)
                        for a in range(3):
                            xty[i, k, j, a] += x[a] * pv
                            for b in range(3):
                                xtx[i, k, j, a, b] += x[a] * x[b]
            prev = s
    return xtx, xty


@nb.njit(cache=True)
def diff_lag_corr(a, b, maxlag, t0):
    """Pearson of first differences da[t] vs db[t - L] for L in [-maxlag, maxlag], pooled, mean over targets."""
    S, T, _ = a.shape
    out = np.zeros(2 * maxlag + 1)
    for j in range(2):
        for q in range(2 * maxlag + 1):
            L = q - maxlag
            n = sx = sy = sxx = syy = sxy = 0.0
            for i in range(S):
                for t in range(t0 + 1 + maxlag, T - maxlag):
                    x = a[i, t, j] - a[i, t - 1, j]
                    u = b[i, t - L, j] - b[i, t - L - 1, j]
                    n += 1
                    sx += x
                    sy += u
                    sxx += x * x
                    syy += u * u
                    sxy += x * u
            out[q] += 0.5 * (sxy / n - sx * sy / n**2) / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))
    return out


@nb.njit(cache=True)
def dtw_band(a, b, w, D, path_i, path_j):
    """Sakoe-Chiba banded DTW of series a (reference) and b (query), squared cost. Returns path length."""
    T = a.shape[0]
    W = 2 * w + 1
    inf = 1e300
    for i in range(T):
        for c in range(W):
            j = i + c - w
            if j < 0 or j >= T:
                D[i, c] = inf
                continue
            d = (a[i] - b[j]) ** 2
            if i == 0 and j == 0:
                D[i, c] = d
                continue
            best = inf
            if c - 1 >= 0 and D[i, c - 1] < best:  # (i, j-1)
                best = D[i, c - 1]
            if i > 0:
                if D[i - 1, c] < best:  # (i-1, j-1)
                    best = D[i - 1, c]
                if c + 1 < W and D[i - 1, c + 1] < best:  # (i-1, j)
                    best = D[i - 1, c + 1]
            D[i, c] = d + best
    i = T - 1
    c = w
    n = 0
    while True:
        path_i[n] = i
        path_j[n] = i + c - w
        n += 1
        if i == 0 and c == w:
            break
        best = 1e300
        mv = -1
        if c - 1 >= 0 and D[i, c - 1] < best:
            best = D[i, c - 1]
            mv = 0
        if i > 0 and D[i - 1, c] < best:
            best = D[i - 1, c]
            mv = 1
        if i > 0 and c + 1 < W and D[i - 1, c + 1] < best:
            best = D[i - 1, c + 1]
            mv = 2
        if mv == 0:
            c -= 1
        elif mv == 1:
            i -= 1
        else:
            i -= 1
            c += 1
    return n


@nb.njit(parallel=True, cache=True)
def dtw_kernel(y, mask, P, E, w, t0):
    """Banded DTW of P onto y (rows >= t0). Offsets by position bucket and warped-path WP sums."""
    S, T = mask.shape
    L = T - t0
    # position classes: 0 first 4 rows after event, 1 last 4 rows before event, 2 other
    off = np.zeros((S, 3, 2))  # sum offset, count
    off_abs = np.zeros((S, 3))
    hist = np.zeros((S, 2 * w + 1))
    wsum = np.zeros((S, 2, 6))
    for i in nb.prange(S):
        D = np.empty((L, 2 * w + 1))
        pi = np.empty(2 * L, np.int64)
        pj = np.empty(2 * L, np.int64)
        start = np.zeros(L, np.int64)
        end = np.zeros(L, np.int64)
        s = 0
        for t in range(L):
            if t > 0 and E[i, t0 + t]:
                s = t
            start[t] = s
        e = L - 1
        for t in range(L - 1, -1, -1):
            end[t] = e
            if E[i, t0 + t]:
                e = t - 1
        a = np.empty(L)
        b = np.empty(L)
        for j in range(2):
            for t in range(L):
                a[t] = min(max(y[i, t0 + t, j], -2.0), 2.0)
                b[t] = min(max(P[i, t0 + t, j], -2.0), 2.0)
            n = dtw_band(a, b, w, D, pi, pj)
            ssum = np.zeros(L)
            cnt = np.zeros(L)
            for q in range(n):
                ssum[pi[q]] += b[pj[q]]
                cnt[pi[q]] += 1
                d = pj[q] - pi[q]
                t = pi[q]
                pc = 0 if t - start[t] < 4 else (1 if end[t] - t < 4 else 2)
                off[i, pc, j] += d
                off_abs[i, pc] += 1
                hist[i, d + w] += 1
            for t in range(L):
                if mask[i, t0 + t]:
                    w_ = abs(a[t])
                    acc(wsum[i, j], w_, a[t], ssum[t] / cnt[t])
    return off, off_abs, hist, wsum


def main():
    y, mask, ft, par, E, fold, cls = load_panel()
    res = {}
    out, acc_el, acc_rm = shift_kernel(y, mask, ft, E, SHIFTS)
    base = summarize(np.load(OUT / "q2_seq_sums.npy")[:, :1], fold, cls)
    summ = summarize(out, fold, cls)
    names = [f"{kind}_{d}" for kind in ("delay", "lead_in_seg", "settle") for d in SHIFTS]
    res["shift"] = {n: {k: float(v[m] - base[k][0]) for k, v in summ.items()} for m, n in enumerate(names)}
    print("shift experiments (delta vs base_ft):")
    for n in names:
        r = res["shift"][n]
        print(f" {n:16s} pooled {r['pooled']:+.4f} public {r['public']:+.4f} " +
              " ".join(f"c{c} {r[f'c{c}']:+.4f}" for c in range(4)))
    acc_rows = {}
    for nm, A in (("elapsed", acc_el), ("remaining", acc_rm)):
        rows = {}
        for c in range(4):
            s = A[cls == c].sum(0)
            tot = s[:, :, 0].sum()
            rows[f"c{c}"] = [(int(2**k - 1), round(float(wp(s[k])), 4), round(float(s[k, :, 0].sum() / tot), 4)) for k in range(NPOS)]
        acc_rows[nm] = rows
        print(f"accrual by {nm} rows bucket (lower edge, WP, weight share):")
        for c in range(4):
            print(f" c{c}", " ".join(f"{a}:{b:.3f}/{w:.2f}" for a, b, w in rows[f"c{c}"]))
    res["accrual"] = acc_rows
    for minlen in (1, 16):
        xtx, xty = oldnew_kernel(y, ft, E, cls, minlen)
        on = {}
        for c in range(4):
            A = xtx[cls == c].sum(0)
            b = xty[cls == c].sum(0)
            coefs = []
            for k in range(KMAX):
                cj = [np.linalg.solve(A[k, j] + 1e-9 * np.eye(3), b[k, j]) for j in range(2)]
                coefs.append((round(float(np.mean([x[1] for x in cj])), 4), round(float(np.mean([x[2] for x in cj])), 4)))
            on[f"c{c}"] = coefs
        res[f"old_new_minlen{minlen}"] = on
        print(f"p[s+k] ~ y_new, y_old coefficients (k=0..{KMAX - 1}), segments >= {minlen} rows:")
        for c in range(4):
            print(f" c{c}", " ".join(f"{a:.3f}/{b:.3f}" for a, b in on[f"c{c}"][:6]), "...", "%.3f/%.3f" % on[f"c{c}"][-1])
    for w in (4, 16):
        off, offn, hist, wsum = dtw_kernel(y, mask, ft, E, w, 99)
        o = off.sum(2)
        rows = {}
        for c in list(range(4)) + ["all"]:
            sel = np.ones(len(cls), bool) if c == "all" else cls == c
            rows[f"c{c}"] = [round(float(o[sel, p].sum() / offn[sel, p].sum()), 3) for p in range(3)]
        sw = summarize(wsum[:, None], fold, cls)
        res[f"dtw_w{w}"] = {"mean_offset_[first4_after_event,last4_before_event,other]": rows,
                            "offset_hist": (hist.sum(0) / hist.sum()).round(4).tolist(),
                            "warped_wp_delta": {k: float(v[0] - base[k][0]) for k, v in sw.items()}}
        print(f"DTW band {w}: mean offset (p index - y index) [first4 after event, last4 before, other]", rows)
        print("  offset hist", (hist.sum(0) / hist.sum()).round(3).tolist())
        print("  warped WP delta", {k: round(float(v[0] - base[k][0]), 4) for k, v in sw.items() if not k.startswith('f')})
    # parent vs FT: lagged correlation of first differences, banded DTW offset
    lagc = {int(L): round(float(v), 4) for L, v in zip(range(-3, 4), diff_lag_corr(ft, par, 3, 99))}
    off, offn, hist, wsum = dtw_kernel(par, mask, ft, E, 4, 99)
    res["par_vs_ft"] = {"diff_corr_by_lag(ft leads if L>0)": lagc,
                        "dtw_w4_mean_offset": float(off.sum() / offn.sum()),
                        "dtw_w4_offset_hist": (hist.sum(0) / hist.sum()).round(4).tolist()}
    print("par vs ft:", res["par_vs_ft"])
    (OUT / "q3_alignment.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
