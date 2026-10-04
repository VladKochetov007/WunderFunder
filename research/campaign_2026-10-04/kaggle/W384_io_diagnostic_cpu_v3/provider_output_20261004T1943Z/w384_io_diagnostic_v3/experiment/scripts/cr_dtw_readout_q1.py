"""Q1: hindsight algebra on the pooled OOF panel.

(a) Do target changes coincide with E rows (a6 lower endpoint)? (b) Is |y| (and y) constant inside E-segments?
(c) Between/within-segment decomposition of the |y|-weighted Var(p) per class, and the non-causal oracle that
replaces p by its within-segment mean (over scored rows) — for y-runs and for causal E-segments; also the oracle
"segment-last" readout (p at the last scored row of the segment broadcast to the whole segment).
"""

from __future__ import annotations

import json

import numba as nb
import numpy as np

from cr_dtw_readout_core import OUT, load_panel, summarize, wp


@nb.njit(parallel=True, cache=True)
def kernel(y, mask, ft, par, E):
    S, T = mask.shape
    cnt = np.zeros((S, 6), np.int64)  # rows t>=1, y-change, E, change&E, change&~E, scored rows in E-seg with y != seg y
    sums = np.zeros((S, 3, 3, 2, 6))  # readout (base, seg-mean, seg-last) x pred (ft, par, mix) x target x raw sums
    sums_y = np.zeros((S, 3, 2, 6))  # seg-mean over y-runs, pred x target
    for i in nb.prange(S):
        for t in range(1, T):
            chg = (y[i, t, 0] != y[i, t - 1, 0]) or (y[i, t, 1] != y[i, t - 1, 1])
            cnt[i, 0] += 1
            cnt[i, 1] += chg
            cnt[i, 2] += E[i, t]
            cnt[i, 3] += chg and E[i, t]
            cnt[i, 4] += chg and not E[i, t]
        pv = np.empty(T)
        for k in range(3):
            for j in range(2):
                for t in range(T):
                    a = ft[i, t, j] if k == 0 else (par[i, t, j] if k == 1 else 0.5 * (ft[i, t, j] + par[i, t, j]))
                    pv[t] = min(max(a, -2.0), 2.0)
                for segdef in range(2):  # 0: E-segments (causal boundaries), 1: y-runs
                    t0 = 0
                    while t0 < T:
                        t1 = t0 + 1
                        while t1 < T and not (E[i, t1] if segdef == 0 else
                                              (y[i, t1, 0] != y[i, t1 - 1, 0] or y[i, t1, 1] != y[i, t1 - 1, 1])):
                            t1 += 1
                        sp = 0.0
                        n = 0
                        last = 0.0
                        yseg = 0.0
                        for t in range(t0, t1):
                            if mask[i, t]:
                                if n == 0:
                                    yseg = y[i, t, j]
                                elif segdef == 0 and k == 0 and y[i, t, j] != yseg:
                                    cnt[i, 5] += 1
                                sp += pv[t]
                                n += 1
                                last = pv[t]
                        if n > 0:
                            mp = sp / n
                            for t in range(t0, t1):
                                if not mask[i, t]:
                                    continue
                                yy = min(max(y[i, t, j], -2.0), 2.0)
                                w = abs(yy)
                                for r in range(3):
                                    if segdef == 1 and r != 1:
                                        continue
                                    v = pv[t] if r == 0 else (mp if r == 1 else last)
                                    o = sums[i, r, k, j] if segdef == 0 else sums_y[i, k, j]
                                    o[0] += w
                                    o[1] += w * yy
                                    o[2] += w * v
                                    o[3] += w * yy * yy
                                    o[4] += w * v * v
                                    o[5] += w * yy * v
                        t0 = t1
    return cnt, sums, sums_y


def main():
    y, mask, ft, par, E, fold, cls = load_panel()
    cnt, sums, sums_y = kernel(y, mask, ft, par, E)
    res = {"counts": {}}
    for c in range(4):
        cc = cnt[cls == c].sum(0)
        res["counts"][f"c{c}"] = {"rows": int(cc[0]), "y_changes": int(cc[1]), "E_rows": int(cc[2]),
                                  "change_and_E": int(cc[3]), "change_not_E": int(cc[4]),
                                  "P(change|E)": cc[3] / max(cc[2], 1), "scored_rows_y_differs_in_Eseg": int(cc[5])}
    # models: base ft, base par, base mix, segmean_E ft/par/mix, seglast_E ft/par/mix, segmean_y ft/par/mix
    names, arrs = [], []
    for r, rn in enumerate(["base", "segmean_E", "seglast_E"]):
        for k, kn in enumerate(["ft", "par", "mix"]):
            names.append(f"{rn}_{kn}")
            arrs.append(sums[:, r, k])
    for k, kn in enumerate(["ft", "par", "mix"]):
        names.append(f"segmean_y_{kn}")
        arrs.append(sums_y[:, k])
    seq = np.stack(arrs, 1)
    summ = summarize(seq, fold, cls)
    res["wp"] = {n: {key: float(v[m]) for key, v in summ.items()} for m, n in enumerate(names)}
    # variance decomposition per class/target (ft and mix): share of weighted Var(p) that is between E-segments
    dec = {}
    for k, kn in enumerate(["ft", "mix"]):
        kk = [0, 2][k]
        for c in list(range(4)) + ["all"]:
            sel = np.ones(len(cls), bool) if c == "all" else cls == c
            b = sums[sel, 0, kk].sum(0)
            o = sums[sel, 1, kk].sum(0)
            n = b[:, 0]
            vt = b[:, 4] / n - (b[:, 2] / n) ** 2
            vb = o[:, 4] / n - (o[:, 2] / n) ** 2
            cov_b = b[:, 5] / n - b[:, 1] * b[:, 2] / n**2
            cov_o = o[:, 5] / n - o[:, 1] * o[:, 2] / n**2
            dec[f"{kn}_c{c}"] = {"var_between_share_t0t1": (vb / vt).round(4).tolist(),
                                 "cov_change_by_segmean": (cov_o / cov_b - 1).round(6).tolist(),
                                 "wp_base": float(wp(b)), "wp_oracle": float(wp(o))}
    res["decomposition"] = dec
    (OUT / "q1_hindsight.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["counts"], indent=1, default=float))
    for n in names:
        r = res["wp"][n]
        print(f"{n:16s} pooled {r['pooled']:.4f} public {r['public']:.4f} " +
              " ".join(f"c{c} {r[f'c{c}']:.4f}" for c in range(4)) + " | " +
              " ".join(f"f{k} {r[f'f{k}']:.4f}" for k in range(4)))
    for key, v in dec.items():
        print(key, v["var_between_share_t0t1"], v["cov_change_by_segmean"], round(v["wp_base"], 4), round(v["wp_oracle"], 4))


if __name__ == "__main__":
    main()
