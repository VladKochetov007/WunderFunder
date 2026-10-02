"""Target emulation on stopped_sums.npz / scored_sums.npz: population bias (cross-fitted per-class LS projection of the
target on causal features [P, TE], scored against y, minus the same for T = y), relative slope on TE, label-noise share.
Run: scripts/limited .venv/bin/python research/campaign_2026-10-02/dtw/objective/analyze_targets.py"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
src = sys.argv[1] if len(sys.argv) > 1 else "stopped_sums.npz"
z = np.load(OUT / src)
M, CLS, FOLD = z["M"], z["CLS"], z["FOLD"]
F = [str(s) for s in z["features"]]
D = len(F)
MIX = np.array([0.183, 0.160, 0.198, 0.459])
SEQW = (MIX / (np.bincount(CLS, minlength=4) / len(CLS)))[CLS]


def vec(**c):
    a = np.zeros(D)
    for k, v in c.items():
        a[F.index(k)] = v
    return a


def pooled(sel, A, sw=None):
    sw = np.ones(len(CLS)) if sw is None else sw
    out = []
    for k in range(2):
        S = np.zeros(6)
        for c in range(4):
            Mc = np.einsum("i,ijk->jk", sw[sel & (CLS == c)], M[sel & (CLS == c), k])
            a, e = A[c, k], vec(y=1)
            S += [Mc[0, 0], Mc[0] @ e, Mc[0] @ a, e @ Mc @ e, a @ Mc @ a, e @ Mc @ a]
        w, sy, sp, syy, spp, syp = S
        out.append((syp / w - sy * sp / w ** 2) / np.sqrt((syy / w - (sy / w) ** 2) * (spp / w - (sp / w) ** 2)))
    return float(np.mean(out))


idxc = [F.index(c) for c in ("one", "P", "TE")]


def proj(tr, t):
    A = np.zeros((4, 2, D))
    for c in range(4):
        for k in range(2):
            Mc = M[tr & (CLS == c), k].sum(0)
            A[c, k, idxc] = np.linalg.solve(Mc[np.ix_(idxc, idxc)], Mc[idxc] @ t)
    return A


def stats(t, sel):
    sl, nz = [], []
    for k in range(2):
        Mk = M[sel, k].sum(0)
        w = Mk[0, 0]
        e, te = vec(y=1), vec(TE=1)
        cov = lambda u, v: u @ Mk @ v / w - (Mk[0] @ u / w) * (Mk[0] @ v / w)
        sl.append(cov(t, te) / cov(e, te))
        nz.append(((t - te) @ Mk @ (t - te)) / ((e - te) @ Mk @ (e - te)))
    return round(float(np.mean(sl)), 3), round(float(np.mean(nz)), 3)


def run(targets):
    res = {}
    print(f"{'target':28s} slope_rel all/c0/c3   noise(vs y-TE) all/c3   population-bias dWP by fold (pooled) | mean | public mean | class(in-sample)")
    for nm, t in targets.items():
        d = [pooled(FOLD == f, proj(FOLD != f, t)) - pooled(FOLD == f, proj(FOLD != f, vec(y=1))) for f in range(4)]
        dp = [pooled(FOLD == f, proj(FOLD != f, t), SEQW) - pooled(FOLD == f, proj(FOLD != f, vec(y=1)), SEQW) for f in range(4)]
        allm = FOLD >= 0
        dc = [round(pooled(CLS == c, proj(allm, t)) - pooled(CLS == c, proj(allm, vec(y=1))), 5) for c in range(4)]
        s_all, s0, s3 = stats(t, allm), stats(t, CLS == 0), stats(t, CLS == 3)
        res[nm] = {"slope_rel": [s_all[0], s0[0], s3[0]], "noise": [s_all[1], s3[1]], "bias_fold": np.round(d, 5).tolist(),
                   "bias_mean": round(float(np.mean(d)), 5), "bias_public": round(float(np.mean(dp)), 5), "bias_class": dc}
        print(f"{nm:28s} {s_all[0]:.3f}/{s0[0]:.3f}/{s3[0]:.3f}   {s_all[1]:.3f}/{s3[1]:.3f}   {np.round(d, 5)} | {np.mean(d):+.5f} | {np.mean(dp):+.5f} | {dc}")
    return res


if src == "stopped_sums.npz":
    tg = {"y": vec(y=1), "TE": vec(TE=1), "a.5": vec(y=.5, TE=.5), "a.7": vec(y=.3, TE=.7), "a.85": vec(y=.15, TE=.85),
          "TE_td (run end, E-174)": vec(TE_td=1), "T_dtw (DTW-aligned TE)": vec(T_dtw=1)}
    for h in (1, 2, 4, 8, 32, 128, 512, 2048):
        tg[f"T_h{h}"] = vec(**{f"T_h{h}": 1})
    for h in (2, 8, 32, 128):
        tg[f".3T_h{h}+.7TE"] = vec(**{f"T_h{h}": .3, "TE": .7})
        tg[f".5T_h{h}+.5TE"] = vec(**{f"T_h{h}": .5, "TE": .5})
    tg[".3T_dtw+.7TE"] = vec(T_dtw=.3, TE=.7)
    tg[".15T_h8+.85TE"] = vec(T_h8=.15, TE=.85)
    tg["H-059 a(u)=.9-.4g"] = vec(y=.1, TE=.9, gy=.4, gTE=-.4)
    tg["a.665 (H-059 mean a)"] = vec(y=.335, TE=.665)
    tg["causal gate a=.9-.4gh"] = vec(y=.1, TE=.9, ghy=.4, ghTE=-.4)
    tg[".7T_h8+.3TE"] = vec(T_h8=.7, TE=.3)
    res = run(tg)
    term = z["term"]
    res["terminated_share"] = {c: np.round(term[c, :, 0] / term[c, :, 1], 3).tolist() for c in range(4)}
    print("share of |y| weight whose run ends within h rows (h = 1, 2, 4, 8, 32, 128, 512, 2048) by class:", res["terminated_share"])
    (OUT / "analysis_stopped.json").write_text(json.dumps(res, indent=1))
