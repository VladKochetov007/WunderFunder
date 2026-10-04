"""Analyse research/campaign_2026-10-02/dtw/objective/panel_sums.npz (built by cr_dtw_objective_panel.py).

Every WP below is pooled over scored rows (step >= 99), |y| weights, clip +-2 on inputs, mean of the two targets.
Linear combinations are not re-clipped (predictions sit inside +-2 almost everywhere). Public-like = whole sequences
reweighted to the class mix [0.183, 0.160, 0.198, 0.459]. Cross-fitted fits use the three other folds only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/campaign_2026-10-02/dtw/objective"
MIX = np.array([0.183, 0.160, 0.198, 0.459])

z = np.load(OUT / "panel_sums.npz")
M, CLS, FOLD = z["M"], z["CLS"], z["FOLD"]
F = [str(s) for s in z["features"]]
D = len(F)
iy = F.index("y")
share = np.bincount(CLS, minlength=4) / len(CLS)
SEQW = (MIX / share)[CLS]  # public-like sequence weights


def vec(**coef):
    a = np.zeros(D)
    for k, v in coef.items():
        a[F.index(k)] = v
    return a


def wp_mat(Mk, a):
    """Weighted Pearson of y with a.z from one pooled moment matrix."""
    e = np.zeros(D)
    e[iy] = 1
    w = Mk[0, 0]
    sy, sp = Mk[0] @ e, Mk[0] @ a
    vy = e @ Mk @ e / w - (sy / w) ** 2
    vp = a @ Mk @ a / w - (sp / w) ** 2
    return (e @ Mk @ a / w - sy * sp / w ** 2) / np.sqrt(vy * vp)


def pooled(sel, a_by_class, seqw=None):
    """a_by_class: (4, 2, D) coefficients per class and target, or a single (D,) vector."""
    a = np.broadcast_to(a_by_class, (4, 2, D)) if np.ndim(a_by_class) == 1 else a_by_class
    sw = np.ones(len(CLS)) if seqw is None else seqw
    out = []
    for k in range(2):
        # WP of a class-dependent linear map: pool sums of [y, p] per class, then combine
        S = np.zeros(6)
        for c in range(4):
            Mc = np.einsum("i,ijk->jk", sw[sel & (CLS == c)], M[sel & (CLS == c), k])
            if Mc[0, 0] == 0:
                continue
            e = np.zeros(D)
            e[iy] = 1
            ac = a[c, k]
            S += [Mc[0, 0], Mc[0] @ e, Mc[0] @ ac, e @ Mc @ e, ac @ Mc @ ac, e @ Mc @ ac]
        w, sy, sp, syy, spp, syp = S
        out.append((syp / w - sy * sp / w ** 2) / np.sqrt((syy / w - (sy / w) ** 2) * (spp / w - (sp / w) ** 2)))
    return float(np.mean(out))


def report(a_fn, name, base_fn=None):
    """a_fn(train_sel) -> (4,2,D) coefficients; cross-fitted when it uses train_sel. Delta vs base_fn per fold."""
    res = {}
    for label, sw in (("pooled", None), ("public", SEQW)):
        vals, base = [], []
        for f in range(4):
            sel, tr = FOLD == f, FOLD != f
            vals.append(pooled(sel, a_fn(tr), sw))
            base.append(pooled(sel, (base_fn or (lambda t: vec(P=1)))(tr), sw))
        res[label] = {"delta_by_fold": np.round(np.array(vals) - np.array(base), 5).tolist(),
                      "delta_mean": round(float(np.mean(np.array(vals) - np.array(base))), 5)}
    allsel = np.ones(len(CLS), bool)
    res["full_panel_wp"] = round(pooled(allsel, a_fn(allsel)), 5)
    cls = {}
    for c in range(4):
        sel = CLS == c
        cls[c] = round(pooled(sel, a_fn(FOLD >= 0)) - pooled(sel, (base_fn or (lambda t: vec(P=1)))(FOLD >= 0)), 5)
    res["class_delta_in_sample_coef"] = cls
    print(f"{name:42s} pooled {res['pooled']['delta_mean']:+.5f} {res['pooled']['delta_by_fold']} "
          f"public {res['public']['delta_mean']:+.5f} cls {list(cls.values())}")
    return res


def ls_fit(sel, cols):
    """Per class/target |y|-weighted least squares of y on [one]+cols (WP-aligned for a linear family)."""
    a = np.zeros((4, 2, D))
    idx = [F.index("one")] + [F.index(c) for c in cols]
    for c in range(4):
        for k in range(2):
            Mc = M[sel & (CLS == c), k].sum(0)
            A = Mc[np.ix_(idx, idx)]
            b = Mc[idx, iy]
            a[c, k, idx] = np.linalg.solve(A + 1e-9 * np.eye(len(idx)), b)
    return a


R = {}
allsel = np.ones(len(CLS), bool)
R["base_wp"] = {"pooled": round(pooled(allsel, vec(P=1)), 5), "public": round(pooled(allsel, vec(P=1), SEQW), 5),
                "class": {c: round(pooled(CLS == c, vec(P=1)), 5) for c in range(4)},
                "fold": {f: round(pooled(FOLD == f, vec(P=1)), 5) for f in range(4)}}
print("base", R["base_wp"])

print("\n== causal inference-time variants of P (event segments reset at a6 events) ==")
const = {
    "hold (P at segment start)": vec(P_hold=1),
    "running mean since event": vec(P_run=1),
    "reset EMA beta .3": vec(P_ema3=1),
    "reset EMA beta .6": vec(P_ema6=1),
    "reset EMA beta .9": vec(P_ema9=1),
    "extrapolate P+.25(P-run)": vec(P=1.25, P_run=-0.25),
    "extrapolate P+.5(P-run)": vec(P=1.5, P_run=-0.5),
    "extrapolate P+.25(P-hold)": vec(P=1.25, P_hold=-0.25),
    "shrink P-.25(P-hold)": vec(P=0.75, P_hold=0.25),
    "P + .25 prev_seg_end": vec(P=1, P_prev_seg_end=0.25),
}
R["causal_fixed"] = {k: report(lambda t, a=a: a, k) for k, a in const.items()}
print("-- cross-fitted per-class |y|-LS on causal features (base = cross-fitted per-class affine [1,P]) --")
aff = lambda t: ls_fit(t, ["P"])
R["causal_ls"] = {}
for cols in (["P", "P_run"], ["P", "P_hold"], ["P", "P_hold", "P_run", "P_ema6"], ["P", "P_prev_seg_end"],
             ["P", "TE"], ["P", "P_hold", "P_run", "P_ema6", "P_prev_seg_end", "TE"]):
    R["causal_ls"]["+".join(cols)] = report(lambda t, cols=cols: ls_fit(t, cols), "LS " + "+".join(cols), aff)
R["affine_vs_raw"] = report(aff, "LS P (affine per class) vs raw P")
R["causal_ls_coef_full"] = {c: ls_fit(allsel, ["P", "P_hold", "P_run"])[c, :, [F.index(x) for x in ("P", "P_hold", "P_run")]].round(3).tolist() for c in range(4)}
print("coef [P,hold,run] per class (rows target0/1):", R["causal_ls_coef_full"])

print("\n== hindsight (non-causal, diagnostic only) ==")
for k, a in {"label-seg mean over scored rows": vec(P_segmean_sc=1), "label-seg mean all rows": vec(P_segmean_hs=1),
             "forward mean to seg end": vec(P_fwd=1), "value at seg end": vec(P_td=1)}.items():
    R.setdefault("hindsight", {})[k] = report(lambda t, a=a: a, "HS " + k)


def var_within(sel, k, x, xs):
    Mk = M[sel, k].sum(0)
    a, b = vec(**{x: 1}), vec(**{xs: 1})
    w = Mk[0, 0]
    tot = a @ Mk @ a / w - (Mk[0] @ a / w) ** 2
    within = (a - b) @ Mk @ (a - b) / w
    return within / tot


R["within_seg_var_share_P"] = {c: round(float(np.mean([var_within(CLS == c, k, "P", "P_segmean_sc") for k in range(2)])), 4) for c in range(4)}
R["within_seg_var_share_TE"] = {c: round(float(np.mean([var_within(CLS == c, k, "TE", "TE_segmean_sc") for k in range(2)])), 4) for c in range(4)}
print("within-label-segment share of Var(P) on scored rows by class:", R["within_seg_var_share_P"], "TE:", R["within_seg_var_share_TE"])

print("\n== training-target emulation with the OOF teacher TE (targets use future rows: training-side only) ==")
targets = {
    "y": vec(y=1), "TE": vec(TE=1), "a.5": vec(y=.5, TE=.5), "a.7": vec(y=.3, TE=.7), "a.85": vec(y=.15, TE=.85),
    "TE_fwd (mean TE over [t,end])": vec(TE_fwd=1), "TE_td (TE at seg end)": vec(TE_td=1),
    "TE_seg (whole-seg mean, all rows)": vec(TE_seg=1),
    ".5y+.5TE_fwd": vec(y=.5, TE_fwd=.5), ".3y+.7TE_fwd": vec(y=.3, TE_fwd=.7),
    ".3y+.7(.5TE+.5TE_fwd)": vec(y=.3, TE=.35, TE_fwd=.35), ".3y+.7TE_seg": vec(y=.3, TE_seg=.7),
}
causal_cols = ["P", "P_hold", "P_run", "P_ema6", "P_prev_seg_end", "TE"]
idxc = [F.index("one")] + [F.index(c) for c in causal_cols]
R["targets"] = {}
for name, t in targets.items():
    row = {}
    for c in list(range(4)) + ["all"]:
        sel = allsel if c == "all" else CLS == c
        vals = {"wp_T_y": [], "noise_vs_TE": [], "slope_T_on_TE_rel": [], "proj_wp": []}
        for k in range(2):
            Mk = M[sel, k].sum(0)
            w = Mk[0, 0]
            e, te = vec(y=1), vec(TE=1)
            vals["wp_T_y"].append(wp_mat(Mk, t))
            d1, d0 = t - te, e - te
            vals["noise_vs_TE"].append((d1 @ Mk @ d1) / (d0 @ Mk @ d0))
            cov = lambda a, b: a @ Mk @ b / w - (Mk[0] @ a / w) * (Mk[0] @ b / w)
            vals["slope_T_on_TE_rel"].append(cov(t, te) / cov(e, te))
        row[c] = {k: round(float(np.mean(v)), 4) for k, v in vals.items() if v}
    # learnable part: cross-fitted per-class LS projection of T on causal features, scored against y
    def proj(tr, t=t):
        a = np.zeros((4, 2, D))
        for c in range(4):
            for k in range(2):
                Mc = M[tr & (CLS == c), k].sum(0)
                A = Mc[np.ix_(idxc, idxc)]
                a[c, k, idxc] = np.linalg.solve(A + 1e-9 * np.eye(len(idxc)), Mc[idxc] @ t)
        return a
    row["proj"] = report(proj, f"proj[{name}]", lambda tr: ls_fit(tr, causal_cols))
    R["targets"][name] = row
    print(f"   {name:36s}", {c: row[c] for c in ("all", 3)})

print("\n== H-059 bias: b(bin) = 0.4 Cov_w(g, y - TE | class, since-bin, TE-bin), g = exp(-u/32) ==")
HB = z["HB"]  # [fold, class, k, sb, tb, stat] stats w, wg, wgg, wr, wgr, wy, wP, wPP, wyP, wyy, wrr


def wp_shift(Hsel, b):
    """WP (mean of targets) of P + b over bins; Hsel [class, k, sb, tb, stat], b same leading shape."""
    out = []
    for k in range(2):
        H = Hsel[:, k]
        bb = b[:, k]
        w, wy, wp, wpp, wyp, wyy = (H[..., j] for j in (0, 5, 6, 7, 8, 9))
        S = [w.sum(), wy.sum(), (wp + bb * w).sum(), wyy.sum(), (wpp + 2 * bb * wp + bb * bb * w).sum(),
             (wyp + bb * wy).sum()]
        n, sy, sp, syy, spp, syp = S
        out.append((syp / n - sy * sp / n ** 2) / np.sqrt((syy / n - (sy / n) ** 2) * (spp / n - (sp / n) ** 2)))
    return float(np.mean(out))


def bias_from(Htr):
    w = Htr[..., 0]
    safe = np.maximum(w, 1e-12)
    return np.where(w > 0, 0.4 * (Htr[..., 4] / safe - Htr[..., 1] / safe * Htr[..., 3] / safe), 0.0)


R["h059"] = {"delta_by_fold": [], "delta_class": {}}
for f in range(4):
    b = bias_from(HB[np.arange(4) != f].sum(0))
    R["h059"]["delta_by_fold"].append(round(wp_shift(HB[f], b) - wp_shift(HB[f], np.zeros_like(b)), 5))
Hall = HB.sum(0)
ball = bias_from(Hall)
for c in range(4):
    R["h059"]["delta_class"][c] = round(wp_shift(Hall[c:c + 1], ball[c:c + 1]) - wp_shift(Hall[c:c + 1], 0 * ball[c:c + 1]), 5)
w = Hall[..., 0]
rms_b = np.sqrt((w * ball ** 2).sum() / w.sum())
rms_p = np.sqrt(Hall[..., 7].sum() / w.sum())
gbar = Hall[..., 1].sum() / w.sum()
var_g = Hall[..., 2].sum() / w.sum() - gbar ** 2
between = ((Hall[..., 1] ** 2 / np.maximum(w, 1e-12)).sum() / w.sum()) - gbar ** 2
R["h059"].update({"rms_bias": round(float(rms_b), 4), "rms_P": round(float(rms_p), 4),
                  "R2_g_on_causal_bins": round(float(between / var_g), 4), "mean_g": round(float(gbar), 4)})
print(R["h059"])

print("\n== calibration of y on the OOF teacher by rows-until-label-change u (non-causal stratifier) ==")
UB = z["UB"].sum(0)  # [class, ubin, stat] w, wy, wTE, wTE2, wyTE, wP, wP2, wyP, wyy
ulabels = ["<2", "2-8", "8-32", "32-128", "128-512", ">=512"]
R["u_calibration"] = {}
for c in range(4):
    rows = []
    for j in range(6):
        w, wy, wt, wtt, wyt = UB[c, j, :5]
        wyy = UB[c, j, 8]
        slope = (wyt / w - wy / w * wt / w) / (wtt / w - (wt / w) ** 2)
        corr = (wyt / w - wy / w * wt / w) / np.sqrt((wtt / w - (wt / w) ** 2) * (wyy / w - (wy / w) ** 2))
        rows.append({"u": ulabels[j], "w_share": round(float(w / UB[c, :, 0].sum()), 3), "slope_y_on_TE": round(float(slope), 3),
                     "corr": round(float(corr), 3), "mean_resid": round(float((wy - wt) / w), 4)})
    R["u_calibration"][c] = rows
    print(c, [(r["u"], r["w_share"], r["slope_y_on_TE"], r["corr"]) for r in rows])

print("\n== teacher / student in-sample vs out-of-fold WP on valid (memorisation gap of distillation targets) ==")
MEM = z["MEM"]  # [model, gfold, in/oof, class, k, 6]


def wp6(S):
    n, sy, sp, syy, spp, syp = np.moveaxis(S, -1, 0)
    return np.mean((syp / n - sy * sp / n ** 2) / np.sqrt((syy / n - (sy / n) ** 2) * (spp / n - (sp / n) ** 2)))


R["memorisation"] = {}
for mi, mname in enumerate(("teacher since s3+s4", "d256 parent")):
    row = {}
    for c in list(range(4)) + ["all"]:
        Sin = MEM[mi, :, 0].sum(0)
        Soo = MEM[mi, :, 1].sum(0)
        if c != "all":
            Sin, Soo = Sin[c], Soo[c]
        else:
            Sin, Soo = Sin.sum(0), Soo.sum(0)
        row[c] = {"in_sample": round(float(wp6(Sin)), 4), "oof": round(float(wp6(Soo)), 4)}
    R["memorisation"][mname] = row
    print(mname, row)

print("\n== fold 0: since parent vs distilled alpha .5 / .7 (within-segment variance share, hindsight seg-mean WP) ==")
M0 = z["M0"]
d0 = [str(s) for s in z["d0"]]
R["fold0_distill"] = {}
for c in list(range(4)) + ["all"]:
    row = {}
    for nm, sg in (("Psince", "Psince_seg"), ("Pa05", "Pa05_seg"), ("Pa07", "Pa07_seg")):
        vals = {"wp": [], "wp_segmean": [], "within_share": []}
        for k in range(2):
            Mk = M0[k].sum(0) if c == "all" else M0[k, c]
            a = np.zeros(len(d0)); a[d0.index(nm)] = 1
            b = np.zeros(len(d0)); b[d0.index(sg)] = 1
            e = np.zeros(len(d0)); e[1] = 1
            w = Mk[0, 0]
            cov = lambda u, v: u @ Mk @ v / w - (Mk[0] @ u / w) * (Mk[0] @ v / w)
            vals["wp"].append(cov(e, a) / np.sqrt(cov(e, e) * cov(a, a)))
            vals["wp_segmean"].append(cov(e, b) / np.sqrt(cov(e, e) * cov(b, b)))
            vals["within_share"].append((a - b) @ Mk @ (a - b) / w / cov(a, a))
        row[nm] = {k: round(float(np.mean(v)), 4) for k, v in vals.items()}
    R["fold0_distill"][c] = row
    print(c, row)

print("\n== response profile: WP of P by rows since the last a6 event (class 3 and all) ==")
RP = z["RP"]  # [class, since, k, 6]
R["response"] = {}
for c in (0, 1, 2, 3):
    R["response"][c] = [round(float(wp6(RP[c, s])), 3) for s in range(17)]
    wsh = RP[c, :, :, 0].sum(-1) / RP[c, :, :, 0].sum()
    R["response"][f"{c}_weight_share"] = np.round(wsh, 4).tolist()
    print(c, R["response"][c][:9], "w-share since<4:", round(float(wsh[:4].sum()), 4))

(OUT / "analysis.json").write_text(json.dumps(R, indent=1, default=float))
