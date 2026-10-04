"""Queue-race falsifier: does in-run history decode the class-3 queue race beyond the current row and the GRU?

    WNN_CPU_PCT=25 WNN_MEM_MAX=12G scripts/limited .venv/bin/python scripts/cr_queue_probe.py TRAIN.npz VALID.npz \
        SURPRISE_ROWS.npz OUT.json

Stage 1 (TRAIN only, class 3, halves by chain component): HistGradientBoosting on (i) the current 112 features
(control) vs (ii) + in-run history. Labels: side = sign(D) (D != 0 rows), winner = max(t0, t1) >= 1 tick, and
|y|-weighted regressions of t0, t1. Held-out AUC / |y|-weighted corr with D, all rows and the quiet-early subset
(>= 32 rows to the next quote move, diagnostic selection only).
Stage 2 (VALID class-3 scored rows, clean OOF since-FT pair): partial correlation of each score beyond p and a
cross-fitted blend p' = p + sum_i beta_i z_i (betas fitted on the other three folds) -> pooled Global WP delta and
class-3 delta with a chain-component bootstrap.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.evaluation import paired_component_bootstrap  # noqa: E402
from wnn.splits import connected_components  # noqa: E402

CLIP = 2.0
TICK = 0.3
HGB = dict(max_iter=int(os.environ.get("CR_ITERS", 250)), learning_rate=0.08, max_leaf_nodes=63,
           min_samples_leaf=400, l2_regularization=1.0, early_stopping=False, random_state=0)


def auc(score, lab, w=None):
    o = np.argsort(score, kind="stable")
    w = np.ones(len(lab)) if w is None else w
    ws, ls = w[o], lab[o]
    cw_neg = np.cumsum(ws * ~ls)
    pos = ws * ls
    return float((pos * (cw_neg - 0.5 * ws * ~ls)).sum() / (pos.sum() * (ws * ~ls).sum()))


def wcorr(a, b, w):
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    return float((w * (a - ma) * (b - mb)).sum() / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum()))


def wp1(y, p):
    y = np.clip(y.astype(np.float64), -CLIP, CLIP)
    return wcorr(y, np.clip(p.astype(np.float64), -CLIP, CLIP), np.abs(y))


def wp(Y, P):
    return 0.5 * (wp1(Y[:, 0], P[:, 0]) + wp1(Y[:, 1], P[:, 1]))


def partial(y, p, z):
    """|y|-weighted partial correlation of y with z given (1, p)."""
    y = np.clip(y.astype(np.float64), -CLIP, CLIP)
    w = np.abs(y)
    A = np.column_stack([np.ones(len(p)), np.clip(p, -CLIP, CLIP)])
    sw = np.sqrt(w)[:, None]
    ry = y - A @ np.linalg.lstsq(A * sw, y * sw[:, 0], rcond=None)[0]
    rz = z - A @ np.linalg.lstsq(A * sw, z * sw[:, 0], rcond=None)[0]
    return wcorr(ry, rz, w)


def seq_moments(seq, Y, P):
    """Per-sequence weighted Welford moments (N, 2, 6) for paired_component_bootstrap(centered=True)."""
    u, inv = np.unique(seq, return_inverse=True)
    out = np.zeros((len(u), 2, 6))
    for k in range(2):
        y = np.clip(Y[:, k].astype(np.float64), -CLIP, CLIP)
        p = np.clip(P[:, k].astype(np.float64), -CLIP, CLIP)
        w = np.abs(y)
        W = np.bincount(inv, w, len(u))
        Ws = np.where(W > 0, W, 1)
        my, mp = np.bincount(inv, w * y, len(u)) / Ws, np.bincount(inv, w * p, len(u)) / Ws
        dy, dp = y - my[inv], p - mp[inv]
        out[:, k] = np.column_stack([W, my, mp, np.bincount(inv, w * dy * dy, len(u)),
                                     np.bincount(inv, w * dp * dp, len(u)), np.bincount(inv, w * dy * dp, len(u))])
    return u, out


def fit_models(X, Y, fit, tag, log):
    """Side/winner classifiers and t0/t1 regressors on rows `fit`."""
    D = 0.5 * (Y[:, 0] - Y[:, 1])
    nz = fit & (np.abs(Y).sum(1) > 0)
    models = {}
    t = time.time()
    s = nz & (D != 0)
    models["side"] = HistGradientBoostingClassifier(**HGB).fit(X[s], D[s] > 0)
    models["win"] = HistGradientBoostingClassifier(**HGB).fit(X[nz], Y[nz].max(1) > TICK)
    for j in (0, 1):
        s = fit & (Y[:, j] != 0)
        models[f"r{j}"] = HistGradientBoostingRegressor(**HGB).fit(X[s], Y[s, j], sample_weight=np.abs(Y[s, j]))
    log(f"fit {tag} {time.time() - t:.0f}s")
    return models


def score(models, X):
    return {"side": models["side"].decision_function(X), "win": models["win"].decision_function(X),
            "r0": models["r0"].predict(X), "r1": models["r1"].predict(X)}


def heldout_metrics(S, Y, sub):
    D = 0.5 * (Y[:, 0] - Y[:, 1])
    nz = sub & (np.abs(Y).sum(1) > 0)
    s = nz & (D != 0)
    wD = np.abs(Y).mean(1)
    sD = 0.5 * (S["r0"] - S["r1"])
    return {"rows": int(sub.sum()),
            "auc_side": auc(S["side"][s], D[s] > 0), "auc_side_wD": auc(S["side"][s], D[s] > 0, np.abs(D[s])),
            "auc_win": auc(S["win"][nz], Y[nz].max(1) > TICK),
            "corr_D_regscore": wcorr(D[nz], sD[nz], wD[nz]),
            "corr_D_sideprob": wcorr(D[nz], S["side"][nz], wD[nz]),
            "wp_reg": wp(Y[nz], np.column_stack([S["r0"], S["r1"]])[nz])}


def blend(Y, P, Z, fold, cols, mode):
    """Cross-fitted class-3 remap from WLS y_j ~ a + b p_j + sum_i c_i z_ij fitted on the other three folds.

    mode "add": p_j + sum_i (c_i / b)(z_ij - mean_fit z_ij) (keeps the class-3 mean and calibration of p);
    mode "cal": a + b p_j + sum_i c_i z_ij (compare against cols=[] i.e. a + b p_j).
    """
    out = P.astype(np.float64).copy()
    betas = []
    for j in (0, 1):
        y = np.clip(Y[:, j].astype(np.float64), -CLIP, CLIP)
        p = np.clip(P[:, j].astype(np.float64), -CLIP, CLIP)
        zz = np.column_stack([Z[c][:, j] for c in cols]) if cols else np.zeros((len(p), 0))
        A = np.column_stack([np.ones(len(p)), p, zz])
        sw = np.sqrt(np.abs(y))
        for k in range(4):
            f, a = fold != k, fold == k
            coef = np.linalg.lstsq(A[f] * sw[f, None], y[f] * sw[f], rcond=None)[0]
            if mode == "add":
                beta = coef[2:] / coef[1]
                out[a, j] = P[a, j] + (zz[a] - zz[f].mean(0)) @ beta
            else:
                out[a, j] = A[a] @ coef
            betas.append(coef.tolist())
    return out, betas


def main() -> None:
    tr_path, va_path, rows_path, out_path = map(Path, sys.argv[1:5])
    res: dict = {}
    lines: list[str] = []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    # ---------------- stage 1: TRAIN halves by chain component
    z = np.load(tr_path)
    sub = slice(None, None, int(os.environ.get("CR_SUB", 1)))  # smoke tests only
    XH = np.concatenate([z["X"][sub], z["H"][sub]], 1)
    Y = z["y"][sub].astype(np.float64)
    comp, until = z["comp"][sub], z["nc_until_E"][sub]
    names = list(z["hist_names"])
    rng = np.random.default_rng(0)
    uc = np.unique(comp)
    half_a = set(rng.permutation(uc)[: len(uc) // 2].tolist())
    fit = np.array([c in half_a for c in comp])
    ho = ~fit
    log(f"train rows {len(Y)} fit {fit.sum()} heldout {ho.sum()} comps {len(uc)}")
    del z
    feats = {"control": XH[:, :112], "hist": XH}
    models = {k: fit_models(v, Y, fit, k, log) for k, v in feats.items()}
    quiet = until >= 32
    res["train_heldout"] = {}
    for k, v in feats.items():
        S = score(models[k], v[ho])
        res["train_heldout"][k] = {"all": heldout_metrics(S, Y[ho], np.ones(ho.sum(), bool)),
                                   "quiet32": heldout_metrics(S, Y[ho], quiet[ho])}
        log(f"{k}: {json.dumps(res['train_heldout'][k])}")
    res["hist_feature_names"] = names
    del XH, feats

    # ---------------- stage 2: VALID class-3 scored rows vs clean OOF GRU
    v = np.load(va_path)
    XHv = np.concatenate([v["X"], v["H"]], 1)
    Yv = v["y"].astype(np.float64)
    seq_v, t_v, until_v, comp_v = v["seq"], v["t"], v["nc_until_E"], v["comp"]
    r = np.load(rows_path)
    rseq, rt, rcls = r["seq"].astype(np.int64), r["t"].astype(np.int64), r["cls"]
    s3 = rcls == 3
    assert np.array_equal(rseq[s3], seq_v) and np.array_equal(rt[s3], t_v)
    Pall = np.column_stack([r["ft0"], r["ft1"]]).astype(np.float64)
    Yall = np.column_stack([r["y0"], r["y1"]]).astype(np.float64)
    assert np.allclose(Yall[s3], Yv)
    foldv = r["seq_fold"][rseq][s3]
    P = Pall[s3]
    Sv = {"control": score(models["control"], XHv[:, :112]), "hist": score(models["hist"], XHv)}
    del XHv
    quiet_v = until_v >= 32
    pD = 0.5 * (P[:, 0] - P[:, 1])
    sg = np.sign(pD)
    base_c3 = wp(Yv, P)
    base_all = wp(Yall, Pall)
    res["valid"] = {"base_wp_all": base_all, "base_wp_c3": base_c3, "base_wp_c3_quiet32": wp(Yv[quiet_v], P[quiet_v]),
                    "rows_c3": int(len(Yv)), "quiet32_share_rows": float(quiet_v.mean())}
    Dv = 0.5 * (Yv[:, 0] - Yv[:, 1])
    np.savez(out_path.with_suffix(".scores.npz"), **{f"{k}_{c}": v for k in Sv for c, v in Sv[k].items()})
    ch = np.load(ROOT / "cache/chains.npz")
    ccomp = connected_components(ch["next"])
    vcomp = {int(g): int(c) for s_, g, c in zip(ch["split"], ch["row_group"], ccomp, strict=True) if s_ == "valid"}
    nz = np.abs(Yv).sum(1) > 0
    useqs, _ = seq_moments(rseq, Yall, Pall)
    comps_all = np.array([vcomp[int(g)] for g in useqs])

    def compare(Pn, ref):
        Pall_n, Pall_r = Pall.copy(), Pall.copy()
        Pall_n[s3], Pall_r[s3] = Pn, ref
        d = {"d_c3": wp(Yv, Pn) - wp(Yv, ref), "d_all": wp(Yall, Pall_n) - wp(Yall, Pall_r),
             "d_c3_quiet32": wp(Yv[quiet_v], Pn[quiet_v]) - wp(Yv[quiet_v], ref[quiet_v]),
             "d_c3_by_fold": [wp(Yv[foldv == f], Pn[foldv == f]) - wp(Yv[foldv == f], ref[foldv == f])
                              for f in range(4)]}
        _, ma = seq_moments(rseq, Yall, Pall_n)
        _, mr = seq_moments(rseq, Yall, Pall_r)
        d["boot_all"] = paired_component_bootstrap(ma, mr, comps_all, n_boot=1000, centered=True)
        u3, ma3 = seq_moments(seq_v, Yv, Pn)
        _, mr3 = seq_moments(seq_v, Yv, ref)
        d["boot_c3"] = paired_component_bootstrap(ma3, mr3, np.array([vcomp[int(g)] for g in u3]), n_boot=1000,
                                                  centered=True)
        return d

    for k in ("control", "hist"):
        S = Sv[k]
        m = {"features_only": heldout_metrics(S, Yv, np.ones(len(Yv), bool)),
             "features_only_quiet32": heldout_metrics(S, Yv, quiet_v)}
        sel = nz & (Dv != 0)
        m["auc_side_p_only"] = auc(pD[sel], Dv[sel] > 0)
        m["auc_side_p_only_quiet32"] = auc(pD[sel & quiet_v], Dv[sel & quiet_v] > 0)
        m["auc_win_p_only"] = auc(np.abs(pD[nz]), Yv[nz].max(1) > TICK)
        Z = {"reg": np.column_stack([S["r0"], S["r1"]]),
             "side": np.column_stack([S["side"], -S["side"]]),
             "win": np.column_stack([S["win"] * sg, -S["win"] * sg])}
        for q, sub in (("all", np.ones(len(Yv), bool)), ("quiet32", quiet_v)):
            for c in Z:
                m[f"partial_{c}_{q}"] = [partial(Yv[sub, j], P[sub, j], Z[c][sub, j]) for j in (0, 1)]
        cal0, _ = blend(Yv, P, Z, foldv, [], "cal")
        for cols in (["reg"], ["reg", "side", "win"]):
            for mode, ref in (("add", P), ("cal", cal0)):
                Pn, betas = blend(Yv, P, Z, foldv, cols, mode)
                d = {"coefs": betas, **compare(Pn, ref)}
                name = f"blend_{mode}_{'+'.join(cols)}"
                m[name] = d
                log(f"valid {k} {name}: d_c3 {d['d_c3']:+.5f} d_all {d['d_all']:+.5f} "
                    f"quiet {d['d_c3_quiet32']:+.5f} folds {np.round(d['d_c3_by_fold'], 5).tolist()} "
                    f"boot_all sd {d['boot_all']['bootstrap_sd']:.5f} p<=0 {d['boot_all']['p_le_0']:.3f} "
                    f"boot_c3 sd {d['boot_c3']['bootstrap_sd']:.5f} p<=0 {d['boot_c3']['p_le_0']:.3f}")
        if k == "control":
            res["valid"]["cal0_vs_raw"] = compare(cal0, P)
        res["valid"][k] = m
        log(f"valid {k}: auc_side {m['features_only']['auc_side']:.4f} (p-only {m['auc_side_p_only']:.4f}) "
            f"quiet {m['features_only_quiet32']['auc_side']:.4f} (p-only {m['auc_side_p_only_quiet32']:.4f}) "
            f"auc_win {m['features_only']['auc_win']:.4f} (p-only {m['auc_win_p_only']:.4f}) "
            f"partial reg {np.round(m['partial_reg_all'], 4).tolist()} quiet {np.round(m['partial_reg_quiet32'], 4).tolist()}")
    res["log"] = lines
    out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
