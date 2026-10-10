import sys, json, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from lib import S, load, P, halves
from policies import *
from wnn.evaluation import paired_component_moments, paired_component_bootstrap

FLAT = {f: load(f) for f in (0, 1)}
WARM = {f: np.load(S / f"warm_f{f}.npy") for f in (0, 1)}
T = 20000

def make(fold, a, b):
    fl = FLAT[fold]; ew = np.load(S / f"ewma_f{fold}_{a}__{b}.npz")
    return D(fl, P(fold, a), P(fold, b), ew, WARM[fold], None), fl

def pad(fl, flat, rows):
    out = np.zeros((int(fl["seq"].max()) + 1, T, 2), np.float32); out[fl["seq"][rows], fl["step"][rows]] = flat; return out

def bootstrap(fl, rows, cand, ref, nb=1000):
    y = pad(fl, fl["y"][rows], rows); m = np.zeros(y.shape[:2], bool); m[fl["seq"][rows], fl["step"][rows]] = True
    ca = pad(fl, cand, rows); cb = pad(fl, ref, rows)
    ma, mb = paired_component_moments(y, ca, cb, m)
    keep = m.any(1)
    r = paired_component_bootstrap(ma[keep], mb[keep], fl["comp"][keep], n_boot=nb, seed=7, centered=True)
    return r

def run_policies(d, train_seq, cls_est, specs):
    """fit every policy on rows of train_seq (bool per seq), return dict name -> (params, predictor)."""
    tr = d.sub(train_seq)
    fits = {}
    fits["P1 static grid"] = (fit_static(tr, False), lambda p, x: apply_static(p, x))
    fits["P2 per-target weights"] = (fit_static(tr, True), lambda p, x: apply_static(p, x))
    for lvl, nm in ((0, "P3a ridge [p1,p2]"), (1, "P3b ridge [p1,p2,d,|d|]"), (2, "P3c ridge +causal feats/EWMA")):
        fits[nm] = (fit_ridge(tr, lvl), (lambda p, x, l=lvl: apply_ridge(p, x, l)))
    # gate: choose half-life by fit-set score
    best = None
    gates = {}
    for i, hl in enumerate(HLS):
        par = fit_gate(tr, i); s = wp(tr.y, apply_gate(par, tr, i)); gates[hl] = (s, par)
        fits[f"P4 gate hl{hl}"] = (par, (lambda p, x, i=i: apply_gate(p, x, i)))
    hlbest = max(gates, key=lambda h: gates[h][0]); fits["P4 gate (hl by fit)"] = (gates[hlbest][1], (lambda p, x, i=HLS.index(hlbest): apply_gate(p, x, i)))
    fits["P4 gate (hl by fit)"] = fits[f"P4 gate hl{hlbest}"]
    # tables with estimated class
    fits["P5 table class x age x vol"] = (fit_table(tr, cls_est, True, True), lambda p, x: apply_table(p, x, cls_est))
    fits["P6 class routing"] = (fit_table(tr, cls_est, False, False), lambda p, x: apply_table(p, x, cls_est))
    return fits, hlbest

def experiment(tag, d, fl, fit_sets, eval_sets, out):
    """fit_sets/eval_sets: lists of (fit seq mask, eval seq mask); held-out predictions pooled over directions."""
    names = None; pooled = {}; fit_scores = {}; held_dir = {}
    rows_all = np.zeros(d.n, bool)
    ref = np.zeros((d.n, 2), np.float32); single = np.zeros((d.n, 2), np.float32)
    best_single_choice = []
    for di, (fs, es) in enumerate(zip(fit_sets, eval_sets)):
        t0 = time.time()
        train_seq = fs
        cls_est, _ = fit_classifier(d.warm, d.cls, train_seq)
        fits, hlbest = run_policies(d, train_seq, cls_est, None)
        trd, ted = d.sub(fs), d.sub(es)
        er = es[d.seq]
        ref[er] = 0.5 * (d.pa[er] + d.pb[er])
        sa, sb = wp(trd.y, trd.pa), wp(trd.y, trd.pb)
        pick = d.pa if sa >= sb else d.pb; single[er] = pick[er]; best_single_choice.append("A" if sa >= sb else "B")
        for nm, (par, f) in fits.items():
            pred_e = f(par, ted)
            pooled.setdefault(nm, np.zeros((d.n, 2), np.float32))[er] = pred_e
            fit_scores.setdefault(nm, []).append(wp(trd.y, f(par, trd)))
            held_dir.setdefault(nm, []).append(wp(ted.y, pred_e))
            out.setdefault("params", {}).setdefault(nm, []).append(summ(par))
        held_dir.setdefault("static 0.5/0.5", []).append(wp(ted.y, 0.5 * (ted.pa + ted.pb)))
        fit_scores.setdefault("static 0.5/0.5", []).append(wp(trd.y, 0.5 * (trd.pa + trd.pb)))
        held_dir.setdefault("best single (by fit)", []).append(wp(ted.y, pick[er]))
        fit_scores.setdefault("best single (by fit)", []).append(wp(trd.y, pick[fs[d.seq]]))
        out.setdefault("gate_hl_by_fit", []).append(int(hlbest))
        print(tag, "direction", di, "done", round(time.time() - t0), "s", flush=True)
    rows = np.zeros(d.n, bool)
    for es in eval_sets: rows |= es[d.seq]
    res = {}
    for nm in pooled:
        r = bootstrap(fl, rows, pooled[nm][rows], ref[rows])
        rs = bootstrap(fl, rows, pooled[nm][rows], single[rows])
        res[nm] = {"fit_wp": fit_scores[nm], "heldout_wp_by_direction": held_dir[nm], "heldout_wp_pooled": r["candidate_wp"],
                   "delta_vs_static": r["mean_diff"], "ci_vs_static": r["ci95"], "delta_vs_best_single": rs["mean_diff"], "ci_vs_best_single": rs["ci95"]}
    res["static 0.5/0.5"] = {"fit_wp": fit_scores["static 0.5/0.5"], "heldout_wp_by_direction": held_dir["static 0.5/0.5"], "heldout_wp_pooled": r["reference_wp"]}
    rb = bootstrap(fl, rows, single[rows], ref[rows])
    res["best single (by fit)"] = {"fit_wp": fit_scores["best single (by fit)"], "heldout_wp_by_direction": held_dir["best single (by fit)"], "heldout_wp_pooled": rb["candidate_wp"],
                                   "delta_vs_static": rb["mean_diff"], "ci_vs_static": rb["ci95"], "choice": best_single_choice}
    out["results"] = res
    return out

def summ(par):
    def conv(o):
        if isinstance(o, np.ndarray): return np.round(o, 5).tolist()
        if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        if isinstance(o, (np.floating, float)): return round(float(o), 5)
        return o
    if isinstance(par, list) and par and isinstance(par[0], tuple) and len(par[0]) == 5:  # ridge: omit big vectors
        return [{"alpha": p[4], "beta_head": np.round(p[0][:6], 4).tolist(), "n_coef": len(p[0])} for p in par]
    return conv(par)

if __name__ == "__main__":
    task = sys.argv[1]
    outdir = Path(__file__).parent
    if task in ("P1", "P2", "P3f1", "P3f0"):
        spec = {"P1": (1, "h230_s3_ep3", "vt_gru512"), "P2": (1, "h230_s3_ep3", "vt_gru384_fixmix"),
                "P3f1": (1, "h230_s3_onecycle", "c20261003_combo_s3"), "P3f0": (0, "h230_s3_onecycle", "c20261003_combo_s3")}[task]
        fold, a, b = spec
        d, fl = make(fold, a, b)
        h = halves(fl["comp"])
        out = {"task": task, "fold": fold, "A": FLAT[fold]["members"][a], "B": FLAT[fold]["members"][b], "scheme": "component-disjoint halves crossfit",
               "n_comp_half0": int(len(np.unique(fl["comp"][h]))), "n_comp_half1": int(len(np.unique(fl["comp"][~h])))}
        out["single_full"] = {"A": wp(d.y, d.pa), "B": wp(d.y, d.pb), "static": wp(d.y, 0.5 * (d.pa + d.pb))}
        experiment(task + "-crossfit", d, fl, [h, ~h], [~h, h], out)
        (outdir / f"res_{task}_crossfit.json").write_text(json.dumps(out, indent=1))
    if task == "P3x":   # fold0 -> fold1 and fold1 -> fold0 transfer with the same pair
        a, b = "h230_s3_onecycle", "c20261003_combo_s3"
        for src, dst in ((0, 1), (1, 0)):
            ds, fs_ = make(src, a, b); dd, fd = make(dst, a, b)
            # train on all of src: emulate by fitting with src and applying to dst through a joint D? do manually
            out = {"task": f"P3x f{src}->f{dst}", "A": FLAT[src]["members"][a], "B": FLAT[src]["members"][b]}
            ones = np.ones(len(fs_["comp"]), bool)
            cls_est_s, clf = fit_classifier(ds.warm, ds.cls, ones)
            cls_est_d = clf.predict(dd.warm)
            fits, hlbest = run_policies(ds, ones, cls_est_s, None)
            rows = np.ones(dd.n, bool); ref = 0.5 * (dd.pa + dd.pb)
            sa, sb = wp(ds.y, ds.pa), wp(ds.y, ds.pb); pick = dd.pa if sa >= sb else dd.pb
            res = {}
            for nm, (par, f) in fits.items():
                fo = f(par, ds) if "P5" not in nm and "P6" not in nm else f(par, ds)
                pe = f(par, dd) if "P5" not in nm and "P6" not in nm else apply_table(par, dd, cls_est_d)
                r = bootstrap(fd, rows, pe, ref); rs = bootstrap(fd, rows, pe, pick)
                res[nm] = {"fit_wp": [wp(ds.y, fo)], "heldout_wp_pooled": r["candidate_wp"], "delta_vs_static": r["mean_diff"], "ci_vs_static": r["ci95"],
                           "delta_vs_best_single": rs["mean_diff"], "ci_vs_best_single": rs["ci95"]}
                out.setdefault("params", {})[nm] = summ(par)
            rb = bootstrap(fd, rows, pick, ref)
            res["static 0.5/0.5"] = {"fit_wp": [wp(ds.y, 0.5 * (ds.pa + ds.pb))], "heldout_wp_pooled": r["reference_wp"]}
            res["best single (by fit)"] = {"fit_wp": [max(sa, sb)], "heldout_wp_pooled": rb["candidate_wp"], "delta_vs_static": rb["mean_diff"], "ci_vs_static": rb["ci95"]}
            out["results"] = res; out["gate_hl_by_fit"] = hlbest
            out["cls_acc_on_target"] = float((cls_est_d == dd.cls).mean())
            (outdir / f"res_P3x_f{src}_to_f{dst}.json").write_text(json.dumps(out, indent=1))
            print("done", src, dst, flush=True)
