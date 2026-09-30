"""CSTAT1: fixed C descriptive statistics and analytic local scale derivatives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/"scripts"))
from fold_eval import pooled, sums
from paired_seed_eval import sha256

MEMBERS = ("ft256dH", "ft128H", "ft128s4H", "ftc500H")


def write(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def fingerprint(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns, s.st_ino]


def moments(s):
    w, sy, sp, syy, spp, syp = np.moveaxis(s, -1, 0)
    ww = np.maximum(w, 1e-30)
    my, mp = sy/ww, sp/ww
    vy, vp = syy/ww-my*my, spp/ww-mp*mp
    cov = syp/ww-my*mp
    return w, my, mp, vy, vp, cov


def coefficients(s):
    """Per-target derivative coefficients for sum(w*a), sum(w*y*a), sum(w*p*a)."""
    w, my, mp, vy, vp, cov = moments(s)
    coef = np.zeros((2, 3), dtype=np.float64)
    valid = (w >= 1e-8) & (vy > 1e-16) & (vp > 1e-16)
    for k in np.flatnonzero(valid):
        den = np.sqrt(vy[k]*vp[k])
        rho = cov[k]/den
        coef[k] = [(-my[k]/den+rho*mp[k]/vp[k])/w[k], 1/(w[k]*den), -rho/(w[k]*vp[k])]
    return coef, valid


def derivative(s, d):
    coef, valid = coefficients(s)
    by_target = np.einsum("kjq,kq->kj", d, coef)
    return {"by_target": by_target.tolist(), "mean_targets": by_target.mean(0).tolist(),
            "nondegenerate_targets": valid.tolist(),
            "common_rescale_derivative": float(by_target.mean(0).sum())}


def derivative_array(s, d):
    coef, valid = coefficients(s)
    if not valid.all():
        raise ValueError("degenerate bootstrap/leave-chain derivative; no silent omission")
    return np.einsum("kjq,kq->kj", d, coef).mean(0)


def block_stats(y, members, mask, blend=None):
    # members=(B,T,targets,4). Float32 order matches frozen I1 originalC exactly.
    if blend is None:
        blend = np.float32(.25)*(members[..., 0]+(members[..., 1]+members[..., 2]+members[..., 3]))
    r = members.astype(np.float64)
    yc = np.clip(y.astype(np.float64), -2, 2)
    e = blend.astype(np.float64)
    p = np.clip(e, -2, 2)
    w = np.abs(yc)*mask[..., None]
    a = .25*r*(np.abs(e)<2)[..., None]
    d = np.stack([(w[..., None]*a).sum(1),
                  (w[..., None]*yc[..., None]*a).sum(1),
                  (w[..., None]*p[..., None]*a).sum(1)], -1)
    preds = np.concatenate([r, e[..., None]], -1)
    tail = np.abs(preds)>2
    return {"s": sums(y, blend, mask), "d": d,
            "raw_sum": (w[..., None]*r).sum(1),
            "raw_y": (w[..., None]*yc[..., None]*r).sum(1),
            "raw_outer": np.einsum("btk,btkj,btkl->bkjl", w, r, r, optimize=True),
            "tail_count": (tail*mask[..., None, None]).sum(1),
            "tail_weight": (tail*w[..., None]).sum(1),
            "boundary_count": ((np.abs(e)==2)*mask[..., None]).sum(1),
            "count": mask.sum(1), "eligible": np.full(len(y), y.shape[1]-99, dtype=np.int64)}


def covariance_report(s, raw_sum, raw_y, raw_outer):
    w, my, _, vy, _, _ = moments(s)
    mean = raw_sum/w[:, None]
    covariance = raw_outer/w[:, None, None]-mean[..., :, None]*mean[..., None, :]
    std = np.sqrt(np.maximum(np.diagonal(covariance, axis1=-2, axis2=-1), 0))
    denominator = std[..., :, None]*std[..., None, :]
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator>0)
    covy = raw_y/w[:, None]-my[:, None]*mean
    residual_covariance = covariance-covy[..., :, None]-covy[..., None, :]+vy[:, None, None]
    return {"raw_mean_by_target_member": mean.tolist(), "raw_sd_by_target_member": std.tolist(),
            "raw_covariance_by_target": covariance.tolist(), "raw_correlation_by_target": correlation.tolist(),
            "raw_residual_covariance_by_target": residual_covariance.tolist(),
            "residual_definition": "raw member prediction minus clipped target, centered with target-specific metric weights"}


def class_decomposition(class_s, overall):
    w, my, mp, vy, vp, cov = moments(overall)
    cw, cy, cp, cvy, cvp, cc = moments(class_s)
    pi = cw/w[None, :]
    within = np.stack([(pi*cc).sum(0), (pi*cvy).sum(0), (pi*cvp).sum(0)])
    between = np.stack([(pi*(cy-my)*(cp-mp)).sum(0), (pi*(cy-my)**2).sum(0), (pi*(cp-mp)**2).sum(0)])
    actual = np.stack([cov, vy, vp])
    err = float(np.max(np.abs(within+between-actual)))
    if err > 1e-10:
        raise ValueError("class covariance decomposition failed")
    return {"order": ["cov_y_p", "var_y", "var_p"], "within_by_target": within.tolist(),
            "between_by_target": between.tolist(), "total_by_target": actual.tolist(),
            "reconstruction_max_abs_error": err, "class_weight_fraction_by_target": pi.tolist(),
            "class_target_mean": cy.tolist(), "class_clipped_ensemble_mean": cp.tolist(),
            "interpretation": "Between-class covariance is a legitimate part of pooled WP, not evidence of spurious signal."}


def concentration(mass):
    share = mass/mass.sum()
    ordered = np.sort(share)[::-1]
    return {"sum": float(mass.sum()), "max_share": float(ordered[0]),
            "top10_share": float(ordered[:10].sum()), "top10percent_share": float(ordered[:max(1, int(np.ceil(len(mass)*.1)))].sum()),
            "inverse_herfindahl_mass_count": float(1/np.square(share).sum()),
            "not_an_inferential_effective_sample_size": True}


def summarize(arrays, folds, chains, classes, repeats=1000):
    unique = sorted(set(zip(folds.tolist(), chains.tolist())))
    if any(len(np.unique(folds[chains == ch])) != 1 for ch in np.unique(chains)):
        raise ValueError("chain spans evaluation folds")
    grouped = {k: np.stack([v[(folds == f)&(chains == ch)].sum(0) for f, ch in unique]) for k, v in arrays.items()}
    gf = np.array([f for f, ch in unique])
    total = {k: v.sum(0) for k, v in grouped.items()}
    score = pooled(total["s"])  # The only real-data WP readout; no member or perturbed-mixture WP.
    if abs(score-0.7024238010052843) > 1e-9:
        raise ValueError(f"fixed C baseline mismatch: {score}")
    report = {"fixed_C_wp": score, "baseline_WP_reproductions": 1, "candidate_WP_comparisons": 0,
              "sequence_count": len(folds), "chain_count": len(unique), "members": list(MEMBERS), "scopes": {}}
    scopes = {"pooled": np.ones(len(unique), bool), **{f"fold_{f}": gf == f for f in (0, 1)}}
    for scope, pick in scopes.items():
        q = {k: v[pick].sum(0) for k, v in grouped.items()}
        deriv = derivative(q["s"], q["d"])
        if not all(deriv["nondegenerate_targets"]):
            raise ValueError("degenerate baseline derivative")
        report["scopes"][scope] = {
            "chains": int(pick.sum()), "derivative": deriv,
            "member_geometry": covariance_report(q["s"], q["raw_sum"], q["raw_y"], q["raw_outer"]),
            "row_tail_fraction_by_target_member_then_C": (q["tail_count"]/q["count"]).tolist(),
            "metric_tail_fraction_by_target_member_then_C": (q["tail_weight"]/q["s"][:, 0, None]).tolist(),
            "exact_abs_C_equals2_scored_count": q["boundary_count"].tolist(),
            "mass_concentration_by_target": [concentration(grouped["s"][pick, k, 0]) for k in (0, 1)],
            "pooled_scored_row_fraction": float(q["count"]/q["eligible"]),
            "macro_chain_scored_row_fraction": float(np.mean(grouped["count"][pick]/grouped["eligible"][pick])),
            "macro_chain_mean_metric_weight_per_scored_row": (grouped["s"][pick, :, 0]/np.maximum(grouped["count"][pick, None], 1)).mean(0).tolist(),
            "pooled_mean_metric_weight_per_scored_row": (q["s"][:, 0]/q["count"]).tolist()}
    cs = np.stack([arrays["s"][classes == c].sum(0) for c in range(4)])
    report["class_decomposition"] = class_decomposition(cs, total["s"])
    report["class_descriptive"] = []
    for c in range(4):
        pick = classes == c
        q = {k: v[pick].sum(0) for k, v in arrays.items()}
        report["class_descriptive"].append({"class": c, "sequences": int(pick.sum()), "scored_rows": int(q["count"]),
            "weight_by_target": q["s"][:, 0].tolist(),
            "member_geometry": covariance_report(q["s"], q["raw_sum"], q["raw_y"], q["raw_outer"]),
            "row_tail_fraction_by_target_member_then_C": (q["tail_count"]/max(q["count"], 1)).tolist()})
    # Chain contributions use GLOBAL centering. They sum exactly to the pooled analytic derivative.
    coef, _ = coefficients(total["s"])
    contributions = np.einsum("gkjq,kq->gj", grouped["d"], coef)/2
    base_deriv = np.asarray(report["scopes"]["pooled"]["derivative"]["mean_targets"])
    np.testing.assert_allclose(contributions.sum(0), base_deriv, atol=1e-12, rtol=1e-9)
    _, my, mp, _, _, _ = moments(total["s"])
    covcontrib = (grouped["s"][..., 5]-my*grouped["s"][..., 2]-mp*grouped["s"][..., 1]
                  +my*mp*grouped["s"][..., 0])
    loo = np.stack([derivative_array(total["s"]-grouped["s"][i], total["d"]-grouped["d"][i]) for i in range(len(unique))])
    report["chain_descriptive"] = [{"fold": f, "chain": ch, "scored_rows": int(grouped["count"][i]),
        "weight_by_target": grouped["s"][i, :, 0].tolist(), "centered_covariance_numerator_by_target": covcontrib[i].tolist(),
        "global_derivative_contribution": contributions[i].tolist(), "leave_chain_out_derivative": loo[i].tolist()}
        for i, (f, ch) in enumerate(unique)]
    report["leave_one_chain_sensitivity"] = {"derivative_min": loo.min(0).tolist(), "derivative_max": loo.max(0).tolist(),
        "max_absolute_derivative_change": np.abs(loo-base_deriv).max(0).tolist(),
        "most_influential_chain_by_member": [list(unique[i]) for i in np.abs(loo-base_deriv).argmax(0)]}
    rng = np.random.default_rng(2026100102)
    draws = {scope: [] for scope in scopes}
    for _ in range(repeats):
        byfold = {f: rng.choice(ix, len(ix), replace=True) for f in (0, 1) for ix in [np.flatnonzero(gf == f)]}
        picks = {"pooled": np.concatenate(list(byfold.values())), **{f"fold_{f}": ix for f, ix in byfold.items()}}
        for scope, ix in picks.items():
            draws[scope].append(derivative_array(grouped["s"][ix].sum(0), grouped["d"][ix].sum(0)))
    for scope, values in draws.items():
        boot = np.asarray(values)
        report["scopes"][scope]["conditional_chain_derivative_uncertainty"] = {
            "draws": repeats, "sd": boot.std(0).tolist(), "ci95": np.quantile(boot, [.025, .975], axis=0).T.tolist(),
            "positive_fraction": (boot > 0).mean(0).tolist()}
    report["uncertainty_scope"] = "Descriptive whole-chain bootstrap within fold, fixed selected models; no training, selection or transfer correction."
    report["interpretation"] = "Derivatives at member scales=(1,1,1,1), weights=.25. No coefficient fit, step size, candidate or gain estimate."
    return report


def self_test():
    rng = np.random.default_rng(61)
    y = rng.normal(size=(3, 101, 2))
    r = rng.normal(size=(3, 101, 2, 4))
    mask = rng.random((3, 101)) > .2
    for gain in (1., 7.):
        rr = r*gain
        e = rr.mean(-1)
        out = block_stats(y, rr, mask, blend=e)
        s, d = out["s"].sum(0), out["d"].sum(0)
        actual = np.asarray(derivative(s, d)["mean_targets"])
        expected = []
        eps = 1e-5
        for j in range(4):
            a = rr[..., j]*.25
            expected.append((pooled(sums(y, e+eps*a, mask).sum(0))-pooled(sums(y, e-eps*a, mask).sum(0)))/(2*eps))
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-7)
        if gain == 1.:
            assert np.max(np.abs(e)) < 2
            assert abs(actual.sum()) < 1e-12
    for yy, mm, label in ((y, np.zeros_like(mask), "all_masked"), (np.ones_like(y), mask, "constant_target")):
        o = block_stats(yy, r, mm, blend=r.mean(-1))
        d = derivative(o["s"].sum(0), o["d"].sum(0))
        assert not any(d["nondegenerate_targets"]), label
        assert d["mean_targets"] == [0., 0., 0., 0.]
    o = block_stats(y, np.ones_like(r), mask, blend=np.ones_like(y))
    assert not any(derivative(o["s"].sum(0), o["d"].sum(0))["nondegenerate_targets"])
    # Clipping kink: convention derivative zero at exact |e|=2; report such rows in real data.
    o = block_stats(y, r, mask, blend=np.full_like(y, 2.))
    assert not o["d"].any() and o["boundary_count"].sum() == 2*mask.sum()
    # Law of total covariance/variance for unequal class masses.
    o = block_stats(y, r, mask, blend=r.mean(-1))
    class_decomposition(o["s"], o["s"].sum(0))
    print("self_test passed: analytic derivatives with/without clipping, degeneracies, kink convention, covariance decomposition")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.source_root is None or args.out is None:
        ap.error("--source-root and --out required")
    out, inv = args.out, args.out.with_suffix(".inputs.json")
    if out.exists() or inv.exists():
        raise FileExistsError("immutable result/inventory exists")
    start = time.monotonic(); result = {"status": "started"}
    def alarm(signum, frame):
        raise TimeoutError(f"signal {signum}")
    signal.signal(signal.SIGTERM, alarm); signal.signal(signal.SIGALRM, alarm); signal.alarm(870)
    try:
        root = args.source_root
        target, split = root/"cache/valid_targets.npz", root/"cache/chain_folds.npz"
        paths = [target, split]+[root/"cache"/f"pred_{r}_f{f}_valid.npy" for f in (0, 1) for r in MEMBERS]
        before = {str(p): fingerprint(p) for p in paths}
        write(inv, {"inputs": {str(p): {"sha256": sha256(p), "fingerprint": before[str(p)]} for p in paths},
                    "code": {str(p): sha256(p) for p in [Path(__file__), Path(__file__).with_name("ensemble-statistics-plan.json"),
                                                                      REPO/"scripts/fold_eval.py", REPO/"scripts/paired_seed_eval.py"]},
                    "alignment": "Caches have no row IDs: rely on predict_run.py full valid_targets row order and frozen provenance; verify shape and split IDs."})
        result["inventory_sha256"] = sha256(inv)
        with np.load(target) as tg:
            rg, y, mask, classes = [tg[k] for k in ("row_group", "y", "mask", "market_class")]
        if y.shape != (len(rg), 20000, 2) or mask.shape != y.shape[:2] or len(np.unique(rg)) != len(rg):
            raise ValueError("target cache alignment invalid")
        cf = np.load(split); v = cf["split"] == "valid"
        info = {int(g): (int(f), int(ch), bool(p)) for g, f, ch, p in zip(cf["row_group"][v], cf["fold"][v], cf["chain"][v], cf["pure"][v])}
        accum, ff, cc, kk, selected = {}, [], [], [], []
        for fold in (0, 1):
            ix = np.array([i for i, g in enumerate(rg) if info[int(g)][0] == fold and info[int(g)][2]])
            ps = [np.load(root/"cache"/f"pred_{r}_f{fold}_valid.npy", mmap_mode="r") for r in MEMBERS]
            if any(p.shape != y.shape or p.dtype != np.float32 for p in ps):
                raise ValueError("prediction cache shape/dtype mismatch")
            for start_row in range(0, len(ix), 16):
                rows = ix[start_row:start_row+16]
                mm = mask[rows].astype(bool, copy=True); mm[:, :99] = False
                pp = np.stack([p[rows] for p in ps], axis=-1)
                if not np.isfinite(pp).all() or not np.isfinite(y[rows]).all():
                    raise ValueError("nonfinite cache")
                stats = block_stats(y[rows], pp, mm)
                for k, arr in stats.items():
                    accum.setdefault(k, []).append(arr)
                ff.extend([fold]*len(rows)); cc.extend(info[int(rg[i])][1] for i in rows)
                kk.extend(int(classes[i]) for i in rows); selected.extend(int(rg[i]) for i in rows)
        result.update(summarize({k: np.concatenate(v) for k, v in accum.items()}, np.array(ff), np.array(cc), np.array(kk)))
        if any(fingerprint(p) != fp for p, fp in before.items()):
            raise RuntimeError("input changed during calculation")
        result["selected_row_groups_sha256"] = hashlib.sha256(json.dumps(selected).encode()).hexdigest()
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "timeout" if isinstance(exc, TimeoutError) else "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0); result["wall_seconds"] = time.monotonic()-start
        write(out, result)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
