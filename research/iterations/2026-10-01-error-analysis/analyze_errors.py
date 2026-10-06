"""Fixed descriptive metric budgets and chain-cross-fitted gradient predictability.

No candidate prediction/ensemble, coefficient search, forecast fit or GPU use.
The ridge response is a fitting-half-defined post-clip metric-gradient component;
its heldout squared-error skill is not a new forecast WP or attainable WP gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261001
RIDGE = 0.01
BOOTSTRAPS = 2000
BLOCKS = (("prediction_confidence", 7), ("plus_current211", 218),
          ("plus_history30", 248), ("plus_member_deviations10", 258))
EXPECTED = np.array([.7023796676142179, .7023672255138177, .7017424296043642])
META = ["cache_position", "row_group", "step", "chain", "fold", "market_class_retrospective"]
PREDS = ["current_student4", "strong4", "fixed3", "shallow_seed3", "plain_seed3",
         "plain_seed4", "chunk500", "shallow_seed4_auxiliary"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, obj):
    path.write_text(json.dumps(serial(obj), indent=2, allow_nan=False) + "\n")


def metric_parameters(y, p):
    """Official clipped weighted-Pearson quantities, each target globally pooled."""
    y = np.clip(np.asarray(y, np.float64), -2, 2)
    p = np.clip(np.asarray(p, np.float64), -2, 2)
    if y.ndim == 1:
        y, p = y[:, None], p[:, None]
    assert y.shape == p.shape and np.isfinite(y).all() and np.isfinite(p).all()
    w = np.abs(y)
    W = w.sum(axis=0)
    denom = np.where(W > 0, W, 1)
    my, mp = (w*y).sum(0)/denom, (w*p).sum(0)/denom
    vy = np.maximum((w*y*y).sum(0)/denom-my*my, 0)
    vp = np.maximum((w*p*p).sum(0)/denom-mp*mp, 0)
    cov = (w*y*p).sum(0)/denom-my*mp
    sy, sp = np.sqrt(vy), np.sqrt(vp)
    valid = (W >= 1e-8) & (sy > 1e-8) & (sp > 1e-8)
    rho = np.zeros_like(W)
    rho[valid] = np.clip(cov[valid]/(sy[valid]*sp[valid]), -1, 1)
    return dict(W=W, mu_y=my, mu_p=mp, sy=sy, sp=sp, rho=rho,
                valid=valid, score=float(rho.mean()))


def encoded_bins(x, cuts):
    """Intervals [left,right); final bin extends to infinity."""
    return np.searchsorted(np.asarray(cuts, np.float64), x, side="right").astype(np.int16)


def interval_labels(cuts):
    edges = ["-inf", *[str(v) for v in cuts], "inf"]
    return [f"[{a},{b})" for a, b in zip(edges[:-1], edges[1:])]


def partitions(X, p, metadata, feature_names):
    ix = {n: i for i, n in enumerate(feature_names)}
    cl = metadata[:, 5]
    classes, codes = np.unique(cl, return_inverse=True)
    chain_keys, chain_code = np.unique(metadata[:, [4, 3]], axis=0, return_inverse=True)
    rows = [{"name": "class", "codes": codes, "labels": [str(c) for c in classes], "target": None},
            {"name": "fold", "codes": metadata[:, 4], "labels": ["0", "1"], "target": None},
            {"name": "chain", "codes": chain_code,
             "labels": [f"fold{f}/chain{c}" for f, c in chain_keys], "target": None}]
    definitions = [
        ("position", metadata[:, 2], [2000, 8000]),
        ("phase_chunk2000", metadata[:, 2] % 2000, [100, 500, 1500]),
        ("i0px_rate256", X[:, ix["rate_i0px_rows256"]], [.01, .05, .2]),
        ("i0px_age_rows", np.rint(np.expm1(X[:, ix["log_since_i0px"]].astype(np.float64))), [1, 4, 16, 64, 256, 1024]),
    ]
    for name, v, cuts in definitions:
        rows.append(dict(name=name, codes=encoded_bins(v, cuts), labels=interval_labels(cuts), target=None))
    for t in range(2):
        for name, v, cuts in [
            ("prediction_confidence", np.abs(np.clip(p[:, 0, t], -2, 2)), [.1, .25, .5, 1]),
            ("current_member_sd", np.std(p[:, 3:7, t].astype(np.float64), axis=1), [.025, .05, .1, .2]),
        ]:
            rows.append(dict(name=f"{name}_target{t}", codes=encoded_bins(v, cuts), labels=interval_labels(cuts), target=t))
    return rows, chain_keys, chain_code


def group_moments(w, u, z, codes, groups):
    """Global-normalized moments partitioned without subgroup restandardization."""
    count = np.bincount(codes, minlength=groups)
    mass = np.bincount(codes, weights=w, minlength=groups)
    out = np.column_stack([mass] + [np.bincount(codes, weights=w*v, minlength=groups)
                                   for v in (u, z, u*u, z*z, u*z, (u-z)**2)])
    return count, out


def component_record(a, W):
    mass, su, sz, suu, szz, suz, se = a
    if mass <= 0:
        return dict(weight=0., weight_share=0., budget=0., error_intensity=None,
                    mean_mismatch=0., amplitude_mismatch=0., covariance_shortfall=0.)
    mu, mz = su/mass, sz/mass
    vu, vz = max(suu/mass-mu*mu, 0), max(szz/mass-mz*mz, 0)
    co = suz/mass-mu*mz
    mean = (mu-mz)**2
    amplitude = (np.sqrt(vu)-np.sqrt(vz))**2
    fluctuation = 2*(np.sqrt(vu*vz)-co)
    assert abs(mean+amplitude+fluctuation-se/mass) < 2e-9
    factor = .25*mass/W
    return dict(weight=mass, weight_share=mass/W, budget=.25*se/W,
                error_intensity=se/mass, mean_mismatch=factor*mean,
                amplitude_mismatch=factor*amplitude, covariance_shortfall=factor*fluctuation,
                global_standardized_mean_y=mu, global_standardized_mean_p=mz,
                global_standardized_sd_y=np.sqrt(vu), global_standardized_sd_p=np.sqrt(vz))


def analyze_budgets(y, p, metadata, X, feature_names, names):
    yc = np.clip(y.astype(np.float64), -2, 2)
    w = np.abs(yc)
    params = [metric_parameters(yc, p[:, m]) for m in range(p.shape[1])]
    for par in params:
        assert np.all(par["valid"]), "Degenerate global metric: budget not defined; stop, do not floor."
    part, chain_keys, chain_code = partitions(X, p, metadata, feature_names)
    result = {n: {"metric": par, "partitions": {}} for n, par in zip(names, params)}
    chain_budget = np.zeros((len(chain_keys), len(names), 2))
    chain_mass = np.zeros((len(chain_keys), 2))
    for m, (name, par) in enumerate(zip(names, params)):
        for t in range(2):
            u = (yc[:, t]-par["mu_y"][t])/par["sy"][t]
            raw = p[:, m, t].astype(np.float64)
            z = (np.clip(raw, -2, 2)-par["mu_p"][t])/par["sp"][t]
            W = par["W"][t]
            total = .25*np.dot(w[:, t], (u-z)**2)/W
            assert abs(total-.5*(1-par["rho"][t])) < 1e-10
            result[name].setdefault("clipping", {})[f"target{t}"] = {
                "outside_row_fraction": float(np.mean(np.abs(raw)>2)),
                "outside_weight_fraction": float(w[np.abs(raw)>2, t].sum()/W),
                "boundary_row_fraction": float(np.mean(np.abs(raw)==2)),
                "boundary_weight_fraction": float(w[np.abs(raw)==2, t].sum()/W),
                "target_boundary_weight_fraction": float(w[np.abs(yc[:, t])==2, t].sum()/W),
            }
            for spec in part:
                if spec["target"] is not None and spec["target"] != t:
                    continue
                count, mom = group_moments(w[:, t], u, z, spec["codes"], len(spec["labels"]))
                records = []
                for lab, n, a in zip(spec["labels"], count, mom):
                    rec = dict(label=lab, rows=int(n), **component_record(a, W))
                    records.append(rec)
                assert abs(sum(r["budget"] for r in records)-total) < 1e-10
                result[name]["partitions"].setdefault(spec["name"], {})[f"target{t}"] = records
                if spec["name"] == "chain":
                    chain_budget[:, m, t] = .25*mom[:, 6]/W
                    if m == 0:
                        chain_mass[:, t] = mom[:, 0]/W
    primary_delta = {}
    for m in (1, 2):
        delta = chain_budget[:, m].sum(1)-chain_budget[:, 0].sum(1)
        assert abs(delta.sum()-(params[0]["score"]-params[m]["score"])) < 1e-10
        primary_delta[names[m]] = dict(WP_difference=params[0]["score"]-params[m]["score"],
                                      chain_delta_min=delta.min(), chain_delta_max=delta.max(),
                                      chains_primary_lower_deficit=int((delta>0).sum()),
                                      chain_sum=delta.sum())
    concentration = {}
    for m, name in enumerate(names):
        b = chain_budget[:, m].sum(1)
        idx = np.argsort(-b, kind="stable")
        concentration[name] = {str(k): {
            "deficit_share": b[idx[:k]].sum()/b.sum(),
            "metric_weight_share_same_chains": chain_mass[idx[:k]].mean(1).sum(),
            "chains": chain_keys[idx[:k]].tolist(),
        } for k in (1, 5, 10, 25, 50)}
    fold_metrics = {str(f): {name: metric_parameters(y[metadata[:, 4]==f], p[metadata[:, 4]==f, m])
                            for m, name in enumerate(names)} for f in (0, 1)}
    return dict(models=result, profile_differences=primary_delta, concentration=concentration,
                fold_standalone_metrics=fold_metrics), params, chain_budget, chain_mass, chain_keys


def consensus_analysis(y, p, metadata, params):
    yc = np.clip(y.astype(np.float64), -2, 2)
    w = np.abs(yc)
    cl = metadata[:, 5]
    result = {}
    for label, members in (("three_fixed_profiles", (0, 1, 2)), ("four_current_members", (3, 4, 5, 6))):
        rows = []
        for t in range(2):
            u = (yc[:, t]-params[0]["mu_y"][t])/params[0]["sy"][t]
            zs = np.column_stack([(np.clip(p[:, m, t].astype(np.float64), -2, 2)-params[m]["mu_p"][t])/params[m]["sp"][t] for m in members])
            zb = zs.mean(1)
            zstudent = (np.clip(p[:, 0, t].astype(np.float64), -2, 2)-params[0]["mu_p"][t])/params[0]["sp"][t]
            r, d = u-zb, zb-zstudent
            energies = np.column_stack([r*r, ((zs-zb[:, None])**2).mean(1), d*d, 2*r*d,
                                        ((u[:, None]-zs)**2).mean(1), (u-zstudent)**2])
            assert np.max(np.abs(energies[:, 0]+energies[:, 1]-energies[:, 4])) < 1e-10
            assert np.max(np.abs(energies[:, 0]+energies[:, 2]+energies[:, 3]-energies[:, 5])) < 1e-10
            for c in [None, *np.unique(cl).tolist()]:
                keep = np.ones(len(y), bool) if c is None else cl == c
                vals = .25*(w[keep, t, None]*energies[keep]).sum(0)/params[0]["W"][t]
                rows.append(dict(target=t, group="all" if c is None else f"class{c}",
                                 consensus_energy=vals[0], dispersion_energy=vals[1],
                                 student_deviation_energy=vals[2], student_signed_cross_term=vals[3],
                                 mean_member_deficit=vals[4], student_deficit=vals[5]))
        result[label] = rows
    return {"groups": result, "interpretation": "Weighted standardized-output identities; consensus is not an ensemble candidate, and shared error is not irreducible noise. Profiles/members are strongly dependent."}


def make_halves(metadata, class_counts, class_values, seed=SEED):
    assert metadata.shape[1] == 2 and len(metadata) == len(class_counts)
    dominant = np.asarray(class_values)[np.argmax(class_counts, axis=1)]
    half = np.full(len(metadata), -1, np.int8)
    rng = np.random.default_rng(seed)
    for f in sorted(np.unique(metadata[:, 0])):
        for c in sorted(np.unique(dominant[metadata[:, 0] == f])):
            idx = np.flatnonzero((metadata[:, 0] == f) & (dominant == c))
            idx = idx[np.argsort(metadata[idx, 1], kind="stable")]
            idx = rng.permutation(idx)
            half[idx] = np.arange(len(idx)) % 2
    assert (half >= 0).all()
    return half, dominant


def q_definition(G, target, prediction_index=None, target_index=None):
    """An affine response vector with fit-only means, SDs and correlation."""
    pidx = 1+target if prediction_index is None else prediction_index
    yidx = 259+target if target_index is None else target_index
    W = float(G[0, 0])
    assert W > 1e-8
    my, mp = G[0, yidx]/W, G[0, pidx]/W
    vy, vp = G[yidx, yidx]/W-my*my, G[pidx, pidx]/W-mp*mp
    assert vy > 1e-16 and vp > 1e-16, "q undefined: constant fitting target/prediction"
    sy, sp = np.sqrt(vy), np.sqrt(vp)
    rho = (G[yidx, pidx]/W-my*mp)/(sy*sp)
    assert abs(rho) < 1-1e-10, "q normalization undefined or unstable at perfect correlation"
    qsd = np.sqrt(1-rho*rho)
    alpha = np.zeros(G.shape[0])
    alpha[yidx] = 1/(sy*qsd)
    alpha[pidx] = -rho/(sp*qsd)
    alpha[0] = (-my/sy+rho*mp/sp)/qsd
    qm = float(G[0]@alpha/W)
    qv = float(alpha@G@alpha/W-qm*qm)
    assert abs(qm) < 1e-8 and abs(qv-1) < 1e-7
    return alpha, dict(W=W, mu_y=my, mu_p=mp, sy=sy, sp=sp, rho=rho,
                       q_scale=qsd, q_fit_mean=qm, q_fit_variance=qv)


def fit_ridge(G, alpha, indices, lam=RIDGE):
    """min E_fit[w (q-intercept-Z beta)^2]/E_fit[w] + lambda ||beta||²."""
    indices = np.asarray(indices, int)
    W = G[0, 0]
    mu = G[0, indices]/W
    cov = G[np.ix_(indices, indices)]/W-np.outer(mu, mu)
    var = np.maximum(np.diag(cov), 0)
    # Exact/roundoff-constant columns map to zero, as a fixed numerical convention.
    tolerance = 1e-12*np.maximum(np.abs(G[indices, indices]/W), 1)
    active = var > tolerance
    scale = np.ones(len(indices))
    scale[active] = np.sqrt(var[active])
    A = cov/np.outer(scale, scale)
    A[~active] = 0
    A[:, ~active] = 0
    qm = float(G[0]@alpha/W)
    b = (G[indices]@alpha/W-mu*qm)/scale
    b[~active] = 0
    normalized = np.linalg.solve(A+lam*np.eye(len(indices)), b)
    beta = np.zeros(G.shape[0])
    beta[indices] = normalized/scale
    beta[0] = qm-mu@beta[indices]
    return beta, dict(features=len(indices), active_columns=int(active.sum()),
                     constant_columns=indices[~active], coefficient_l2_normalized=float(np.linalg.norm(normalized)),
                     fit_q_mean=qm, fit_weight=float(W))


def quadratic_chain(G, coeff):
    out = np.einsum("i,nij,j->n", coeff, G, coeff, optimize=True)
    tolerance = 2e-9*np.maximum(G[:, 0, 0], 1)
    assert np.all(out >= -tolerance), "negative squared error outside numerical tolerance"
    return np.maximum(out, 0)


def crossfit_grams(grams, metadata, class_counts, class_values, bootstrap=BOOTSTRAPS):
    n, targets, dim, _ = grams.shape
    assert targets == 2 and dim == 261
    halves, dominant = make_halves(metadata, class_counts, class_values)
    sse = np.full((n, 2, len(BLOCKS)), np.nan)
    baseline = np.full((n, 2), np.nan)
    fit_records = []
    coeffs, alphas = [], []
    for fold in (0, 1):
        for held in (0, 1):
            train = (metadata[:, 0] == fold) & (halves != held)
            test = (metadata[:, 0] == fold) & (halves == held)
            assert train.any() and test.any() and not np.any(train & test)
            for t in range(2):
                G = grams[train, t].sum(axis=0)
                alpha, qinfo = q_definition(G, t)
                const = np.zeros(dim)
                const[0] = qinfo["q_fit_mean"]
                baseline[test, t] = quadratic_chain(grams[test, t], alpha-const)
                for b, (name, k) in enumerate(BLOCKS):
                    beta, info = fit_ridge(G, alpha, np.arange(1, k+1))
                    sse[test, t, b] = quadratic_chain(grams[test, t], alpha-beta)
                    fit_records.append(dict(fold=fold, heldout_half=held, target=t, block=name,
                                            fitting_chains=int(train.sum()), heldout_chains=int(test.sum()),
                                            q=qinfo, ridge=info))
                    coeffs.append(beta)
                    alphas.append(alpha)
    assert np.isfinite(sse).all() and np.isfinite(baseline).all()
    weights = grams[:, :, 0, 0]

    def summarize(idx):
        bs = baseline[idx].sum(0)
        ss = sse[idx].sum(0)
        ww = weights[idx].sum(0)
        return dict(chains=int(idx.sum()), weight=ww, baseline_SSE=bs, baseline_MSE=bs/ww,
                    model_SSE=ss.T, model_MSE=(ss/ww[:, None]).T,
                    skill_R2=(1-ss/bs[:, None]).T,
                    improvement_MSE=((bs[:, None]-ss)/ww[:, None]).T,
                    equal_target_mean_skill=(1-ss/bs[:, None]).mean(0))

    summary = {"pooled": summarize(np.ones(n, bool))}
    for f in (0, 1):
        summary[f"fold{f}"] = summarize(metadata[:, 0] == f)
        for h in (0, 1):
            summary[f"fold{f}_half{h}"] = summarize((metadata[:, 0] == f) & (halves == h))
    rng = np.random.default_rng(SEED+1)
    draws = np.zeros((bootstrap, n), np.int16)
    for f in (0, 1):
        for h in (0, 1):
            idx = np.flatnonzero((metadata[:, 0] == f) & (halves == h))
            sampled = rng.integers(len(idx), size=(bootstrap, len(idx)))
            for b in range(bootstrap):
                draws[b, idx] = np.bincount(sampled[b], minlength=len(idx))
    boot_base = draws@baseline
    boot_sse = (draws@sse.reshape(n, -1)).reshape(bootstrap, 2, len(BLOCKS))
    skills = 1-boot_sse/boot_base[:, :, None]
    incremental = np.diff(skills, axis=2)
    ci = dict(skill_by_target=np.quantile(skills, [.025, .975], axis=0).transpose(2, 1, 0),
              skill_equal_target_mean=np.quantile(skills.mean(1), [.025, .975], axis=0).T,
              incremental_skill_by_target=np.quantile(incremental, [.025, .975], axis=0).transpose(2, 1, 0),
              incremental_equal_target_mean=np.quantile(incremental.mean(1), [.025, .975], axis=0).T)
    fold_ci = {}
    for f in (0, 1):
        keep = metadata[:, 0] == f
        bb = draws[:, keep]@baseline[keep]
        ss = (draws[:, keep]@sse[keep].reshape(keep.sum(), -1)).reshape(bootstrap, 2, len(BLOCKS))
        sk = 1-ss/bb[:, :, None]
        fold_ci[f"fold{f}"] = dict(
            skill_by_target=np.quantile(sk, [.025, .975], axis=0).transpose(2, 1, 0),
            skill_equal_target_mean=np.quantile(sk.mean(1), [.025, .975], axis=0).T,
            incremental_equal_target_mean=np.quantile(np.diff(sk, axis=2).mean(1), [.025, .975], axis=0).T)
    result = dict(blocks=[n for n, _ in BLOCKS], lambda_normalized=RIDGE, summaries=summary,
                  conditional_chain_bootstrap=dict(draws=bootstrap, seed=SEED+1,
                      strata="outer fold and fixed heldout inner half", intervals_95=ci, fold_intervals_95=fold_ci,
                      scope="conditional on fitted q definitions, feature normalizers and ridge coefficients; no refitting, selection/seed/domain-shift uncertainty"),
                  fits=fit_records, splits=dict(seed=SEED, chain_keys=metadata, half=halves,
                      dominant_retrospective_class=dominant, mixed_class_chains=int((np.count_nonzero(class_counts, axis=1)>1).sum())),
                  interpretation="Post-clip fit-defined gradient-response skill, not candidate WP, attainable WP improvement, or proof that remaining noise is irreducible.")
    return result, dict(sse=sse, baseline=baseline, weights=weights, metadata=metadata,
                        halves=halves, coefficients=np.asarray(coeffs), q_coefficients=np.asarray(alphas))


def concise_report(result):
    models = result["budget"]["models"]
    student = models[PREDS[0]]
    lines = ["# Fixed error diagnosis", "", "These are descriptive results on previously exposed OOF chains. No forecast model, ensemble weights or new candidate WP were fitted.", "",
             "| Profile | WP | Deficit |", "|---|---:|---:|"]
    for name in PREDS[:3]:
        s = models[name]["metric"]["score"]
        lines.append(f"| {name} | {s:.12f} | {1-s:.8f} |")
    lines += ["", "| Retrospective class | Deficit share | Metric-weight share | Mean mismatch | Amplitude mismatch | Within-class covariance shortfall |",
              "|---|---:|---:|---:|---:|---:|"]
    a, b = [student["partitions"]["class"][f"target{t}"] for t in range(2)]
    for r, s in zip(a, b):
        vals = [(r[k]+s[k]) for k in ("budget", "mean_mismatch", "amplitude_mismatch", "covariance_shortfall")]
        mass = .5*(r["weight_share"]+s["weight_share"])
        lines.append(f"| {r['label']} | {vals[0]/(1-student['metric']['score']):.2%} | {mass:.2%} | {vals[1]:.6f} | {vals[2]:.6f} | {vals[3]:.6f} |")
    lines += ["", "| Gradient-response probe | t0 skill | t1 skill | Mean skill | Conditional 95% interval |", "|---|---:|---:|---:|---| "]
    ridge = result["probe"]
    pooled = ridge["summaries"]["pooled"]
    cis = ridge["conditional_chain_bootstrap"]["intervals_95"]["skill_equal_target_mean"]
    for i, name in enumerate(ridge["blocks"]):
        r = pooled["skill_R2"][i]
        lines.append(f"| {name} | {r[0]:.4%} | {r[1]:.4%} | {pooled['equal_target_mean_skill'][i]:.4%} | [{cis[i,0]:.4%}, {cis[i,1]:.4%}] |")
    lines += ["", "The probe uses fitting-half normalization of its response and predictors. Its skill is a heldout weighted squared-error diagnostic in post-clip gradient space, not a WP gain or deployable correction. Retrospective class is used for stratification and reporting only. Shared errors among highly dependent profiles do not establish an irreducible ceiling.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--plan-sha256", required=True)
    ap.add_argument("--source-sha256", required=True)
    a = ap.parse_args()
    start = time.monotonic()
    assert sha(__file__) == a.source_sha256 and sha(a.plan) == a.plan_sha256
    assert not os.environ.get("CUDA_VISIBLE_DEVICES", ""), "CPU-only analysis"
    assert not a.output_dir.exists()
    receipt_path = a.input_dir/"receipt.json"
    receipt = json.loads(receipt_path.read_text())
    hashes = {}
    for name in ("scored-rows.npz", "chain-gram.npz"):
        p = a.input_dir/name
        h = sha(p)
        assert h == receipt["outputs"][name]["sha256"], (name, "input digest mismatch")
        hashes[name] = dict(sha256=h, bytes=p.stat().st_size)
    a.output_dir.mkdir(parents=True)
    started = dict(source_sha256=a.source_sha256, plan_sha256=a.plan_sha256,
                   input_receipt_sha256=sha(receipt_path), inputs=hashes,
                   UTC=datetime.now(timezone.utc).isoformat(), PID=os.getpid(),
                   cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    write_json(a.output_dir/"started.json", started)
    try:
        with np.load(a.input_dir/"scored-rows.npz", allow_pickle=False) as z:
            y, p, metadata, X = z["y"], z["predictions"], z["metadata"], z["X"]
            feature_names = z["feature_names"].tolist()
            names = z["prediction_names"].tolist()
            assert z["metadata_names"].tolist() == META and names == PREDS
        assert y.shape == (1614484, 2) and p.shape == (len(y), 8, 2) and X.shape == (len(y), 30)
        assert metadata.shape == (len(y), 6) and np.all(metadata[:, 2] >= 99)
        assert np.isfinite(y).all() and np.isfinite(p).all() and np.isfinite(X).all()
        budget, params, chain_budget, chain_mass, chain_keys = analyze_budgets(y, p, metadata, X, feature_names, names)
        scores = np.asarray([d["score"] for d in params[:3]])
        assert np.max(np.abs(scores-EXPECTED)) < 2e-10, ("WP reproduction failed", scores, EXPECTED)
        consensus = consensus_analysis(y, p, metadata, params)
        with np.load(a.input_dir/"chain-gram.npz", allow_pickle=False) as z:
            grams = z["gram"]
            gm = z["chain_metadata"]
            class_counts = z["chain_class_scored_counts"]
            class_values = z["class_values"]
            gram_names = z["feature_names"].tolist()
            counts = z["chain_scored_counts"]
        assert grams.shape == (250, 2, 261, 261) and gm.shape == (250, 2)
        assert np.isfinite(grams).all()
        # Extractor preserves first-seen chain order; join by identifiers, never position.
        lookup = {tuple(key): i for i, key in enumerate(gm)}
        assert len(lookup) == len(gm) and set(lookup) == set(map(tuple, chain_keys))
        order = np.asarray([lookup[tuple(key)] for key in chain_keys])
        grams, gm, class_counts, counts = grams[order], gm[order], class_counts[order], counts[order]
        assert counts.sum() == len(y) and class_counts.sum() == len(y)
        # Independently bind Gram weights and target/prediction sufficient sums to rows.
        for t in range(2):
            G = grams[:, t].sum(0)
            par = params[0]
            expected = [par["W"][t], par["mu_y"][t], par["mu_p"][t]]
            actual = [G[0, 0], G[0, 259+t]/G[0, 0], G[0, 1+t]/G[0, 0]]
            assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10), ("Gram/row metric mismatch", actual, expected)
            assert np.allclose(grams[:, t, 0, 0]/par["W"][t], chain_mass[:, t], rtol=1e-9, atol=1e-12)
        probe, arrays = crossfit_grams(grams, gm, class_counts, class_values)
        result = dict(status="COMPLETE_FIXED_ERROR_DIAGNOSTICS_NO_FORECAST_CANDIDATE", provenance=started,
                      rows=len(y), chains=len(gm), budget=budget, consensus=consensus, probe=probe,
                      gram_feature_names=gram_names,
                      numerical=dict(WP_expected=EXPECTED, WP_observed=scores, WP_max_abs_error=float(np.max(np.abs(scores-EXPECTED))),
                                     additive_identities="all target/group, component and consensus identities asserted"),
                      scope="Previously exposed GPU/AMP OOF caches; no native-serving equivalence or unseen validation claim.",
                      elapsed_seconds=time.monotonic()-start, peak_RSS_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        np.savez_compressed(a.output_dir/"probe-sufficient-results.npz", **arrays,
                            chain_budget=chain_budget, chain_mass=chain_mass, chain_keys=chain_keys)
        write_json(a.output_dir/"results.json", result)
        (a.output_dir/"tables.md").write_text(concise_report(result))
        assert sha(__file__) == a.source_sha256 and sha(a.plan) == a.plan_sha256
        write_json(a.output_dir/"receipt.json", dict(status=result["status"], **started,
                   elapsed_seconds=result["elapsed_seconds"], outputs={p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                    for p in sorted(a.output_dir.iterdir()) if p.is_file()}))
        print(json.dumps(serial(dict(status=result["status"], WP=scores, probe_mean_skill=probe["summaries"]["pooled"]["equal_target_mean_skill"],
                                    elapsed_seconds=result["elapsed_seconds"]))))
    except Exception as exc:
        write_json(a.output_dir/"failure.json", dict(status="FAILED_NO_RECIPE_RETRY", exception_type=type(exc).__name__,
                   message=str(exc), elapsed_seconds=time.monotonic()-start, provenance=started))
        raise


if __name__ == "__main__":
    main()
