"""Allocated SELRISK01 CPU analysis. No forecast fitting, prediction editing or WP."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import resource
import signal
import time
import warnings
import numpy as np
from common import (BINS, SELECTORS, load_allocation, role_indices, derangement,
                    outcome_features, sha)
from gradient_core import sufficient, gradient, cross_panel_comparisons

HGB_ARGS = dict(max_iter=64, learning_rate=.1, max_depth=3, max_leaf_nodes=7,
                min_samples_leaf=64, l2_regularization=10, max_bins=32,
                early_stopping=False, random_state=0)
CALIBRATION_ARGS = dict(C=1, solver="lbfgs", tol=1e-8, max_iter=200)


def design(X, Y, indices, panel, kind, donors=None):
    x = np.asarray(X[indices, panel], dtype=np.float32).reshape(-1, 341)
    if kind == "X":
        assert donors is None
        return x
    assert kind in ("XY", "SHUFFLED")
    source = donors if kind == "SHUFFLED" else indices
    assert source is not None
    extra = outcome_features(Y[source, panel]).reshape(-1, 6)
    return np.concatenate([x, extra], axis=1)


def weight_vector(indices):
    w = np.tile(np.diff(BINS), len(indices)).astype(np.float64)
    return w / w.mean()


def summarize(numerator, denominator, squared_mass, indices):
    n = numerator[indices].sum(0)
    d = denominator[indices].sum(0)
    sq = squared_mass[indices].sum(0)
    panels, panel_report = [], {}
    for panel, label in enumerate(("A", "B")):
        g = {name: gradient(n[panel, j], d[panel, j]) for j, name in enumerate(SELECTORS)}
        panels.append(g)
        errors = {name: g[name] - g["M"] for name in SELECTORS[1:]}
        panel_report[label] = {
            "true_mask_gradient_norm": float(np.linalg.norm(g["M"])),
            "error_norm": {name: float(np.linalg.norm(e)) for name, e in errors.items()},
            "error_norm_by_target": {name: np.linalg.norm(e, axis=1).tolist() for name, e in errors.items()},
            "cosine_to_true_mask": {name: (float(np.sum(g[name] * g["M"]) /
                (np.linalg.norm(g[name]) * np.linalg.norm(g["M"])))
                if np.linalg.norm(g[name]) * np.linalg.norm(g["M"]) > 0 else None) for name in errors},
            "denominator": dict(zip(SELECTORS, d[panel].tolist())),
            "row_mass_ESS": {name: (float(d[panel, j]**2 / sq[panel, j]) if sq[panel, j] > 0 else 0.)
                             for j, name in enumerate(SELECTORS)},
            "squared_error_reduction_vs_control": {name: float(np.sum(errors[name]**2) - np.sum(errors["XY"]**2))
                                                    for name in ("X", "SHUFFLED")},
        }
    return {"cross_panel_error_reduction_vs_control": cross_panel_comparisons(*panels),
            "panels": panel_report}


def bootstrap(numerator, denominator, classes, indices, draws=1000, seed=0):
    """Same chain draws for all selector probabilities and both row panels."""
    rng = np.random.default_rng(seed)
    strata = [indices[classes[indices] == c] for c in range(4)]
    assert all(len(s) == 16 for s in strata)
    result = {name: [] for name in ("X", "SHUFFLED")}
    for _ in range(draws):
        pick = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
        n, d = numerator[pick].sum(0), denominator[pick].sum(0)
        g = [{name: gradient(n[panel, j], d[panel, j]) for j, name in enumerate(SELECTORS)}
             for panel in range(2)]
        contrasts = cross_panel_comparisons(*g)
        for name in result:
            result[name].append(contrasts[name])
    return {name: {"conditional_ci95": np.quantile(values, [.025, .975]).tolist(),
                   "conditional_one_sided95_lower": float(np.quantile(values, .05)),
                   "conditional_sd": float(np.std(values)),
                   "fraction_nonpositive": float(np.mean(np.asarray(values) <= 0))}
            for name, values in result.items()}


def analyze(X, Y, M, records, progress=None):
    """Pure array interface, also exercised by generated-only CPU smoke."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    assert X.shape == (128, 3, 256, 341) and Y.shape == (128, 3, 256, 2) and M.shape == (128, 3, 256)
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    q = np.full((128, 2, 3, 256), np.nan, dtype=np.float64)
    audit_count = np.zeros(128, dtype=int)
    fits = []
    with threadpool_limits(limits=2):
        for fold in (0, 1):
            for rotation in range(4):
                roles = role_indices(records, fold, rotation)
                donors = {role: derangement(records, ids, fold, rotation, role) for role, ids in roles.items()}
                fit, cal, audit = (roles[role] for role in ("fit", "calibration", "audit"))
                assert len(np.unique(M[fit, 0])) == len(np.unique(M[cal, 0])) == 2
                for j, kind in enumerate(SELECTORS[1:]):
                    ds = lambda role: donors[role] if kind == "SHUFFLED" else None
                    fit_x = design(X, Y, fit, 0, kind, ds("fit"))
                    cal_x = design(X, Y, cal, 0, kind, ds("calibration"))
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        selector = HistGradientBoostingClassifier(**HGB_ARGS)
                        selector.fit(fit_x, M[fit, 0].reshape(-1), sample_weight=weight_vector(fit))
                        calibration = LogisticRegression(**CALIBRATION_ARGS)
                        calibration.fit(selector.decision_function(cal_x)[:, None], M[cal, 0].reshape(-1),
                                        sample_weight=weight_vector(cal))
                    for panel in (1, 2):
                        audit_x = design(X, Y, audit, panel, kind, ds("audit"))
                        logits = selector.decision_function(audit_x)[:, None]
                        probs = calibration.predict_proba(logits)[:, 1]
                        q[audit, panel - 1, j] = np.clip(probs.reshape(len(audit), 256), 1e-4, 1 - 1e-4)
                    fits.append({"fold": fold, "rotation": rotation, "kind": kind,
                                 "fit_indices": fit.tolist(), "calibration_indices": cal.tolist(),
                                 "audit_indices": audit.tolist(), "fit_panel": "F", "calibration_panel": "F",
                                 "prediction_panels": ["A", "B"],
                                 "donors": {role: ids.tolist() for role, ids in donors.items()} if kind == "SHUFFLED" else None,
                                 "hgb_iterations": int(selector.n_iter_),
                                 "calibration_iterations": calibration.n_iter_.tolist(),
                                 "warnings": [str(w.message) for w in caught]})
                    if progress:
                        progress(fits[-1])
                    del fit_x, cal_x, audit_x, selector, calibration
                audit_count[audit] += 1
    assert np.all(audit_count == 1) and np.isfinite(q).all() and len(fits) == 24
    numerator = np.empty((128, 2, 4, 2, 129), dtype=np.float64)
    denominator = np.empty((128, 2, 4), dtype=np.float64)
    squared_mass = np.empty_like(denominator)
    loss_numerator = np.empty_like(denominator)
    a = np.diff(BINS).astype(np.float64)
    for chain in range(128):
        for panel in range(2):
            actual_panel = panel + 1
            for j, name in enumerate(SELECTORS):
                probability = M[chain, actual_panel].astype(float) if name == "M" else q[chain, panel, j-1]
                n, d, loss = sufficient(X[chain, actual_panel, :, 211:339], X[chain, actual_panel, :, 339:341],
                                        Y[chain, actual_panel], probability, a)
                numerator[chain, panel, j] = n
                denominator[chain, panel, j] = d
                loss_numerator[chain, panel, j] = loss
                mass = a * probability * abs(np.clip(Y[chain, actual_panel], -2, 2)).sum(1)
                squared_mass[chain, panel, j] = np.square(mass).sum()
    fold_id = np.asarray([r["fold"] for r in records])
    classes = np.asarray([r["class"] for r in records])
    groups = np.asarray([r["group"] for r in records])
    summaries = {}
    for fold in (0, 1):
        ids = np.flatnonzero(fold_id == fold)
        summaries[str(fold)] = summarize(numerator, denominator, squared_mass, ids)
        summaries[str(fold)]["conditional_chain_bootstrap"] = bootstrap(numerator, denominator, classes, ids)
        summaries[str(fold)]["by_class"] = {str(c): summarize(numerator, denominator, squared_mass, ids[classes[ids] == c])
                                             for c in range(4)}
        summaries[str(fold)]["by_rotation"] = {str(g): summarize(numerator, denominator, squared_mass, ids[groups[ids] == g])
                                                for g in range(4)}
    result = {"folds": summaries, "selector_fits": fits, "hgb_arguments": HGB_ARGS,
              "calibration_arguments": CALIBRATION_ARGS, "bootstrap_draws": 1000, "bootstrap_seed": 0,
              "interpretation": "Descriptive finite-model loss-gradient surrogate; root assesses size, uncertainty and heterogeneity. No automatic threshold, MNAR claim or next fit.",
              "uncertainty_scope": "Paired chains conditional on fitted selectors; independent audit row panels, finite ratio bias remains; selector-refit/seed/selection/domain uncertainty not estimated.",
              "subgroup_scope": "Class/rotation diagnostics use subset denominators; they are not additive contributions.",
              "forecast_fits": 0, "WP_queries": 0, "selector_fit_count": 24, "calibration_fit_count": 24}
    stats = dict(numerator=numerator, denominator=denominator, squared_row_mass=squared_mass,
                 loss_numerator=loss_numerator, fold_id=fold_id, market_class=classes, group=groups,
                 chain=np.asarray([r["chain"] for r in records]))
    return result, stats


def main():
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU analysis cap")))
    signal.alarm(298)
    ap = argparse.ArgumentParser()
    ap.add_argument("--allocation", type=Path, required=True)
    ap.add_argument("--allocation-sha256", required=True)
    args = ap.parse_args()
    allocation, bindings, output = load_allocation(args.allocation, args.allocation_sha256, "analysis")
    receipt_path = output / "extraction.json"
    receipt_sha = sha(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "COMPLETE" and receipt["allocation_sha256"] == args.allocation_sha256
    manifest_path = output / "selected-manifest.json"
    assert sha(manifest_path) == receipt["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["bindings_sha256"] == allocation["bindings_sha256"]
    for file, digest in receipt["files"].items():
        assert file in ("X.npy", "Y.npy", "M.npy") and sha(output / file) == digest
    assert set(receipt["files"]) == {"X.npy", "Y.npy", "M.npy"}
    with (output / "analysis-started.json").open("x") as stream:
        json.dump({"utc": datetime.now(timezone.utc).isoformat(), "allocation_sha256": args.allocation_sha256,
                   "extraction_sha256": receipt_sha, "source_sha256": sha(__file__)}, stream)
    X, Y, M = (np.load(output / name, mmap_mode="r", allow_pickle=False) for name in ("X.npy", "Y.npy", "M.npy"))
    with (output / "selector-progress.jsonl").open("x") as log:
        def progress(value):
            log.write(json.dumps({**value, "elapsed_seconds": time.monotonic() - start}) + "\n")
            log.flush()
        result, stats = analyze(X, Y, M, manifest["records"], progress)
    for file, digest in receipt["files"].items():
        assert sha(output / file) == digest
    assert sha(receipt_path) == receipt_sha and sha(manifest_path) == receipt["manifest_sha256"]
    load_allocation(args.allocation, args.allocation_sha256, "analysis")
    result["seconds"] = time.monotonic() - start
    result["maxRSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["provenance"] = {"allocation_sha256": args.allocation_sha256, "extraction_sha256": receipt_sha,
                             "manifest_sha256": receipt["manifest_sha256"], "sample_cache_sha256": receipt["files"]}
    assert result["seconds"] < 300
    np.savez_compressed(output / "chain-sufficient.npz", **stats)
    with (output / "report.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    signal.alarm(0)
    print(json.dumps({"report": str(output / "report.json"), "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
