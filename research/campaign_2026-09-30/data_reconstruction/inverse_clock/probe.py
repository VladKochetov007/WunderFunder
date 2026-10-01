from __future__ import annotations

import hashlib
import json
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import LinearConstraint, minimize
from scipy.special import ndtr
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
KNOTS = np.array(
    [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 1.0]
)
NAMES = (
    [f"i0_p{i}" for i in range(22)]
    + [f"i0_{kind}{i}" for kind in ("dp", "dv") for i in range(4)]
    + ["a5", "a6"]
)
LOWER = float(np.float32(norm.ppf(1e-7 - np.finfo(float).eps)))
UPPER = -LOWER


def write(name: str, data: object) -> None:
    """Persist a compact deterministic research record."""
    (HERE / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read(source: pq.ParquetFile, group: int) -> tuple[np.ndarray, dict[str, int]]:
    """Return conservative observed-quiet current/past CDF coordinates."""
    tab = source.read_row_group(group, columns=NAMES, use_threads=False)
    x = np.column_stack([tab.column(k).to_numpy() for k in NAMES]).astype(np.float64)
    quiet = np.all(x[1:, :22] == x[:-1, :22], 1) & np.all(x[1:, 22:30] == x[:-1, 22:30], 1)
    old, cur = x[:-1, 30:], x[1:, 30:]
    reset = np.any(cur == LOWER, 1)
    clipped = np.any((cur <= LOWER) | (cur >= UPPER) | (old <= LOWER) | (old >= UPPER), 1)
    negative = np.any(cur < old, 1)
    keep = quiet & ~reset & ~clipped & ~negative
    u = np.column_stack((ndtr(old[:, 0]), ndtr(cur[:, 0]), ndtr(old[:, 1]), ndtr(cur[:, 1])))
    counts = {
        "transitions": len(quiet),
        "observed_joint_quiet": int(quiet.sum()),
        "quiet_endpoint_reset_excluded": int((quiet & reset).sum()),
        "quiet_clipped_excluded": int((quiet & clipped).sum()),
        "quiet_negative_excluded": int((quiet & negative).sum()),
        "eligible": int(keep.sum()),
        "eligible_both_CDF_zero": int((keep & np.all(cur == old, 1)).sum()),
    }
    return u[keep], counts


def basis(u: np.ndarray) -> np.ndarray:
    """Use fixed linear interpolation heights with the lower node fixed at zero."""
    interval = np.clip(np.searchsorted(KNOTS, u, side="right") - 1, 0, 14)
    fraction = (u - KNOTS[interval]) / np.diff(KNOTS)[interval]
    result = np.zeros((len(u), 16))
    rows = np.arange(len(u))
    result[rows, interval] = 1 - fraction
    result[rows, interval + 1] = fraction
    return result[:, 1:]


def knot_ids(u: np.ndarray) -> np.ndarray:
    """Assign current CDF coordinates to fixed knot intervals."""
    return np.clip(np.searchsorted(KNOTS, u, side="right") - 1, 0, 14)


def cells(u: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Assign fixed TRAIN quartile bins in the joint current-age space."""
    a = np.clip(np.searchsorted(bins[0], u[:, 1], side="right") - 1, 0, 3)
    b = np.clip(np.searchsorted(bins[1], u[:, 3], side="right") - 1, 0, 3)
    return a * 4 + b


def quantiles(v: np.ndarray) -> list[float] | None:
    """Report nonempty finite increment quantiles."""
    return np.quantile(v, [0.1, 0.5, 0.9, 0.99, 1]).tolist() if len(v) else None


def metrics(d: np.ndarray) -> dict[str, object]:
    """Use symmetric discrepancy without an epsilon denominator."""
    if not len(d):
        return {"rows": 0}
    numerator = float(np.square(d[:, 0] - d[:, 1]).mean())
    denominator = float(np.square(d).mean())
    return {
        "rows": len(d),
        "symmetric_relative_RMS": float(np.sqrt(numerator / denominator))
        if denominator > 0
        else None,
        "absolute_difference_RMS": float(np.sqrt(numerator)),
        "paired_positive_fraction": float(np.all(d > 0, 1).mean()),
        "clock_mean": d.mean(0).tolist(),
        "clock_RMS": np.sqrt(np.square(d).mean(0)).tolist(),
        "clock_all_quantiles_q10_q50_q90_q99_max": [quantiles(d[:, i]) for i in range(2)],
        "clock_positive_quantiles_q10_q50_q90_q99_max": [
            quantiles(d[d[:, i] > 0, i]) for i in range(2)
        ],
    }


def mass(weights: np.ndarray, ids: np.ndarray, n: int) -> dict[str, object]:
    """Expose concentration and effective occupied bins."""
    counts = np.bincount(ids, minlength=n)
    raw = np.bincount(ids, weights=weights, minlength=n)
    total = raw.sum()
    fraction = raw / total if total > 0 else np.zeros(n)
    effective = float(1 / np.square(fraction).sum()) if total > 0 else None
    return {
        "rows": counts.tolist(),
        "squared_mass_fraction": fraction.tolist(),
        "maximum_mass_fraction": float(fraction.max()),
        "effective_bins": effective,
    }


def controls(u: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Fit positive alignment scalars on TRAIN and supported observed mean scales."""
    output, parameters = {}, {}
    for name in ("CDF", "log_survival"):
        values = u if name == "CDF" else -np.log1p(-np.minimum(u, 1 - 1e-7))
        d5, d6 = values[:, 1] - values[:, 0], values[:, 3] - values[:, 2]
        denominator = float(np.dot(d6, d6))
        scalar = float(np.dot(d5, d6) / denominator) if denominator > 0 else 0
        pair = np.column_stack((d5, scalar * d6))
        scale = float(pair.mean())
        if scalar <= 0 or scale <= 0:
            raise ValueError(f"Nonpositive {name} control scale")
        output[name] = pair / scale
        parameters[name] = {"clock6_positive_scalar": scalar, "observed_mean_scale": scale}
    return output, parameters


def apply_controls(u: np.ndarray, parameters: dict[str, dict[str, float]]) -> dict[str, np.ndarray]:
    """Hold scalar control parameters fixed on test rows."""
    output = {}
    for name, values in parameters.items():
        z = u if name == "CDF" else -np.log1p(-np.minimum(u, 1 - 1e-7))
        output[name] = (
            np.column_stack(
                (z[:, 1] - z[:, 0], values["clock6_positive_scalar"] * (z[:, 3] - z[:, 2]))
            )
            / values["observed_mean_scale"]
        )
    return output


def analyze(
    d: np.ndarray,
    u: np.ndarray,
    components: np.ndarray,
    cell: np.ndarray,
    support: dict[str, np.ndarray],
    reference: dict[str, object],
) -> dict[str, object]:
    """Apply frozen support and noncollapse checks with full concentration evidence."""
    m = metrics(d)
    weight = np.square(d).mean(1)
    unique, comp_ids = np.unique(components, return_inverse=True)
    component_mass = mass(weight, comp_ids, len(unique))
    cell_mass = mass(weight, cell, 16)
    kid = [knot_ids(u[:, j]) for j in (1, 3)]
    kmass = [mass(weight, v, 15) for v in kid]
    unknown = (
        (support["cells"][cell] < 32)
        | (support["knot5"][kid[0]] < 32)
        | (support["knot6"][kid[1]] < 32)
    )
    unknown_mass = float(weight[unknown].sum() / weight.sum()) if weight.sum() > 0 else None
    medians = [
        v[1] if v is not None else 0 for v in m["clock_positive_quantiles_q10_q50_q90_q99_max"]
    ]
    rms = np.array(m["clock_RMS"])
    ref_rms = np.array(reference["clock_RMS"])
    ref_median = np.array([v[1] for v in reference["clock_positive_quantiles_q10_q50_q90_q99_max"]])
    gates = {
        "positive_fraction": m["paired_positive_fraction"] >= 0.90,
        "median_vs_frozen_CDF": bool(np.all(np.array(medians) >= 0.1 * ref_median)),
        "RMS_vs_frozen_CDF": bool(np.all((rms >= 0.25 * ref_rms) & (rms <= 4 * ref_rms))),
        "component_mass": component_mass["maximum_mass_fraction"] <= 0.25,
        "knot_mass": all(v["maximum_mass_fraction"] <= 0.50 for v in kmass),
        "cell_mass": cell_mass["maximum_mass_fraction"] <= 0.50,
        "effective_cells": cell_mass["effective_bins"] is not None
        and cell_mass["effective_bins"] >= 4,
        "unsupported_mass": unknown_mass is not None and unknown_mass <= 0.01,
        "unsupported_rows": float(unknown.mean()) <= 0.05,
    }
    return {
        "metrics": m,
        "noncollapse_gates": gates,
        "noncollapse_pass": all(gates.values()),
        "component_mass": component_mass | {"components": unique.tolist()},
        "cell_mass": cell_mass,
        "current_knot_mass": kmass,
        "unsupported_row_fraction": float(unknown.mean()),
        "unsupported_squared_mass_fraction": unknown_mass,
    }


def main() -> None:
    """Run exactly one bounded joint-clock fit before opening the held-out features."""
    start = time.monotonic()
    panel = json.loads((HERE.parent / "selection.json").read_text())
    prior_classes = {
        v["group"]: v["diagnostic_target_class"]
        for v in json.loads((HERE.parent / "results.json").read_text())["by_group"]
    }
    if set(panel["fit_components"]) & set(panel["test_components"]):
        raise ValueError("Fit/test component overlap")
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    fit, fit_components, fit_counts = [], [], []
    for group, component in zip(panel["fit_train_groups"], panel["fit_components"], strict=True):
        u, count = read(source, group)
        loc = np.linspace(0, len(u) - 1, min(512, len(u)), dtype=int)
        fit.append(u[loc])
        fit_components.extend([component] * len(loc))
        fit_counts.append(
            {"group": group, "component": component, "counts": count, "selected": len(loc)}
        )
    ufit, cf = np.concatenate(fit), np.array(fit_components)
    bins = np.array([np.r_[0, np.quantile(ufit[:, j], [0.25, 0.5, 0.75]), 1] for j in (1, 3)])
    cell_fit = cells(ufit, bins)
    support = {
        "knot5": np.bincount(knot_ids(ufit[:, 1]), minlength=15),
        "knot6": np.bincount(knot_ids(ufit[:, 3]), minlength=15),
        "cells": np.bincount(cell_fit, minlength=16),
    }
    raw_d = np.column_stack((ufit[:, 1] - ufit[:, 0], ufit[:, 3] - ufit[:, 2]))
    s0 = float(1 / raw_d.mean())
    fit_controls, control_parameters = controls(ufit)
    reference = metrics(fit_controls["CDF"])
    write(
        "fit_support.json",
        {
            "selection_sha256": hashlib.sha256(
                (HERE.parent / "selection.json").read_bytes()
            ).hexdigest(),
            "knots": KNOTS.tolist(),
            "offset_bins": bins.tolist(),
            "support": {k: v.tolist() for k, v in support.items()},
            "fit_group_counts": fit_counts,
            "s0_observed_increment_scale": s0,
            "control_parameters": control_parameters,
            "frozen_CDF_noncollapse_reference": reference,
            "test_features_not_yet_read": True,
        },
    )
    d5 = basis(ufit[:, 1]) - basis(ufit[:, 0])
    d6 = basis(ufit[:, 3]) - basis(ufit[:, 2])
    design = s0 * np.column_stack((d5, -d6))
    widths = np.diff(KNOTS)
    slope = np.diag(1 / widths)
    slope[1:, :-1] -= np.diag(1 / widths[1:])
    curvature = (slope[1:] - slope[:-1]) * np.sqrt((widths[1:] + widths[:-1]) / 2)[:, None]
    r = np.zeros((28, 30))
    r[:14, :15], r[14:, 15:] = curvature, curvature
    gram = design.T @ design / len(design)
    hessian = gram + 1e-4 * (r.T @ r)
    scale_vector = s0 * np.r_[d5.mean(0), d6.mean(0)] / 2
    monotone = np.eye(15)
    monotone[1:, :-1] -= np.eye(14)
    constraint = np.zeros((33, 30))
    constraint[0] = scale_vector
    constraint[1:16, :15], constraint[16:31, 15:] = monotone, monotone
    constraint[31, 14], constraint[32, 29] = 1, 1
    lower = np.r_[1.0, 0.001 * widths, 0.001 * widths, -np.inf, -np.inf]
    upper = np.r_[1.0, np.full(30, np.inf), 10.0, 10.0]
    initial = np.tile(KNOTS[1:], 2)
    np.savez(
        HERE / "qp_artifacts.npz",
        gram=gram,
        hessian=hessian,
        curvature=r,
        constraint=constraint,
        lower=lower,
        upper=upper,
        initial=initial,
        scale_vector=scale_vector,
    )
    fit_result = minimize(
        lambda z: float(z @ hessian @ z),
        initial,
        jac=lambda z: 2 * hessian @ z,
        method="SLSQP",
        constraints=[LinearConstraint(constraint, lower, upper)],
        options={"maxiter": 200, "ftol": 1e-10, "disp": False},
    )
    z = fit_result.x
    constrained = constraint @ z
    violation = float(
        max(np.maximum(lower - constrained, 0).max(), np.maximum(constrained - upper, 0).max())
    )
    h5, h6 = np.r_[0, z[:15]] * s0, np.r_[0, z[15:]] * s0
    candidate_fit = np.column_stack((d5 @ h5[1:], d6 @ h6[1:]))
    fit_analysis = analyze(candidate_fit, ufit, cf, cell_fit, support, reference)
    solver = {
        "success": bool(fit_result.success),
        "status": int(fit_result.status),
        "message": str(fit_result.message),
        "iterations": int(fit_result.nit),
        "max_constraint_violation": violation,
        "objective": float(fit_result.fun),
        "discrepancy_objective": float(z @ gram @ z),
        "curvature_objective": float(1e-4 * np.square(r @ z).sum()),
        "objective_gradient": (2 * hessian @ z).tolist(),
        "clock5_heights": h5.tolist(),
        "clock6_heights": h6.tolist(),
        "map_total_ranges": [float(h5[-1]), float(h6[-1])],
        "range_bound": 10 * s0,
        "fit_analysis": fit_analysis,
        "fit_controls": {k: metrics(v) for k, v in fit_controls.items()},
        "test_features_not_yet_read": True,
    }
    write("fit_result.json", solver)
    print(
        json.dumps(
            {
                "stage": "fit_frozen_before_test",
                "solver_success": solver["success"],
                "fit_noncollapse_pass": fit_analysis["noncollapse_pass"],
            }
        ),
        flush=True,
    )
    test, components, class_ids, test_counts = [], [], [], []
    for group, component in zip(panel["test_train_groups"], panel["test_components"], strict=True):
        u, count = read(source, group)
        test.append(u)
        components.extend([component] * len(u))
        class_ids.extend([prior_classes[group]] * len(u))
        test_counts.append(
            {
                "group": group,
                "component": component,
                "diagnostic_class": prior_classes[group],
                "counts": count,
            }
        )
    ut, ct, cl = np.concatenate(test), np.array(components), np.array(class_ids)
    cell_t = cells(ut, bins)
    candidate = np.column_stack(
        (
            np.interp(ut[:, 1], KNOTS, h5) - np.interp(ut[:, 0], KNOTS, h5),
            np.interp(ut[:, 3], KNOTS, h6) - np.interp(ut[:, 2], KNOTS, h6),
        )
    )
    methods = {
        "candidate": candidate,
        **apply_controls(ut, control_parameters),
        "publication_row_age_identity": np.ones_like(candidate),
    }
    method_reports = {}
    for name, d in methods.items():
        method_reports[name] = {
            "all": analyze(d, ut, ct, cell_t, support, reference),
            "by_component": {
                str(component): metrics(d[ct == component]) for component in np.unique(ct)
            },
            "by_diagnostic_class": {str(c): metrics(d[cl == c]) for c in range(4)},
            "by_fixed_offset_cell": {
                str(cell): {
                    "fit_support_rows": int(support["cells"][cell]),
                    "supported": bool(support["cells"][cell] >= 32),
                    **metrics(d[cell_t == cell]),
                }
                for cell in range(16)
            },
        }
    candidate_error = method_reports["candidate"]["all"]["metrics"]["symmetric_relative_RMS"]
    best_control = min(
        method_reports[k]["all"]["metrics"]["symmetric_relative_RMS"]
        for k in ("CDF", "log_survival")
    )
    component_wins = sum(
        all(
            method_reports["candidate"]["by_component"][str(c)]["symmetric_relative_RMS"]
            < method_reports[k]["by_component"][str(c)]["symmetric_relative_RMS"]
            for k in ("CDF", "log_survival")
        )
        for c in np.unique(ct)
    )
    gates = {
        "solver_converged_feasible": bool(fit_result.success and violation <= 1e-6),
        "fit_noncollapse": fit_analysis["noncollapse_pass"],
        "holdout_noncollapse": method_reports["candidate"]["all"]["noncollapse_pass"],
        "aggregate_improvement10percent": candidate_error < 0.90 * best_control,
        "component_improvement75percent": component_wins >= 0.75 * len(np.unique(ct)),
        "fit_test_components_disjoint": True,
        "independent_fresh_component_replay": False,
    }
    result = {
        "preregistration_sha256": hashlib.sha256(
            (HERE / "preregistration.json").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "solver": solver,
        "test_group_counts": test_counts,
        "methods": method_reports,
        "gates": gates,
        "components_beating_both_controls": component_wins,
        "test_components": len(np.unique(ct)),
        "claim": "exploratory_shared_coordinate_only; physical_time/generator/units_unknown; independent_replay_required",
        "resources": {
            "wall_seconds": time.monotonic() - start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cpu_seconds": time.process_time(),
        },
        "exit_code": 0,
    }
    write("results.json", result)
    print(
        json.dumps(
            {
                "stage": "finished",
                "gates": gates,
                "candidate_error": candidate_error,
                "best_control_error": best_control,
                "resources": result["resources"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write(
            "failure.json",
            {
                "type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "exit_code": 1,
            },
        )
        raise
