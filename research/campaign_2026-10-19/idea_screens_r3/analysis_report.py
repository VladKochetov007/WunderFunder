"""Outer scoring and compact reporting for the fixed R3 screen."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from idea_screens_r2.ridge_math import RidgeModel, predict

from wnn.gmetric import GlobalAccumulator

from .analysis_views import CASE_BUDGET, CASE_NAMES, build_case_views
from .probe_helpers import ALPHAS, metric_without, safe_result


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def score_outer(
    sequences: list[dict],
    models: dict[str, dict[str, RidgeModel]],
    progress: Callable[[dict], None],
) -> tuple[dict[str, dict], dict[str, float]]:
    """Score every evaluation row and retain canonical component moments."""
    evaluation = [item for item in sequences if item["role"] == "eval"]
    components = sorted({int(item["component"]) for item in evaluation})
    totals = {name: {label: GlobalAccumulator() for label in models[name]} for name in CASE_NAMES}
    by_component = {
        name: {
            label: {component: GlobalAccumulator() for component in components}
            for label in models[name]
        }
        for name in CASE_NAMES
    }
    score_seconds = {name: 0.0 for name in CASE_NAMES}
    shared_transform_seconds = 0.0
    total_eval_rows = 0
    for index, sequence in enumerate(evaluation):
        transform_started = time.process_time()
        views = build_case_views(np.asarray(sequence["x"], dtype=np.float32))
        shared_transform_seconds += time.process_time() - transform_started
        component = int(sequence["component"])
        y = np.asarray(sequence["y"], dtype=np.float32)[None, :, :]
        mask = np.asarray(sequence["scored"], dtype=bool)[None, :]
        total_eval_rows += int(mask.sum())
        for name in CASE_NAMES:
            for label, model in models[name].items():
                started = time.process_time()
                prediction = predict(views[name], model)[None, :, :]
                totals[name][label].add(y, prediction, mask, group=component)
                by_component[name][label][component].add(y, prediction, mask)
                score_seconds[name] += time.process_time() - started
        if (index + 1) % 4 == 0 or index + 1 == len(evaluation):
            progress({"stage": "outer_components_scored", "completed": index + 1})
    if total_eval_rows == 0:
        raise ValueError("the fixed outer panel has no scored evaluation rows")

    results = {}
    for name in CASE_NAMES:
        results[name] = {}
        for label, total in totals[name].items():
            full_metric = safe_result(total)
            component_rows, drop_rows = [], []
            for component in components:
                part = by_component[name][label][component]
                component_rows.append(
                    {
                        "component": component,
                        "scored_rows": part.selected_rows,
                        "official_metric": safe_result(part),
                        "additive_raw_sums_W_WY_WP_WYY_WPP_WYP": total.groups[component].tolist(),
                    }
                )
                removed_rows = total.selected_rows - part.selected_rows
                removed_metric = (
                    metric_without(by_component[name][label], component) if removed_rows else None
                )
                drop_rows.append(
                    {
                        "omitted_component": component,
                        "scored_rows": removed_rows,
                        "official_metric": removed_metric,
                        "delta_wp_from_full_fixed_predictions": (
                            None
                            if removed_metric is None
                            else removed_metric["wp"] - full_metric["wp"]
                        ),
                    }
                )
            results[name][label] = {
                "official_metric": full_metric,
                "component_moments": component_rows,
                "drop_one_component": drop_rows,
            }
    return results, {
        "outer_score_cpu_s_by_case": score_seconds,
        "outer_shared_transform_cpu_s": shared_transform_seconds,
    }


def _metric_delta(left: dict, right: dict, model: str) -> dict[str, float]:
    a = left["outer"][model]["official_metric"]
    b = right["outer"][model]["official_metric"]
    return {key: float(a[key] - b[key]) for key in ("t0", "t1", "wp")}


def build_result(
    output: Path,
    sequences: list[dict],
    fit_component_order: list[int],
    fit_counts: dict,
    alpha_table: dict,
    selections: dict[str, int],
    cv_counts: dict,
    model_archives: dict,
    refit_seconds: dict[str, float],
    selected_models: dict[str, dict[str, RidgeModel]],
    outer: dict,
    outer_cost: dict,
    feature_collection_cpu: float,
    started_wall: float,
    started_cpu: float,
    sample_binding: dict,
    source_path: Path,
    progress_path: Path,
) -> dict:
    """Write final result and completion receipt after all fixed cases finish."""
    cases = {}
    for name in CASE_NAMES:
        components = set(fit_component_order[: CASE_BUDGET[name]])
        budget_rows = {
            component: rows
            for component, rows in fit_counts["sampled_rows_by_component"].items()
            if int(component) in components
        }
        selected = selected_models[name]["selected"]
        baseline = selected_models[name]["alpha1_baseline"]
        cases[name] = {
            "fit_component_budget": CASE_BUDGET[name],
            "fit_component_order": fit_component_order[: CASE_BUDGET[name]],
            "feature_dim": int(selected.coefficients.shape[0]),
            "fit_rows_by_component": budget_rows,
            "fit_sampled_rows": int(sum(budget_rows.values())),
            "inner_cv": alpha_table[name],
            "selected_alpha": int(selections[name]),
            "selected_coefficient_l2_norm_by_target": list(selected.norms),
            "alpha1_coefficient_l2_norm_by_target": list(baseline.norms),
            "final_refit_rows": int(sum(budget_rows.values())),
            "final_refit_cpu_s": refit_seconds[name],
            "case_model_cpu_s": (
                alpha_table[name]["cv_fit_cpu_s"]
                + alpha_table[name]["cv_score_cpu_s"]
                + refit_seconds[name]
                + outer_cost["outer_score_cpu_s_by_case"][name]
            ),
            "model_archive": model_archives[name],
            "outer": outer[name],
        }
    pairs = {
        "full64_books104_minus_i0": ("full64_books104", "full64_i0_52"),
        "full64_raw112_minus_i0_tail60": ("full64_raw112", "full64_i0_tail60"),
        "full64_i0_tail60_minus_i0": ("full64_i0_tail60", "full64_i0_52"),
        "full64_raw112_minus_books104": ("full64_raw112", "full64_books104"),
        "nested32_raw112_minus_full64_raw112": ("nested32_raw112", "full64_raw112"),
        "nested32_i0_tail60_minus_full64_i0_tail60": (
            "nested32_i0_tail60",
            "full64_i0_tail60",
        ),
        "full64_signed_ema_minus_raw112": ("full64_raw_signed_ema_hl32", "full64_raw112"),
        "full64_absolute_ema_minus_raw112": (
            "full64_raw_absolute_innovation_ema_hl32",
            "full64_raw112",
        ),
        "full64_both_ema_minus_raw112": ("full64_raw_both_ema_parts_hl32", "full64_raw112"),
    }
    source_sha = _digest(source_path)
    result = {
        "schema": "bounded_r3_component_factorial_nested_ridge_v1",
        "metric_id": "global_wp_clip2_v1",
        "sample_manifest_sha256": sample_binding["sha256"],
        "source_manifest_sha256": source_sha,
        "fit_selection": {
            "alpha_grid": list(ALPHAS),
            "shared_alpha_for_two_targets": True,
            "normalizer": "fit rows only within each fold; full applicable fit budget for refits",
            "target_weight": "abs(clip(y, -2, 2))",
            "training_stride": 40,
            "history_before_stride": True,
            "low32_components": fit_component_order[:32],
            "folds": cv_counts["budget_folds"],
            "selection_rule": "maximum pooled held-fit-component official mean WP; exact ties choose larger alpha",
            "pooled_cv_tables_include_per_fold_component_additive_raw_sums": True,
        },
        "counts": {
            **fit_counts,
            "eval_sequences": sum(item["role"] == "eval" for item in sequences),
            "eval_components": 32,
            "eval_scored_rows": sum(
                int(np.asarray(item["scored"], dtype=bool).sum())
                for item in sequences
                if item["role"] == "eval"
            ),
            **{key: value for key, value in cv_counts.items() if key != "budget_folds"},
            "final_refit_calls": len(CASE_NAMES),
            "final_refit_alpha_models": sum(len({1, int(selections[name])}) for name in CASE_NAMES),
            "final_target_solves": 2 * sum(len({1, int(selections[name])}) for name in CASE_NAMES),
            "outer_prediction_calls": 32 * len(CASE_NAMES) * 2,
            "failures": [],
        },
        "cost": {
            "feature_collection_cpu_s": feature_collection_cpu,
            "outer_cpu": outer_cost,
            "total_process_cpu_s": time.process_time() - started_cpu,
            "total_wall_s": time.monotonic() - started_wall,
        },
        "cases": cases,
        "primary_descriptive_contrasts": {
            key: {
                "selected_alpha_metric_delta_t0_t1_wp": _metric_delta(
                    cases[left], cases[right], "selected"
                ),
                "alpha1_baseline_metric_delta_t0_t1_wp": _metric_delta(
                    cases[left], cases[right], "alpha1_baseline"
                ),
            }
            for key, (left, right) in pairs.items()
        },
        "interpretation_limits": [
            "R3 is an exposed F0 discovery panel, not a fresh pristine validation or full recipe comparison.",
            "Component-first inclusion and one group per component differ from R1/R2; do not subtract their scores.",
            "Nested32 versus full64 also changes selected components and composition, so it is not an isolated sample-size causal effect.",
            "The supplied-valid evaluation panel is descriptive and its realized official metric is not an unbiased full-fold WP estimate.",
        ],
        "case_failures": [],
        "model_archives": model_archives,
        "source_binding": sample_binding,
        "progress_file": progress_path.name,
    }
    result_path = output / "RESULT.json"
    _write_json(result_path, result)
    _write_json(
        output / "DONE.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "result_sha256": _digest(result_path),
            "result_bytes": result_path.stat().st_size,
            "model_archives": model_archives,
            "process_cpu_s": time.process_time() - started_cpu,
            "wall_s": time.monotonic() - started_wall,
            "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )
    return result
