"""Fit-only full-history collection and pooled component-fold ridge selection."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from idea_screens_r2.ridge_math import fit_candidates, predict

from wnn.gmetric import GlobalAccumulator

from .analysis_views import CASE_BUDGET, CASE_NAMES, build_case_views
from .probe_helpers import (
    ALPHAS,
    FIT_ROW_STRIDE,
    component_folds,
    raw_sums_by_group,
    safe_result,
    select_alpha,
)


def fit_component_sequences(
    sequences: list[dict], fit_component_order: list[int]
) -> dict[int, dict]:
    """Resolve fit sequence metadata without touching any evaluation arrays or labels."""
    fit_items = [item for item in sequences if item["role"] == "fit"]
    fit = {int(item["component"]): item for item in fit_items}
    if (
        len(fit_items) != 64
        or len(fit) != 64
        or len(fit_component_order) != 64
        or len(set(fit_component_order)) != 64
        or set(fit) != set(fit_component_order)
    ):
        raise ValueError("the sample must bind one fit sequence to each ordered 64 components")
    return fit


def collect_fit_rows(
    sequences: list[dict], fit_component_order: list[int]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
    """Build reset-local histories over whole fit sequences, then apply stride 40."""
    fit = fit_component_sequences(sequences, fit_component_order)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in CASE_NAMES}
    targets, row_components = [], []
    scored_by_component: dict[str, int] = {}
    sampled_by_component: dict[str, int] = {}
    for component in fit_component_order:
        sequence = fit[component]
        x = np.asarray(sequence["x"], dtype=np.float32)
        y = np.asarray(sequence["y"], dtype=np.float32)
        scored = np.asarray(sequence["scored"], dtype=bool)
        if x.shape != (20_000, 112) or y.shape != (20_000, 2) or scored.shape != (20_000,):
            raise ValueError("R3 requires complete source-bound 20,000-row sequences")
        full_views = build_case_views(x)
        rows = np.arange(0, len(x), FIT_ROW_STRIDE)
        rows = rows[scored[rows]]
        if not np.isfinite(y[rows]).all():
            raise ValueError("selected fit labels must be finite")
        for name in CASE_NAMES:
            parts[name].append(full_views[name][rows])
        targets.append(y[rows])
        row_components.append(np.full(rows.size, component, dtype=np.int32))
        scored_by_component[str(component)] = int(scored.sum())
        sampled_by_component[str(component)] = int(rows.size)
    joined_targets = np.concatenate(targets, axis=0)
    if not len(joined_targets):
        raise ValueError("the fit panel has no scored stride-40 training rows")
    return (
        {name: np.concatenate(blocks, axis=0) for name, blocks in parts.items()},
        joined_targets,
        np.concatenate(row_components),
        {
            "fit_components": len(fit),
            "fit_sequences": len(fit),
            "fit_sampled_rows": int(len(joined_targets)),
            "fit_scored_rows": int(sum(scored_by_component.values())),
            "scored_rows_by_component": scored_by_component,
            "sampled_rows_by_component": sampled_by_component,
            "fit_stride_after_full_history": FIT_ROW_STRIDE,
        },
    )


def cross_validate(
    sequences: list[dict],
    features: dict[str, np.ndarray],
    targets: np.ndarray,
    row_components: np.ndarray,
    fit_component_order: list[int],
    progress: Callable[[dict], None],
) -> tuple[dict[str, dict], dict[str, int], dict]:
    """Select one shared alpha per case by pooled WP over held fit components."""
    budgets = {
        64: list(fit_component_order),
        32: list(fit_component_order[:32]),
    }
    folds = {budget: component_folds(order) for budget, order in budgets.items()}
    models_by_fold: list[dict[str, dict[int, object]]] = [dict() for _ in range(3)]
    fit_seconds = {name: 0.0 for name in CASE_NAMES}
    score_seconds = {name: 0.0 for name in CASE_NAMES}
    predictions = {name: 0 for name in CASE_NAMES}
    for fold_index in range(3):
        for name in CASE_NAMES:
            order = budgets[CASE_BUDGET[name]]
            held = folds[CASE_BUDGET[name]][fold_index]
            training = set(order) - held
            train_rows = np.isin(row_components, list(training))
            if not train_rows.any():
                raise ValueError(f"empty inner training rows for {name}, fold {fold_index}")
            started = time.process_time()
            models_by_fold[fold_index][name] = fit_candidates(
                features[name][train_rows], targets[train_rows], ALPHAS
            )
            fit_seconds[name] += time.process_time() - started
            progress({"stage": "fold_model_fit", "fold": fold_index, "case": name})

    pooled = {name: {alpha: GlobalAccumulator() for alpha in ALPHAS} for name in CASE_NAMES}
    by_fold = {
        name: {alpha: [GlobalAccumulator() for _ in range(3)] for alpha in ALPHAS}
        for name in CASE_NAMES
    }
    fit_sequences = [item for item in sequences if item["role"] == "fit"]
    for fold_index in range(3):
        for sequence in fit_sequences:
            component = int(sequence["component"])
            held_cases = [
                name for name in CASE_NAMES if component in folds[CASE_BUDGET[name]][fold_index]
            ]
            if not held_cases:
                continue
            transform_started = time.process_time()
            full_views = build_case_views(np.asarray(sequence["x"], dtype=np.float32))
            score_seconds["_shared_transform"] = score_seconds.get("_shared_transform", 0.0) + (
                time.process_time() - transform_started
            )
            y = np.asarray(sequence["y"], dtype=np.float32)[None, :, :]
            mask = np.asarray(sequence["scored"], dtype=bool)[None, :]
            for name in held_cases:
                for alpha, model in models_by_fold[fold_index][name].items():
                    model_started = time.process_time()
                    prediction = predict(full_views[name], model)[None, :, :]
                    pooled[name][alpha].add(y, prediction, mask, group=component)
                    by_fold[name][alpha][fold_index].add(y, prediction, mask, group=component)
                    score_seconds[name] += time.process_time() - model_started
                    predictions[name] += 1
            progress(
                {
                    "stage": "held_fit_sequence_scored",
                    "fold": fold_index,
                    "component": component,
                    "cases": held_cases,
                }
            )
        progress({"stage": "held_fit_fold_scored", "fold": fold_index})

    tables, selected = {}, {}
    for name in CASE_NAMES:
        alpha_table = {}
        for alpha in ALPHAS:
            accumulator = pooled[name][alpha]
            if accumulator.selected_rows == 0:
                raise ValueError(f"no held fit rows for {name}, alpha {alpha}")
            fold_rows = []
            for fold_index in range(3):
                held = sorted(folds[CASE_BUDGET[name]][fold_index])
                part = by_fold[name][alpha][fold_index]
                fold_rows.append(
                    {
                        "fold": fold_index,
                        "held_components": held,
                        "training_components": sorted(set(budgets[CASE_BUDGET[name]]) - set(held)),
                        "scored_rows": part.selected_rows,
                        "official_metric": safe_result(part),
                        "component_additive_raw_sums_W_WY_WP_WYY_WPP_WYP": raw_sums_by_group(part),
                    }
                )
            alpha_table[str(alpha)] = {
                "pooled_held_fit_components": safe_result(accumulator),
                "folds": fold_rows,
            }
        selected[name] = select_alpha(pooled[name])
        tables[name] = {
            "fit_component_budget": CASE_BUDGET[name],
            "folds": [sorted(part) for part in folds[CASE_BUDGET[name]]],
            "alphas": alpha_table,
            "selected_alpha": selected[name],
            "selection_rule": "maximum pooled official mean WP; exact ties choose the larger alpha",
            "prediction_count": predictions[name],
            "cv_fit_cpu_s": fit_seconds[name],
            "cv_score_cpu_s": score_seconds[name],
        }
    return (
        tables,
        selected,
        {
            "fold_seed": 19_025,
            "fold_count": 3,
            "budget_folds": {
                str(budget): [sorted(part) for part in value] for budget, value in folds.items()
            },
            "candidate_fit_sets": 3 * len(CASE_NAMES),
            "inner_alpha_models": 3 * len(CASE_NAMES) * len(ALPHAS),
            "two_target_solves": 2 * 3 * len(CASE_NAMES) * len(ALPHAS),
            "shared_transform_cpu_s": score_seconds.get("_shared_transform", 0.0),
        },
    )
