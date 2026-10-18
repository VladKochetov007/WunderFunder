"""Fixed-projection fit-only collection, component CV, and final ridge refits."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from idea_screens_r2.ridge_math import RidgeModel, fit_candidates, predict
from idea_screens_r3.probe_helpers import (
    ALPHAS,
    FIT_ROW_STRIDE,
    component_folds,
    raw_sums_by_group,
    safe_result,
    select_alpha,
)

from wnn.gmetric import GlobalAccumulator

from .random_features import (
    BASE_SEEDS,
    BaseNormalizer,
    ProjectionDraw,
    augment_pair,
    build_bases,
    draw_projection,
    fit_base_normalizer,
    projection_diagnostics,
)

CASE_NAMES = (
    "raw112_identity64",
    "raw112_tanh64",
    "raw_signed224_identity64",
    "raw_signed224_tanh64",
)
CASE_BASE = {name: name.removesuffix("_identity64").removesuffix("_tanh64") for name in CASE_NAMES}
CASE_ACTIVATION = {name: "tanh" if name.endswith("_tanh64") else "identity" for name in CASE_NAMES}


def collect_fit_rows(
    sequences: list[dict], fit_component_order: list[int]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
    """Build whole-sequence causal bases before selecting scored stride-40 fit rows."""
    fit_items = [item for item in sequences if item["role"] == "fit"]
    by_component = {int(item["component"]): item for item in fit_items}
    if (
        len(fit_items) != 64
        or len(by_component) != 64
        or set(by_component) != set(fit_component_order)
    ):
        raise ValueError("R4 requires one fit sequence for each ordered 64 components")
    parts: dict[str, list[np.ndarray]] = {name: [] for name in BASE_SEEDS}
    targets, component_rows = [], []
    sampled_by_component, scored_by_component = {}, {}
    for component in fit_component_order:
        sequence = by_component[component]
        x = np.asarray(sequence["x"], dtype=np.float32)
        y = np.asarray(sequence["y"], dtype=np.float32)
        scored = np.asarray(sequence["scored"], dtype=bool)
        if x.shape != (20_000, 112) or y.shape != (20_000, 2) or scored.shape != (20_000,):
            raise ValueError("R4 requires complete source-bound 20,000-row sequences")
        bases = build_bases(x)
        rows = np.arange(0, len(x), FIT_ROW_STRIDE)
        rows = rows[scored[rows]]
        if not np.isfinite(y[rows]).all():
            raise ValueError("selected fit labels must be finite")
        for name in BASE_SEEDS:
            parts[name].append(bases[name][rows])
        targets.append(y[rows])
        component_rows.append(np.full(rows.size, component, dtype=np.int32))
        sampled_by_component[str(component)] = int(rows.size)
        scored_by_component[str(component)] = int(scored.sum())
    all_targets = np.concatenate(targets, axis=0)
    if not len(all_targets):
        raise ValueError("R4 fit panel has no scored stride-40 rows")
    total_scored = sum(scored_by_component.values())
    if len(all_targets) != 31_808 or total_scored != 1_273_664:
        raise ValueError(
            f"R4 fit masks differ from frozen panel counts: {len(all_targets)} stride rows, {total_scored} scored rows"
        )
    return (
        {name: np.concatenate(values, axis=0) for name, values in parts.items()},
        all_targets,
        np.concatenate(component_rows),
        {
            "fit_sequences": len(fit_items),
            "fit_components": len(by_component),
            "fit_scored_rows": int(total_scored),
            "fit_sampled_rows": int(len(all_targets)),
            "scored_rows_by_component": scored_by_component,
            "sampled_rows_by_component": sampled_by_component,
            "training_stride": FIT_ROW_STRIDE,
            "causal_history_before_stride": True,
        },
    )


def cross_validate(
    sequences: list[dict],
    base_rows: dict[str, np.ndarray],
    targets: np.ndarray,
    row_components: np.ndarray,
    fit_component_order: list[int],
    progress: Callable[[dict], None],
) -> tuple[dict[str, dict], dict[str, int], dict[str, ProjectionDraw], dict]:
    """Select each fixed case's shared two-target alpha on pooled held-fit WP."""
    folds = component_folds(fit_component_order)
    draws = {
        name: draw_projection(base_rows[name].shape[1], seed) for name, seed in BASE_SEEDS.items()
    }
    models: list[dict[str, dict[int, dict[int, RidgeModel]]]] = [
        {name: {} for name in CASE_NAMES} for _ in folds
    ]
    fold_norms: list[dict[str, BaseNormalizer]] = [{} for _ in folds]
    fit_seconds = {name: 0.0 for name in CASE_NAMES}
    model_stats: dict[str, dict[str, dict[str, dict]]] = {name: {} for name in CASE_NAMES}
    for fold_index, held_components in enumerate(folds):
        train_components = set(fit_component_order) - held_components
        train_rows_mask = np.isin(row_components, list(train_components))
        if not train_rows_mask.any():
            raise ValueError(f"empty inner training rows in fold {fold_index}")
        for base_name, draw in draws.items():
            normalizer = fit_base_normalizer(base_rows[base_name][train_rows_mask])
            fold_norms[fold_index][base_name] = normalizer
            pair = augment_pair(base_rows[base_name][train_rows_mask], normalizer, draw)
            diagnostic = projection_diagnostics(
                base_rows[base_name][train_rows_mask], normalizer, draw
            )
            for activation_index, activation in enumerate(("identity", "tanh")):
                case = f"{base_name}_{activation}64"
                started = time.process_time()
                candidates = fit_candidates(
                    pair[activation_index], targets[train_rows_mask], ALPHAS
                )
                fit_seconds[case] += time.process_time() - started
                models[fold_index][case] = candidates
                model_stats[case][str(fold_index)] = {
                    str(alpha): {
                        "training_rows": int(train_rows_mask.sum()),
                        "training_components": len(train_components),
                        "held_components": sorted(held_components),
                        "coefficient_l2_norm_by_target": list(candidates[alpha].norms),
                        "ridge_feature_dimension": int(candidates[alpha].coefficients.shape[0]),
                        "base_normalizer_scale_min": float(normalizer.scale.min()),
                        "base_normalizer_scale_max": float(normalizer.scale.max()),
                        "projection_diagnostics": diagnostic,
                    }
                    for alpha in ALPHAS
                }
                progress({"stage": "inner_models_fit", "fold": fold_index, "case": case})

    pooled = {case: {alpha: GlobalAccumulator() for alpha in ALPHAS} for case in CASE_NAMES}
    by_fold = {
        case: {alpha: [GlobalAccumulator() for _ in folds] for alpha in ALPHAS}
        for case in CASE_NAMES
    }
    held_sequences = [item for item in sequences if item["role"] == "fit"]
    score_seconds = {name: 0.0 for name in CASE_NAMES}
    for fold_index, held_components in enumerate(folds):
        for sequence in held_sequences:
            component = int(sequence["component"])
            if component not in held_components:
                continue
            bases = build_bases(np.asarray(sequence["x"], dtype=np.float32))
            y = np.asarray(sequence["y"], dtype=np.float32)[None, :, :]
            mask = np.asarray(sequence["scored"], dtype=bool)[None, :]
            for base_name, draw in draws.items():
                pair = augment_pair(bases[base_name], fold_norms[fold_index][base_name], draw)
                for activation_index, activation in enumerate(("identity", "tanh")):
                    case = f"{base_name}_{activation}64"
                    for alpha, model in models[fold_index][case].items():
                        started = time.process_time()
                        prediction = predict(pair[activation_index], model)[None, :, :]
                        pooled[case][alpha].add(y, prediction, mask, group=component)
                        by_fold[case][alpha][fold_index].add(y, prediction, mask, group=component)
                        score_seconds[case] += time.process_time() - started
            progress(
                {"stage": "held_fit_sequence_scored", "fold": fold_index, "component": component}
            )

    tables, selected = {}, {}
    for case in CASE_NAMES:
        alpha_rows = {}
        for alpha in ALPHAS:
            fold_rows = []
            for fold_index, held_components in enumerate(folds):
                accumulator = by_fold[case][alpha][fold_index]
                fold_rows.append(
                    {
                        "fold": fold_index,
                        "held_components": sorted(held_components),
                        "training_components": sorted(set(fit_component_order) - held_components),
                        "scored_rows": accumulator.selected_rows,
                        "official_metric": safe_result(accumulator),
                        "component_additive_raw_sums_W_WY_WP_WYY_WPP_WYP": raw_sums_by_group(
                            accumulator
                        ),
                    }
                )
            alpha_rows[str(alpha)] = {
                "pooled_held_fit_components": safe_result(pooled[case][alpha]),
                "folds": fold_rows,
            }
        selected[case] = select_alpha(pooled[case])
        tables[case] = {
            "fit_component_budget": 64,
            "fold_models": model_stats[case],
            "folds": [sorted(fold) for fold in folds],
            "alphas": alpha_rows,
            "selected_alpha": selected[case],
            "selection_rule": "maximum pooled official mean WP; exact ties choose larger alpha",
            "cv_fit_cpu_s": fit_seconds[case],
            "cv_score_cpu_s": score_seconds[case],
        }
    counts = {
        "fold_seed": 19_025,
        "fold_count": 3,
        "budget_folds": [sorted(fold) for fold in folds],
        "candidate_fit_sets": 12,
        "inner_alpha_models": 36,
        "inner_two_target_solves": 72,
        "pooled_inner_row_mask": "original full scored mask on held fit sequences",
        "training_row_mask": "scored rows at stride 40 after whole-sequence causal history",
    }
    return tables, selected, draws, counts


def refit_final(
    base_rows: dict[str, np.ndarray],
    targets: np.ndarray,
    selected: dict[str, int],
    draws: dict[str, ProjectionDraw],
    progress: Callable[[dict], None],
) -> tuple[
    dict[str, dict[str, RidgeModel]],
    dict[str, BaseNormalizer],
    dict[str, dict],
    dict[str, float],
]:
    """Refit selected and fixed-alpha-one heads on all64 fit components."""
    models, normalizers, diagnostics, refit_seconds = {}, {}, {}, {}
    for base_name, base in base_rows.items():
        normalizer = fit_base_normalizer(base)
        normalizers[base_name] = normalizer
        pair = augment_pair(base, normalizer, draws[base_name])
        diagnostics[base_name] = projection_diagnostics(base, normalizer, draws[base_name])
        for activation_index, activation in enumerate(("identity", "tanh")):
            case = f"{base_name}_{activation}64"
            alpha = int(selected[case])
            alphas = (1,) if alpha == 1 else (1, alpha)
            started = time.process_time()
            fitted = fit_candidates(pair[activation_index], targets, alphas)
            models[case] = {
                "selected": fitted[alpha],
                "alpha1_baseline": fitted[1],
            }
            refit_seconds[case] = time.process_time() - started
            progress(
                {
                    "stage": "final_refit_complete",
                    "case": case,
                    "selected_alpha": alpha,
                    "training_rows": int(len(targets)),
                }
            )
    return models, normalizers, diagnostics, refit_seconds
