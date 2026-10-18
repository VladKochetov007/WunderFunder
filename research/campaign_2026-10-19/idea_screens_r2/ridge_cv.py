"""Component-held-out regularization screen with fixed causal R1 features."""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

import numpy as np

from wnn.gmetric import METRIC_ID, GlobalAccumulator

from .ridge_math import (
    FIT_ROW_STRIDE,
    FOLD_SEED,
    RidgeModel,
)
from .ridge_math import (
    collect_fit_rows as _collect_fit_rows,
)
from .ridge_math import (
    component_folds as _component_folds,
)
from .ridge_math import (
    fit_candidates as _fit_candidates,
)
from .ridge_math import (
    predict as _predict,
)
from .views import VIEW_NAMES, build_view_for

ALPHAS = (1, 100, 10_000)


def _cross_validate(
    sequences: list[dict[str, object]],
    feature_rows: dict[str, np.ndarray],
    target_rows: np.ndarray,
    row_components: np.ndarray,
    view: str,
    folds: list[set[int]],
) -> dict[str, object]:
    """Score all fixed alphas on complete held-component masks and pool globally."""
    pooled = {alpha: GlobalAccumulator() for alpha in ALPHAS}
    by_fold = [{alpha: GlobalAccumulator() for alpha in ALPHAS} for _ in folds]
    fold_rows = [0 for _ in folds]
    for fold_index, held_components in enumerate(folds):
        train_rows = ~np.isin(row_components, list(held_components))
        if not train_rows.any():
            raise ValueError("empty inner training partition")
        models = _fit_candidates(feature_rows[view][train_rows], target_rows[train_rows], ALPHAS)
        held_sequences = [
            item
            for item in sequences
            if item["role"] == "fit" and int(item["component"]) in held_components
        ]
        for sequence in held_sequences:
            full = build_view_for(np.asarray(sequence["x"], dtype=np.float32), view)
            y = np.asarray(sequence["y"], dtype=np.float32)
            mask = np.asarray(sequence["scored"], dtype=bool)[None, :]
            fold_rows[fold_index] += int(mask.sum())
            for alpha, model in models.items():
                prediction = _predict(full, model)
                pooled[alpha].add(y[None, :, :], prediction[None, :, :], mask)
                by_fold[fold_index][alpha].add(y[None, :, :], prediction[None, :, :], mask)
    scores: dict[str, object] = {}
    for alpha in ALPHAS:
        scores[str(alpha)] = {
            "pooled_held_fit_components": pooled[alpha].result(),
            "folds": [
                {
                    "fold": index,
                    "held_components": sorted(folds[index]),
                    "scored_rows": fold_rows[index],
                    "official_metric": by_fold[index][alpha].result(),
                }
                for index in range(len(folds))
            ],
        }
    selected = max(ALPHAS, key=lambda alpha: (pooled[alpha].result()["wp"], alpha))
    return {"alpha_scores": scores, "selected_alpha": selected}


def _outer_score(
    sequences: list[dict[str, object]], view: str, models: dict[str, RidgeModel]
) -> dict[str, object]:
    """Score untouched outer rows and fixed-prediction component drop-one panels."""
    eval_components = sorted(
        {int(item["component"]) for item in sequences if item["role"] == "eval"}
    )
    totals = {name: GlobalAccumulator() for name in models}
    by_component = {
        name: {component: GlobalAccumulator() for component in eval_components} for name in models
    }
    drops = {
        name: {component: GlobalAccumulator() for component in eval_components} for name in models
    }
    for sequence in sequences:
        if sequence["role"] != "eval":
            continue
        component = int(sequence["component"])
        features = build_view_for(np.asarray(sequence["x"], dtype=np.float32), view)
        y = np.asarray(sequence["y"], dtype=np.float32)
        mask = np.asarray(sequence["scored"], dtype=bool)[None, :]
        for name, model in models.items():
            prediction = _predict(features, model)
            totals[name].add(y[None, :, :], prediction[None, :, :], mask, group=component)
            by_component[name][component].add(y[None, :, :], prediction[None, :, :], mask)
            for omitted in eval_components:
                if omitted != component:
                    drops[name][omitted].add(y[None, :, :], prediction[None, :, :], mask)
    result: dict[str, object] = {}
    for name, accumulator in totals.items():
        full_metric = accumulator.result()
        component_rows = []
        drop_rows = []
        for component in eval_components:
            component_metric = by_component[name][component].result()
            component_rows.append(
                {
                    "component": component,
                    "selected_rows": component_metric["selected_rows"],
                    "official_metric": component_metric,
                    "pooled_raw_moments": accumulator.groups[component].tolist(),
                }
            )
            drop_metric = drops[name][component].result()
            drop_rows.append(
                {
                    "omitted_component": component,
                    "official_metric": drop_metric,
                    "delta_wp_from_full_fixed_predictions": drop_metric["wp"] - full_metric["wp"],
                }
            )
        result[name] = {
            "official_metric": full_metric,
            "component_moments": component_rows,
            "drop_one_component": drop_rows,
        }
    return result


def run(
    sequences: list[dict[str, object]], output: Path, sample_binding: dict[str, object]
) -> dict[str, object]:
    """Run the fixed fit-only R2 selection, refits and one exposed outer score.

    Args:
        sequences: Whole source-bound sequences from the exact R1 sample.
        output: Create-only result JSON path in a fresh output directory.
        sample_binding: Manifest SHA256 and exact sequence/file identity records.

    Returns:
        Compact per-view CV and outer-panel results, also written to ``output``.
    """
    fit_components = sorted({int(item["component"]) for item in sequences if item["role"] == "fit"})
    eval_components = {int(item["component"]) for item in sequences if item["role"] == "eval"}
    if set(fit_components) & eval_components:
        raise ValueError("fit and outer evaluation components must be disjoint")
    if not any(item["role"] == "eval" for item in sequences):
        raise ValueError("exact sample must include its fixed outer evaluation panel")
    folds = _component_folds(fit_components)
    fit_features, fit_targets, row_components, fit_sequences = _collect_fit_rows(sequences)
    eval_rows = sum(
        int(np.asarray(item["scored"], dtype=bool).sum())
        for item in sequences
        if item["role"] == "eval"
    )
    progress_path = output.parent / "incremental_progress.jsonl"
    if output.exists() or progress_path.exists():
        raise FileExistsError("R2 result or progress output already exists")
    variants = []
    with progress_path.open("x", encoding="utf-8") as progress:
        for view in VIEW_NAMES:
            cpu_started = time.process_time()
            cv = _cross_validate(sequences, fit_features, fit_targets, row_components, view, folds)
            selected = int(cv["selected_alpha"])
            final_alphas = (1,) if selected == 1 else (selected, 1)
            final_models = _fit_candidates(fit_features[view], fit_targets, final_alphas)
            selected_model = final_models[selected]
            baseline_model = final_models[1]
            outer = _outer_score(
                sequences,
                view,
                {"selected": selected_model, "alpha1_baseline": baseline_model},
            )
            metric = outer["selected"]["official_metric"]
            variant = {
                "view": view,
                "feature_dim": int(fit_features[view].shape[1]),
                "fit_sampled_rows": int(len(fit_targets)),
                "fit_sequences": fit_sequences,
                "outer_eval_scored_rows": eval_rows,
                "inner_folds": [sorted(fold) for fold in folds],
                "inner_cv": cv,
                "selected_alpha": selected,
                "i0_ablation_limit": (
                    "i0_only is raw columns 0:52; versus raw112 it drops both the i1 block and the eight tail features, so it cannot isolate i1's incremental effect"
                    if view == "i0_only"
                    else None
                ),
                "selection_rule": "maximum pooled held-fit-component official mean WP; exact ties choose larger alpha",
                "selected_coeff_l2_norm_by_target": list(selected_model.norms),
                "alpha1_coeff_l2_norm_by_target": list(baseline_model.norms),
                "outer": outer,
                "elapsed_cpu_s": time.process_time() - cpu_started,
            }
            variants.append(variant)
            progress.write(
                json.dumps(
                    {"view": view, "selected_alpha": selected, "outer_metric": metric},
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            progress.flush()
            os.fsync(progress.fileno())
    by_view = {str(item["view"]): item for item in variants}
    raw = by_view["raw"]["outer"]
    i0 = by_view["i0_only"]["outer"]
    for variant in variants:
        selected_metric = variant["outer"]["selected"]["official_metric"]
        baseline_metric = variant["outer"]["alpha1_baseline"]["official_metric"]
        variant["primary_comparisons"] = {
            "selected_wp_delta_vs_raw": selected_metric["wp"]
            - raw["selected"]["official_metric"]["wp"],
            "alpha1_wp_delta_vs_raw": baseline_metric["wp"]
            - raw["alpha1_baseline"]["official_metric"]["wp"],
            "selected_wp_delta_vs_i0_only": selected_metric["wp"]
            - i0["selected"]["official_metric"]["wp"],
            "alpha1_wp_delta_vs_i0_only": baseline_metric["wp"]
            - i0["alpha1_baseline"]["official_metric"]["wp"],
            "interpretation": "same exposed outer panel only; compare raw-prefixed views to raw and treat i0_only as a joint channel ablation",
        }
    result: dict[str, object] = {
        "schema": "bounded_r2_fit_only_alpha_ema_v1",
        "metric_id": METRIC_ID,
        "metric_source": inspect.getsourcefile(GlobalAccumulator),
        "sample_manifest_sha256": sample_binding["sha256"],
        "source_pins": sample_binding["records"],
        "fit_sampling": {
            "row_stride_after_full_causal_views": FIT_ROW_STRIDE,
            "fit_partition_only": True,
            "normalizer": "unweighted mean/std from each inner training partition; final refit uses all fit rows",
            "ridge_weight": "abs(clip(y, -2, 2))",
            "targets_share_alpha": True,
        },
        "grouping": {"seed": FOLD_SEED, "unit": "component", "folds": [sorted(x) for x in folds]},
        "candidate_count": {
            "inner_model_fits": len(VIEW_NAMES) * len(ALPHAS) * len(folds),
            "final_selected_and_alpha1_refits_max": 2 * len(VIEW_NAMES),
            "deduplicate_when_selected_alpha_is_1": True,
        },
        "views": variants,
        "numeric_note": "float64 normal equations for an exploratory CPU screen; not strict-FP32 campaign training",
        "outer_interpretation": "same exposed R1 panel; descriptive only, not confirmation or a fresh evaluation",
        "progress_file": progress_path.name,
    }
    with output.open("x", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, sort_keys=True, allow_nan=False)
        result_file.write("\n")
    return result
