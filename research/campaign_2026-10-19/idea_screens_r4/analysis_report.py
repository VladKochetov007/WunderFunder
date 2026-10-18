"""Numeric model archives and compact R4 result/contrast construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from idea_screens_r2.ridge_math import RidgeModel

from .analysis_fit import CASE_ACTIVATION, CASE_BASE, CASE_NAMES
from .random_features import BaseNormalizer, ProjectionDraw


def digest(path: Path) -> str:
    """Hash a source or bounded output file with a fixed-size read buffer."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write one create-only JSON evidence record."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _model_payload(model: RidgeModel, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_ridge_mean": model.mean,
        f"{prefix}_ridge_scale": model.scale,
        f"{prefix}_coefficients": model.coefficients,
        f"{prefix}_intercepts": model.intercepts,
        f"{prefix}_coefficient_norms": np.asarray(model.norms, dtype=np.float64),
    }


def save_models(
    output: Path,
    models: dict[str, dict[str, RidgeModel]],
    draws: dict[str, ProjectionDraw],
    normalizers: dict[str, BaseNormalizer],
) -> dict[str, dict[str, int | str]]:
    """Save final numeric ridge and projection parameters without pickle."""
    archives = {}
    for case in CASE_NAMES:
        base = CASE_BASE[case]
        path = output / f"models_{case}.npz"
        payload = {
            "projection_weights": draws[base].weights,
            "projection_bias": draws[base].bias,
            "projection_seed": np.asarray(draws[base].seed, dtype=np.int64),
            "base_mean": normalizers[base].mean,
            "base_scale": normalizers[base].scale,
            **_model_payload(models[case]["selected"], "selected"),
            **_model_payload(models[case]["alpha1_baseline"], "alpha1_baseline"),
        }
        with path.open("xb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
        archives[case] = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "numeric_npz_no_pickle": True,
            "projection_seed": draws[base].seed,
            "projection_dimension": int(draws[base].weights.shape[0]),
            "projection_width": int(draws[base].weights.shape[1]),
        }
    return archives


def _metric_delta(left: dict, right: dict) -> dict[str, float]:
    return {key: float(left[key] - right[key]) for key in ("t0", "t1", "wp")}


def _primary_contrasts(outer: dict[str, dict]) -> dict[str, dict]:
    contrasts = {}
    for base in ("raw112", "raw_signed224"):
        identity, tanh = f"{base}_identity64", f"{base}_tanh64"
        comparisons = {}
        for label in ("selected", "alpha1_baseline"):
            left = outer[tanh][label]
            right = outer[identity][label]
            drops = []
            left_drop = left["drop_one_component"]
            right_drop = right["drop_one_component"]
            if [row["omitted_component"] for row in left_drop] != [
                row["omitted_component"] for row in right_drop
            ]:
                raise ValueError("paired R4 cases have different drop-one components")
            for a, b in zip(left_drop, right_drop, strict=True):
                if a["official_metric"] is None or b["official_metric"] is None:
                    delta = None
                else:
                    delta = _metric_delta(a["official_metric"], b["official_metric"])
                drops.append(
                    {"omitted_component": a["omitted_component"], "tanh_minus_identity": delta}
                )
            comparisons[label] = {
                "full_tanh_minus_identity_t0_t1_wp": _metric_delta(
                    left["official_metric"], right["official_metric"]
                ),
                "drop_one_fixed_prediction_tanh_minus_identity": drops,
            }
        contrasts[base] = comparisons
    return contrasts


def _reference_contrasts(
    outer: dict[str, dict], reference: dict, reference_sha256: str
) -> dict[str, dict]:
    if reference.get("schema") != "bounded_r3_component_factorial_nested_ridge_v1":
        raise ValueError("R3 reference result schema differs")
    if (
        reference.get("sample_manifest_sha256")
        != "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
    ):
        raise ValueError("R3 reference result does not bind the R4 sample")
    contrasts = {}
    for base, r3_case in (
        ("raw112", "full64_raw112"),
        ("raw_signed224", "full64_raw_signed_ema_hl32"),
    ):
        reference_case = reference["cases"][r3_case]["outer"]
        reference_metrics = {
            label: reference_case[label]["official_metric"]
            for label in ("selected", "alpha1_baseline")
        }
        contrasts[base] = {}
        for activation in ("identity", "tanh"):
            case = f"{base}_{activation}64"
            contrasts[base][case] = {
                label: {
                    "r4_minus_r3_linear_t0_t1_wp": _metric_delta(
                        outer[case][label]["official_metric"],
                        reference_metrics["selected" if label == "selected" else "alpha1_baseline"],
                    )
                }
                for label in ("selected", "alpha1_baseline")
            }
    return {"result_sha256": reference_sha256, "bases": contrasts}


def build_result(
    output: Path,
    sample_sha256: str,
    preparation_sha256: str,
    reference_sha256: str,
    source_manifest_sha256: str,
    sample_binding: dict,
    fit_counts: dict,
    cv_tables: dict[str, dict],
    selections: dict[str, int],
    cv_counts: dict,
    final_models: dict[str, dict[str, RidgeModel]],
    refit_seconds: dict[str, float],
    archives: dict[str, dict],
    draws: dict[str, ProjectionDraw],
    base_normalizers: dict[str, BaseNormalizer],
    final_projection_stats: dict[str, dict],
    outer: dict[str, dict],
    outer_cost: dict[str, float | int],
    elapsed_wall_s: float,
    elapsed_cpu_s: float,
    maxrss_kib: int,
    reference: dict,
) -> dict:
    """Write fixed-case outcomes, paired contrasts, reference deltas and DONE receipt."""
    cases = {}
    for case in CASE_NAMES:
        selected_model = final_models[case]["selected"]
        baseline = final_models[case]["alpha1_baseline"]
        base = CASE_BASE[case]
        cases[case] = {
            "base": base,
            "activation": CASE_ACTIVATION[case],
            "projection_seed": draws[base].seed,
            "feature_dimension": int(selected_model.coefficients.shape[0]),
            "selected_alpha": int(selections[case]),
            "selected_coefficient_l2_norm_by_target": list(selected_model.norms),
            "alpha1_coefficient_l2_norm_by_target": list(baseline.norms),
            "base_normalizer_scale_min": float(base_normalizers[base].scale.min()),
            "base_normalizer_scale_max": float(base_normalizers[base].scale.max()),
            "projection_diagnostics": final_projection_stats[base],
            "final_refit_cpu_s": refit_seconds[case],
            "inner_cv": cv_tables[case],
            "model_archive": archives[case],
            "outer": outer[case],
        }
    result = {
        "schema": "bounded_r4_fixed_random_features_v1",
        "metric_id": "global_wp_clip2_v1",
        "sample_manifest_sha256": sample_sha256,
        "preparation_manifest_sha256": preparation_sha256,
        "reference_result_sha256": reference_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "fit_selection": {
            "alpha_grid": list((1, 100, 10_000)),
            "training_stride": 40,
            "history_before_stride": True,
            "fold_seed": 19_025,
            "shared_alpha_for_two_targets": True,
            "normalization": "fold-train-only base mean/std then R2 augmented fit-only normalizer; final refit on all64 fit components",
            "projection": "one seeded N(0, 1/sqrt(d)) W and uniform[-1,1] bias per base, shared by identity/tanh pair",
            "fixed_projection_seeds": {
                key: value for key, value in (("raw112", 19_031), ("raw_signed224", 19_032))
            },
            "selection_rule": "maximum pooled held-fit-component official mean WP; exact ties choose larger alpha",
            "fit_order": sample_binding["fit_full64_component_order"],
            "folds": cv_counts["budget_folds"],
        },
        "counts": {
            **fit_counts,
            "eval_sequences": 32,
            "eval_components": 32,
            "eval_scored_rows": outer_cost["outer_scored_rows"],
            "inner_alpha_models": cv_counts["inner_alpha_models"],
            "inner_two_target_solves": cv_counts["inner_two_target_solves"],
            "final_refit_calls": 4,
            "final_unique_alpha_model_sets": sum(len({1, selections[case]}) for case in CASE_NAMES),
            "final_target_solves": 2 * sum(len({1, selections[case]}) for case in CASE_NAMES),
            "outer_prediction_calls": 32 * 4 * 2,
            "failures": [],
        },
        "cost": {
            "outer": outer_cost,
            "total_process_cpu_s": elapsed_cpu_s,
            "total_wall_s": elapsed_wall_s,
            "max_rss_kib": maxrss_kib,
            "final_refit_cpu_s_by_case": refit_seconds,
        },
        "cases": cases,
        "primary_tanh_minus_identity_contrasts": _primary_contrasts(outer),
        "secondary_r3_linear_reference_contrasts": _reference_contrasts(
            outer, reference, reference_sha256
        ),
        "sample_binding": sample_binding,
        "interpretation_limits": [
            "The fixed random projection is not a learned MLP and one projection per base is not seed replication.",
            "The R3 F0 evaluation panel is exposed discovery evidence, not a fresh pristine holdout.",
            "Identity features add no linear span when original base features are retained; their control is feature-count and penalty geometry.",
            "R3 reference deltas are descriptive same-panel comparisons and do not establish an independent replication or full-recipe gain.",
        ],
    }
    result_path = output / "RESULT.json"
    write_json(result_path, result)
    write_json(
        output / "DONE.json",
        {
            "result_sha256": digest(result_path),
            "result_bytes": result_path.stat().st_size,
            "sample_manifest_sha256": sample_sha256,
            "preparation_manifest_sha256": preparation_sha256,
            "reference_result_sha256": reference_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "model_archives": archives,
            "elapsed_wall_s": elapsed_wall_s,
            "elapsed_cpu_s": elapsed_cpu_s,
            "max_rss_kib": maxrss_kib,
        },
    )
    return result
