"""Assemble paired F0 diagnostic tables from streamed additive moments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import cast

import numpy as np
from residual_moments import _global_summary, _regime_definitions, _regime_rows

from wnn.gmetric import METRIC_CLIP, METRIC_ID


def _optional_delta(left: object, right: object) -> float | None:
    """Keep a descriptive difference undefined when either bin has zero weight."""
    if left is None or right is None:
        return None
    return float(right) - float(left)


def analyze_arrays(
    targets: np.ndarray,
    predictions: dict[str, np.ndarray],
    row_group: np.ndarray,
    component: np.ndarray,
    position: np.ndarray | None,
    comparisons: list[dict[str, str]],
) -> dict[str, object]:
    """Build endpoint, paired-regime and component tables without fitting or blending."""
    row_count = len(targets)
    if targets.dtype != np.float32 or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets must be float32 with shape (rows, 2)")
    if row_group.shape != (row_count,) or component.shape != (row_count,):
        raise ValueError("row-group and component vectors must align with targets")
    if position is not None and position.shape != (row_count,):
        raise ValueError("position must align with targets")
    if not predictions:
        raise ValueError("at least one prediction model is required")
    for name, values in predictions.items():
        if values.dtype != np.float32 or values.shape != targets.shape:
            raise ValueError(f"prediction model {name} must be float32 and align with targets")
        if not np.isfinite(values).all():
            raise ValueError(f"prediction model {name} contains nonfinite values")
    summaries = {name: _global_summary(targets, values) for name, values in predictions.items()}
    models: dict[str, object] = {}
    for name, values in predictions.items():
        regimes: dict[str, list[dict[str, object]]] = {}
        for regime, codes, labels, pairable in _regime_definitions(
            targets, values, component, position
        ):
            regimes[regime] = _regime_rows(
                targets, values, summaries[name], codes, labels, regime, pairable
            )
        models[name] = {
            "score": summaries[name]["official"],
            "global_moments": summaries[name]["targets"],
            "regimes": regimes,
        }
    pair_results: list[dict[str, object]] = []
    for pair in comparisons:
        baseline, candidate = pair["baseline"], pair["candidate"]
        if baseline not in predictions or candidate not in predictions:
            raise ValueError("comparison names an unavailable model")
        baseline_wp = float(cast(dict[str, object], summaries[baseline]["official"])["wp"])
        candidate_wp = float(cast(dict[str, object], summaries[candidate]["official"])["wp"])
        regime_deltas: dict[str, list[dict[str, object]]] = {}
        common = set(cast(dict[str, list[dict[str, object]]], models[baseline])["regimes"])
        common &= set(cast(dict[str, list[dict[str, object]]], models[candidate])["regimes"])
        for regime in sorted(common):
            base_rows = cast(dict[str, list[dict[str, object]]], models[baseline])["regimes"][
                regime
            ]
            cand_rows = cast(dict[str, list[dict[str, object]]], models[candidate])["regimes"][
                regime
            ]
            if not base_rows[0]["membership_shared_across_models"]:
                continue
            rows = []
            for left, right in zip(base_rows, cand_rows, strict=True):
                if left["category"] != right["category"] or left["target"] != right["target"]:
                    raise AssertionError("paired regime rows are not identically ordered")
                rows.append(
                    {
                        "category": left["category"],
                        "target": left["target"],
                        "global_wp_contribution_delta": float(right["global_wp_contribution"])
                        - float(left["global_wp_contribution"]),
                        "weighted_bias_delta": _optional_delta(
                            left["weighted_bias_prediction_minus_target"],
                            right["weighted_bias_prediction_minus_target"],
                        ),
                        "within_regime_rho_delta_descriptive_only": _optional_delta(
                            left["within_regime_rho_descriptive_only"],
                            right["within_regime_rho_descriptive_only"],
                        ),
                    }
                )
            delta_sum = sum(float(row["global_wp_contribution_delta"]) for row in rows)
            if abs(delta_sum - (candidate_wp - baseline_wp)) > 5e-10:
                raise AssertionError(
                    f"{regime} paired contribution deltas do not reproduce endpoint delta"
                )
            regime_deltas[regime] = rows
        pair_results.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "candidate_minus_baseline_wp": candidate_wp - baseline_wp,
                "shared_membership_regime_contributions": regime_deltas,
                "predicted_magnitude_bins_paired": False,
            }
        )
    return {
        "schema": "wnn.quality_gap.residual_diagnostic.v1",
        "metric_id": METRIC_ID,
        "selected_rows": row_count,
        "regime_policy": {
            "targets_and_predictions_clipped_to": [-METRIC_CLIP, METRIC_CLIP],
            "weight": "absolute clipped target",
            "label_bins_are_retrospective": True,
            "predicted_magnitude_membership_shared": False,
            "pooled_contribution_definition": "0.5 * regime Σw(y-μy_global)(p-μp_global) / sqrt(global Syy*global Spp), per target; sums across a partition and both targets to official WP.",
            "within_regime_rho": "Descriptive only; not additive to global WP.",
        },
        "models": models,
        "comparisons": pair_results,
        "position_status": "available"
        if position is not None
        else "unavailable: compact source bundle has no source-bound step_in_seq sidecar; no parquet scan was performed",
        "component_coverage": {
            "unique_row_groups": int(len(np.unique(row_group))),
            "unique_components": int(len(np.unique(component))),
            "row_group_counts": {
                str(int(key)): int(value)
                for key, value in zip(*np.unique(row_group, return_counts=True), strict=True)
            },
        },
    }


def write_csv(path: Path, report: dict[str, object]) -> None:
    """Write one flat per-model, per-target regime table beside the JSON report."""
    fields = [
        "model",
        "regime",
        "category",
        "target",
        "rows",
        "weight_mass",
        "weight_mass_share",
        "pooled_covariance_numerator",
        "pooled_covariance_share",
        "global_wp_contribution",
        "within_regime_rho_descriptive_only",
        "weighted_bias_prediction_minus_target",
        "weighted_mean_target",
        "weighted_mean_prediction",
        "weighted_prediction_on_target_slope",
        "weighted_prediction_on_target_intercept",
        "membership_shared_across_models",
    ]
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        models = cast(dict[str, dict[str, object]], report["models"])
        for model, values in models.items():
            regimes = cast(dict[str, list[dict[str, object]]], values["regimes"])
            for records in regimes.values():
                for record in records:
                    writer.writerow({"model": model, **record})
