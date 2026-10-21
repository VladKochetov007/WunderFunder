"""Canonical fixed-score and scale summaries for the native precision probe."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def weighted_diagnostics(
    y: np.ndarray, mask: np.ndarray, need: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    """Summarize raw scale and clipped target-weighted variation on scored rows."""
    valid = np.asarray(mask, dtype=bool) & np.asarray(need, dtype=bool)
    result: dict[str, Any] = {"selected_rows": int(valid.sum()), "channels": []}
    for channel in range(2):
        raw_y = np.asarray(y[valid, channel], dtype=np.float64)
        raw_p = np.asarray(prediction[valid, channel], dtype=np.float64)
        clipped_y, clipped_p = np.clip(raw_y, -2, 2), np.clip(raw_p, -2, 2)
        weight = np.abs(clipped_y)
        total = float(weight.sum())
        if total == 0:
            result["channels"].append(
                {
                    "prediction_rms": None,
                    "clipping_fraction": None,
                    "weighted_prediction_std": None,
                    "weighted_residual_std": None,
                }
            )
            continue
        pred_mean = float(np.dot(weight, clipped_p) / total)
        residual = clipped_y - clipped_p
        residual_mean = float(np.dot(weight, residual) / total)
        result["channels"].append(
            {
                "prediction_rms": float(np.sqrt(np.mean(raw_p**2))),
                "clipping_fraction": float(np.mean(np.abs(raw_p) > 2)),
                "weighted_prediction_std": float(
                    np.sqrt(np.dot(weight, (clipped_p - pred_mean) ** 2) / total)
                ),
                "weighted_residual_std": float(
                    np.sqrt(np.dot(weight, (residual - residual_mean) ** 2) / total)
                ),
            }
        )
    return result


def score_outputs(
    targets: dict[str, np.ndarray],
    selected: list[dict[str, Any]],
    needs: list[np.ndarray],
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Score fixed endpoint/member predictions using the canonical scorer."""
    sys.path.insert(0, str(ROOT / "research/campaign_2026-09-30/audit"))
    from campaign_score import moments, pooled

    row_positions = {int(group): index for index, group in enumerate(targets["row_group"])}
    scores: dict[str, list[np.ndarray]] = {name: [] for name in predictions}
    diagnostics: dict[str, list[dict[str, Any]]] = {name: [] for name in predictions}
    selected_rows = 0
    for position, member in enumerate(selected):
        target_index = row_positions[int(member["row_group"])]
        y, mask, need = targets["y"][target_index], targets["mask"][target_index], needs[position]
        inclusion = np.asarray(mask, dtype=bool) & np.asarray(need, dtype=bool)
        selected_rows += int(inclusion.sum())
        for name, value in predictions.items():
            prediction = value[position]
            scores[name].append(
                np.stack(
                    [
                        moments(y[:, channel], prediction[:, channel], inclusion)
                        for channel in range(2)
                    ]
                )
            )
            diagnostics[name].append(weighted_diagnostics(y, mask, need, prediction))
    result: dict[str, Any] = {
        "metric_id": "global_wp_clip2_v1",
        "sequence_count": len(selected),
        "selected_rows": selected_rows,
        "exact_global_wp": {},
        "prediction_diagnostics": {},
    }
    for name, values in scores.items():
        result["exact_global_wp"][name] = pooled(np.stack(values))
        result["prediction_diagnostics"][name] = diagnostics[name]
    baseline = result["exact_global_wp"]["original_tl3_reconstructed"]["wp"]
    cp5 = result["exact_global_wp"]["h512_cp5_cached_native"]["wp"]
    result["paired_deltas"] = {
        f"{name}_minus_original_tl3": value["wp"] - baseline
        for name, value in result["exact_global_wp"].items()
        if name != "original_tl3_reconstructed"
    }
    result["paired_deltas_vs_h512_cp5_cached_native"] = {
        f"{name}_minus_h512_cp5": value["wp"] - cp5
        for name, value in result["exact_global_wp"].items()
        if name != "h512_cp5_cached_native"
    }
    return result
