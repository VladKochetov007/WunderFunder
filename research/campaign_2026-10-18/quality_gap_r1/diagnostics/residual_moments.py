"""Streaming metric moments and regime aggregation primitives."""

from __future__ import annotations

import math
from typing import cast

import numpy as np

from wnn.gmetric import METRIC_CLIP, GlobalAccumulator

CHUNK_ROWS = 50_000
LABEL_EDGES = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0, METRIC_CLIP)
POSITION_EDGES = (99, 1_000, 5_000, 10_000, 15_000, 20_000)


def _bin_codes(values: np.ndarray, edges: tuple[float, ...] | tuple[int, ...]) -> np.ndarray:
    """Assign values to left-closed, right-open fixed retrospective bins."""
    return np.searchsorted(np.asarray(edges[1:-1]), values, side="right").astype(np.int16)


def _global_summary(targets: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    """Compute the official score and model/target global weighted centers."""
    accumulator = GlobalAccumulator()
    mass = np.zeros(2, dtype=np.float64)
    sum_y = np.zeros(2, dtype=np.float64)
    sum_p = np.zeros(2, dtype=np.float64)
    rows = len(targets)
    for start in range(0, rows, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, rows)
        y = np.clip(targets[start:stop], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        p = np.clip(predictions[start:stop], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        weight = np.abs(y)
        mass += weight.sum(axis=0)
        sum_y += (weight * y).sum(axis=0)
        sum_p += (weight * p).sum(axis=0)
        accumulator.add(
            targets[start:stop], predictions[start:stop], np.ones(stop - start, dtype=bool)
        )
    means_y = np.divide(sum_y, mass, out=np.zeros_like(sum_y), where=mass > 0)
    means_p = np.divide(sum_p, mass, out=np.zeros_like(sum_p), where=mass > 0)
    s_yy = np.zeros(2, dtype=np.float64)
    s_pp = np.zeros(2, dtype=np.float64)
    s_yp = np.zeros(2, dtype=np.float64)
    for start in range(0, rows, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, rows)
        y = np.clip(targets[start:stop], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        p = np.clip(predictions[start:stop], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        weight = np.abs(y)
        dy, dp = y - means_y, p - means_p
        s_yy += (weight * dy**2).sum(axis=0)
        s_pp += (weight * dp**2).sum(axis=0)
        s_yp += (weight * dy * dp).sum(axis=0)
    official = accumulator.result()
    target_rows = []
    for target in range(2):
        denom = math.sqrt(float(s_yy[target] * s_pp[target]))
        rho = 0.0 if denom <= 1e-8 else float(s_yp[target] / denom)
        if abs(rho - float(official[f"t{target}"])) > 2e-10:
            raise AssertionError("centered pooled moments disagree with official metric")
        target_rows.append(
            {
                "target": target,
                "weight_mass": float(mass[target]),
                "weighted_mu_y": float(means_y[target]) if mass[target] else None,
                "weighted_mu_p": float(means_p[target]) if mass[target] else None,
                "Syy": float(s_yy[target]),
                "Spp": float(s_pp[target]),
                "Syp": float(s_yp[target]),
                "rho": rho,
                "official_rho": float(official[f"t{target}"]),
            }
        )
    return {"official": official, "targets": target_rows}


def _regime_rows(
    targets: np.ndarray,
    predictions: np.ndarray,
    global_summary: dict[str, object],
    codes_by_target: tuple[np.ndarray, np.ndarray],
    categories: list[object],
    regime: str,
    pairable: bool,
) -> list[dict[str, object]]:
    """Accumulate globally centered additive terms and descriptive local calibration."""
    rows = len(targets)
    target_summary = cast(list[dict[str, object]], global_summary["targets"])
    output: list[dict[str, object]] = []
    for target in range(2):
        global_row = target_summary[target]
        mu_y = 0.0 if global_row["weighted_mu_y"] is None else float(global_row["weighted_mu_y"])
        mu_p = 0.0 if global_row["weighted_mu_p"] is None else float(global_row["weighted_mu_p"])
        mass_total = float(global_row["weight_mass"])
        s_yy, s_pp, s_yp = (
            float(global_row["Syy"]),
            float(global_row["Spp"]),
            float(global_row["Syp"]),
        )
        denom = math.sqrt(max(0.0, s_yy * s_pp))
        total_cov = s_yp
        stats = np.zeros((len(categories), 7), dtype=np.float64)
        codes = codes_by_target[target]
        for start in range(0, rows, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, rows)
            code = codes[start:stop]
            y = np.clip(targets[start:stop, target], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
            p = np.clip(predictions[start:stop, target], -METRIC_CLIP, METRIC_CLIP).astype(
                np.float64
            )
            weight = np.abs(y)
            dy, dp = y - mu_y, p - mu_p
            values = (
                weight,
                weight * y,
                weight * p,
                weight * dy * dp,
                weight * dy**2,
                weight * dp**2,
            )
            stats[:, 0] += np.bincount(code, minlength=len(categories))
            for index, value in enumerate(values, start=1):
                stats[:, index] += np.bincount(code, weights=value, minlength=len(categories))
        for index, category in enumerate(categories):
            count, mass, sum_y, sum_p, cov_global, var_y_global, var_p_global = stats[index]
            has_mass = mass > 0.0
            mean_y_value = sum_y / mass if has_mass else mu_y
            mean_p_value = sum_p / mass if has_mass else mu_p
            cov_local = cov_global - mass * (mean_y_value - mu_y) * (mean_p_value - mu_p)
            var_y_local = max(0.0, var_y_global - mass * (mean_y_value - mu_y) ** 2)
            var_p_local = max(0.0, var_p_global - mass * (mean_p_value - mu_p) ** 2)
            local_denom = math.sqrt(var_y_local * var_p_local)
            local_rho = (
                None
                if not has_mass
                else 0.0
                if local_denom <= 1e-8
                else float(np.clip(cov_local / local_denom, -1.0, 1.0))
            )
            slope = cov_local / var_y_local if var_y_local > 1e-8 else None
            pooled = 0.0 if denom <= 1e-8 else 0.5 * cov_global / denom
            output.append(
                {
                    "regime": regime,
                    "category": category,
                    "target": target,
                    "rows": int(count),
                    "weight_mass": float(mass),
                    "weight_mass_share": float(mass / mass_total) if mass_total else 0.0,
                    "pooled_covariance_numerator": float(cov_global),
                    "pooled_covariance_share": float(cov_global / total_cov)
                    if abs(total_cov) > 1e-12
                    else None,
                    "global_wp_contribution": float(pooled),
                    "weighted_mean_target": float(mean_y_value) if has_mass else None,
                    "weighted_mean_prediction": float(mean_p_value) if has_mass else None,
                    "weighted_bias_prediction_minus_target": float(mean_p_value - mean_y_value)
                    if has_mass
                    else None,
                    "weighted_prediction_on_target_slope": None if slope is None else float(slope),
                    "weighted_prediction_on_target_intercept": None
                    if slope is None
                    else float(mean_p_value - slope * mean_y_value),
                    "within_regime_rho_descriptive_only": local_rho,
                    "membership_shared_across_models": pairable,
                }
            )
    official_wp = float(cast(dict[str, object], global_summary["official"])["wp"])
    if abs(sum(float(row["global_wp_contribution"]) for row in output) - official_wp) > 3e-10:
        raise AssertionError(f"{regime} contributions do not sum to official WP")
    return output


def _regime_definitions(
    targets: np.ndarray,
    predictions: np.ndarray,
    component: np.ndarray,
    position: np.ndarray | None,
) -> list[tuple[str, tuple[np.ndarray, np.ndarray], list[object], bool]]:
    """Build fixed retrospective bins; predicted-magnitude membership is model-specific."""
    y_clipped = np.clip(targets, -METRIC_CLIP, METRIC_CLIP)
    label_codes = _bin_codes(np.abs(y_clipped), LABEL_EDGES)
    label_names = [
        f"[{a:g},{b:g}{']' if b == METRIC_CLIP else ')'}"
        for a, b in zip(LABEL_EDGES[:-1], LABEL_EDGES[1:], strict=True)
    ]
    component_ids = np.unique(component)
    component_codes = np.searchsorted(component_ids, component).astype(np.int16)
    component_names = [int(value) for value in component_ids]
    definitions: list[tuple[str, tuple[np.ndarray, np.ndarray], list[object], bool]] = [
        (
            "retrospective_label_abs_clipped",
            (label_codes[:, 0], label_codes[:, 1]),
            label_names,
            True,
        ),
        ("component", (component_codes, component_codes), component_names, True),
    ]
    if position is not None:
        position_codes = _bin_codes(position, POSITION_EDGES)
        position_names = [
            f"[{a},{b})" for a, b in zip(POSITION_EDGES[:-1], POSITION_EDGES[1:], strict=True)
        ]
        definitions.append(
            ("sequence_position", (position_codes, position_codes), position_names, True)
        )
    pred = np.asarray(predictions)
    pred_clipped = np.clip(pred, -METRIC_CLIP, METRIC_CLIP)
    pred_codes = _bin_codes(np.abs(pred_clipped), LABEL_EDGES)
    definitions.append(
        ("prediction_abs_clipped", (pred_codes[:, 0], pred_codes[:, 1]), label_names, False)
    )
    clip_codes = (np.abs(pred) > METRIC_CLIP).astype(np.int16)
    definitions.append(
        (
            "prediction_raw_clipping",
            (clip_codes[:, 0], clip_codes[:, 1]),
            ["within_clip", "clipped_gt_2"],
            False,
        )
    )
    return definitions
