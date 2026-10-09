from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from disagreement_gate import (
    BOOTSTRAP_REPLICATES,
    CLIP,
    WARMUP_ROWS,
    _paired_bootstrap,
)
from disagreement_screen import E514Inputs
from numpy.typing import NDArray

from wnn.gmetric import global_wp

EXPERIMENT_ID = "E-518"
HYPOTHESIS_ID = "H-211"
GATE_QUANTILE = 0.75
BETA_GRID = tuple(index / 100.0 for index in range(101))
EXPECTED_MANIFEST_SHA256 = "8de4dc6c1e2a9023510d3edbb3a46771579470bfb859ac50f451f7baa269d6fd"
EXPECTED_PARTITION_SHA256 = "e9e0cb9391789f54e34c1d02836d02c336c5d3a13bde6f9473196bbbc71da1c2"
EXPECTED_ROW_GROUPS_SHA256 = "e9584232498a4ac40ca0ad2bebeb1b136862e2de242c2ec4378ea425357aa4be"


def _fit_exact_wp_beta(
    targets: NDArray[np.float32],
    anchor: NDArray[np.float32],
    candidate: NDArray[np.float32],
    metric_mask: NDArray[np.bool_],
    route: NDArray[np.bool_],
) -> tuple[float, float]:
    """Choose the smallest beta maximizing calibration target-0 official WP."""
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError("targets must have shape [group, time, 2]")
    if anchor.shape != targets.shape or candidate.shape != targets.shape:
        raise ValueError("anchor and candidate shapes must match targets")
    if metric_mask.shape != targets.shape[:2] or route.shape != metric_mask.shape:
        raise ValueError("mask and route must match the group/time shape")
    if not metric_mask.any():
        raise ValueError("calibration mask selects no scored rows")

    target = np.clip(targets[:, :, 0].astype(np.float64), -CLIP, CLIP)[metric_mask]
    weights = np.abs(target)
    positive = weights > 0
    target, weights = target[positive], weights[positive]
    clipped_anchor = np.clip(anchor[:, :, 0].astype(np.float64), -CLIP, CLIP)[metric_mask][positive]
    clipped_candidate = np.clip(candidate[:, :, 0].astype(np.float64), -CLIP, CLIP)[metric_mask][positive]
    selected_route = route[metric_mask][positive]
    delta = clipped_candidate - clipped_anchor

    total_weight = float(weights.sum(dtype=np.float64))
    if total_weight <= 0:
        return 0.0, 0.0
    target_shifted = target - target[0]
    mean_target = float(np.sum(weights * target_shifted, dtype=np.float64) / total_weight)
    centered_target = target_shifted - mean_target
    variance_target = float(
        np.sum(weights * centered_target * centered_target, dtype=np.float64) / total_weight
    )

    def exact_target_wp(prediction: NDArray[np.float64]) -> float:
        prediction_shifted = prediction - prediction[0]
        mean_prediction = float(np.sum(weights * prediction_shifted, dtype=np.float64) / total_weight)
        pc = prediction_shifted - mean_prediction
        variance_prediction = float(
            np.sum(weights * pc * pc, dtype=np.float64) / total_weight
        )
        if np.sqrt(max(variance_target, 0.0)) <= 1e-8 or np.sqrt(max(variance_prediction, 0.0)) <= 1e-8:
            return 0.0
        covariance = float(
            np.sum(weights * centered_target * pc, dtype=np.float64) / total_weight
        )
        return float(np.clip(covariance / np.sqrt(variance_target * variance_prediction), -1.0, 1.0))

    scores: list[float] = []
    for beta in BETA_GRID:
        prediction = np.clip(
            clipped_anchor + beta * selected_route * delta, -CLIP, CLIP
        ).astype(np.float32).astype(np.float64)
        scores.append(exact_target_wp(prediction))
    return _select_smallest_beta_within_tolerance(tuple(scores))


def _select_smallest_beta_within_tolerance(
    scores: tuple[float, ...], *, tolerance: float = 1e-12
) -> tuple[float, float]:
    """Select the first grid beta within tolerance of the global score maximum."""
    if len(scores) != len(BETA_GRID) or not np.isfinite(scores).all():
        raise ValueError("one finite calibration score is required for each beta-grid point")
    maximum = max(scores)
    chosen = next(
        index for index, score in enumerate(scores) if maximum - score <= tolerance
    )
    return BETA_GRID[chosen], scores[chosen]


def _apply(
    anchor: NDArray[np.float32],
    candidate: NDArray[np.float32],
    beta: float,
    route: NDArray[np.bool_],
) -> NDArray[np.float32]:
    result = np.asarray(anchor, dtype=np.float32).copy()
    a = np.clip(anchor[:, :, 0].astype(np.float64), -CLIP, CLIP)
    c = np.clip(candidate[:, :, 0].astype(np.float64), -CLIP, CLIP)
    result[:, :, 0] = np.clip(a + beta * route * (c - a), -CLIP, CLIP).astype(np.float32)
    return result


def run_registered_e518(
    root: Path,
    inputs: E514Inputs,
    *,
    bootstrap_seed: int = 998_465_967,
) -> tuple[dict[str, Any], NDArray[np.float32], NDArray[np.float32]]:
    """Run the one registered metric-aligned gate screen on E-514 exposed inputs."""
    root = Path(root).resolve()
    if inputs.source_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("source manifest differs from E-518 registration")
    if inputs.chain_partition_sha256 != EXPECTED_PARTITION_SHA256:
        raise ValueError("chain partition differs from E-518 registration")
    row_groups_digest = hashlib.sha256(
        np.asarray(inputs.row_groups, dtype="<i8").tobytes()
    ).hexdigest()
    if inputs.row_groups.shape != (432,) or row_groups_digest != EXPECTED_ROW_GROUPS_SHA256:
        raise ValueError("row-group population/order differs from registration")
    if inputs.targets.shape != (432, 20_000, 2):
        raise ValueError("target tensor differs from the frozen E-514 population")
    if inputs.anchor.shape != inputs.targets.shape:
        raise ValueError("anchor tensor shape differs from targets")
    if inputs.candidate_predictions["ab12"].shape != inputs.targets.shape:
        raise ValueError("ab12 prediction tensor shape differs from targets")

    chain_ids = np.unique(inputs.chain_by_sequence).astype(np.int64)
    outer_folds = sorted(set(inputs.chain_to_outer_fold.values()))
    if outer_folds != [0, 1, 2] or chain_ids.size != 123:
        raise ValueError("chain membership differs from the frozen three-fold partition")
    warmup = np.arange(inputs.targets.shape[1]) >= WARMUP_ROWS
    metric_mask = inputs.is_scored & warmup[None, :]
    candidate = inputs.candidate_predictions["ab12"]
    anchor_t0 = np.clip(inputs.anchor[:, :, 0].astype(np.float64), -CLIP, CLIP)
    candidate_t0 = np.clip(candidate[:, :, 0].astype(np.float64), -CLIP, CLIP)
    abs_delta = np.abs(candidate_t0 - anchor_t0)
    gated_oof = inputs.anchor.copy()
    global_oof = inputs.anchor.copy()
    fold_records: list[dict[str, Any]] = []

    for outer_fold in outer_folds:
        calibration_rows = np.flatnonzero(
            np.asarray(
                [inputs.chain_to_outer_fold[int(chain)] != outer_fold for chain in inputs.chain_by_sequence],
                dtype=bool,
            )
        )
        evaluation_rows = np.flatnonzero(
            np.asarray(
                [inputs.chain_to_outer_fold[int(chain)] == outer_fold for chain in inputs.chain_by_sequence],
                dtype=bool,
            )
        )
        calibration_mask = metric_mask[calibration_rows]
        threshold = float(
            np.quantile(abs_delta[calibration_rows][calibration_mask], GATE_QUANTILE, method="linear")
        )
        calibration_route = abs_delta[calibration_rows] >= threshold
        global_route = np.ones_like(calibration_route, dtype=bool)
        gated_beta, gated_calibration_t0 = _fit_exact_wp_beta(
            inputs.targets[calibration_rows],
            inputs.anchor[calibration_rows],
            candidate[calibration_rows],
            calibration_mask,
            calibration_route,
        )
        global_beta, global_calibration_t0 = _fit_exact_wp_beta(
            inputs.targets[calibration_rows],
            inputs.anchor[calibration_rows],
            candidate[calibration_rows],
            calibration_mask,
            global_route,
        )
        evaluation_route = abs_delta[evaluation_rows] >= threshold
        gated_oof[evaluation_rows] = _apply(
            inputs.anchor[evaluation_rows], candidate[evaluation_rows], gated_beta, evaluation_route
        )
        global_oof[evaluation_rows] = _apply(
            inputs.anchor[evaluation_rows],
            candidate[evaluation_rows],
            global_beta,
            np.ones_like(evaluation_route, dtype=bool),
        )
        eval_mask = metric_mask[evaluation_rows]
        eval_weight = np.abs(
            np.clip(inputs.targets[evaluation_rows, :, 0].astype(np.float64), -CLIP, CLIP)
        ) * eval_mask
        routed = eval_mask & evaluation_route
        fold_records.append(
            {
                "outer_fold": int(outer_fold),
                "calibration_chains": int(sum(inputs.chain_to_outer_fold[int(c)] != outer_fold for c in chain_ids)),
                "evaluation_chains": int(sum(inputs.chain_to_outer_fold[int(c)] == outer_fold for c in chain_ids)),
                "calibration_groups": int(calibration_rows.size),
                "evaluation_groups": int(evaluation_rows.size),
                "threshold": threshold,
                "beta_gate": gated_beta,
                "beta_global": global_beta,
                "calibration_gate_t0": gated_calibration_t0,
                "calibration_global_t0": global_calibration_t0,
                "evaluation_gate_rows": int(routed.sum()),
                "evaluation_scored_rows": int(eval_mask.sum()),
                "evaluation_gate_row_fraction": float(routed.sum() / max(int(eval_mask.sum()), 1)),
                "evaluation_gate_official_weight_fraction": float(
                    (eval_weight * evaluation_route).sum() / max(float(eval_weight.sum()), 1e-12)
                ),
                "anchor": global_wp(
                    inputs.targets[evaluation_rows], inputs.anchor[evaluation_rows], eval_mask
                ),
                "gated": global_wp(
                    inputs.targets[evaluation_rows], gated_oof[evaluation_rows], eval_mask
                ),
                "global": global_wp(
                    inputs.targets[evaluation_rows], global_oof[evaluation_rows], eval_mask
                ),
            }
        )

    endpoints = [
        global_wp(inputs.targets, prediction, metric_mask)
        for prediction in (inputs.anchor, gated_oof, global_oof)
    ]
    bootstrap_intervals, gated_global_interval = _paired_bootstrap(
        inputs,
        (inputs.anchor, gated_oof, global_oof),
        metric_mask,
        chain_ids,
        bootstrap_seed,
    )
    positive_gate_folds = sum(row["beta_gate"] > 0 for row in fold_records)
    passed = bool(
        endpoints[1]["wp"] > endpoints[0]["wp"]
        and endpoints[1]["wp"] > endpoints[2]["wp"]
        and bootstrap_intervals[0][0] > 0
        and gated_global_interval[0][0] > 0
        and positive_gate_folds >= 2
    )
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed_exposed_metric_aligned_screen",
        "metric": "global_wp_clip2_v1",
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "chain_partition_sha256": inputs.chain_partition_sha256,
        "row_groups_sha256": row_groups_digest,
        "population": {"groups": 432, "chains": 123, "outer_folds": [41, 41, 41]},
        "exposure": "All input groups were previously exposed in E-514/E-517; exploratory only.",
        "candidate": "ab12, the least anchor-correlated E-514 member",
        "gate_quantile": GATE_QUANTILE,
        "beta_grid": list(BETA_GRID),
        "fit_objective": "maximize exact calibration global_wp_clip2_v1 target-0 WP after official clipping",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_seed_source_sha256": "3b8361afda9622252fe8da30596f2734c8be8f1753708b3274e0c5d74cf89b17",
        "anchor": endpoints[0],
        "gated": endpoints[1],
        "global": endpoints[2],
        "delta_gated_vs_anchor": float(endpoints[1]["wp"] - endpoints[0]["wp"]),
        "delta_gated_vs_global": float(endpoints[1]["wp"] - endpoints[2]["wp"]),
        "delta_global_vs_anchor": float(endpoints[2]["wp"] - endpoints[0]["wp"]),
        "gated_vs_anchor_ci95": bootstrap_intervals[0],
        "global_vs_anchor_ci95": bootstrap_intervals[1],
        "gated_vs_global_ci95": gated_global_interval[0],
        "positive_gate_folds": int(positive_gate_folds),
        "development_gate_passed": passed,
        "folds": fold_records,
        "new_public_gain_verified": False,
        "official_metric_score": None,
        "expected_public_gain": None,
        "public_forecast": None,
        "numerical_prior_update": None,
    }
    output_dir = root / "research/campaign_2026-10-06/bandit_continuation/disagreement_guided_H204"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "E518_GATED_OOF_PREDICTIONS.npy", gated_oof, allow_pickle=False)
    np.save(output_dir / "E518_GLOBAL_OOF_PREDICTIONS.npy", global_oof, allow_pickle=False)
    result["oof_gated_sha256"] = hashlib.sha256(memoryview(gated_oof).cast("B")).hexdigest()
    result["oof_global_sha256"] = hashlib.sha256(memoryview(global_oof).cast("B")).hexdigest()
    result["oof_gated_file_sha256"] = hashlib.sha256(
        (output_dir / "E518_GATED_OOF_PREDICTIONS.npy").read_bytes()
    ).hexdigest()
    result["oof_global_file_sha256"] = hashlib.sha256(
        (output_dir / "E518_GLOBAL_OOF_PREDICTIONS.npy").read_bytes()
    ).hexdigest()
    result["analysis_completed_at_utc"] = datetime.now(UTC).isoformat()
    (output_dir / "E518_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result, gated_oof, global_oof
