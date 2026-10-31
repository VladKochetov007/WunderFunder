from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from wnn.data import SEQUENCE_LENGTH, WARMUP


def build_stage4_ensemble_result(
    candidates: Sequence[Mapping[str, object]],
    weights: np.ndarray,
    score: Mapping[str, object],
    trial_count: int,
    warmstart_weights: np.ndarray,
    warmstart_mse: float,
    auxiliary_sha: str,
    primary_sha: str,
    control_members: list[list[object]],
    tl3_reference_wp: object,
) -> dict[str, object]:
    """Build the durable comparison report for an inner-control simplex search."""
    return {
        "schema": "wnn.multistage-stage4-ensemble.v1",
        "status": "complete",
        "scope": "inner_train_proxy",
        "auxiliary_split_sha256": auxiliary_sha,
        "primary_split_sha256": primary_sha,
        "evaluation_source": "train",
        "evaluation_members": control_members,
        "evaluation_mask": "all_after_warmup AND row_index>=99",
        "selected_rows": len(control_members) * (SEQUENCE_LENGTH - WARMUP),
        "optimization": {
            "warm_start": {
                "method": "simplex_constrained_weighted_least_squares",
                "objective": "weighted_mse",
                "row_weight": "abs(clipped_target)",
                "weighted_mse": warmstart_mse,
                "weights": warmstart_weights.tolist(),
            },
            "optuna_search": {
                "sampler": "TPE",
                "objective": "pooled_clipped_wp_after_weighted_blend",
                "trial_count": trial_count,
                "complete_trial_count": trial_count,
                "weights_nonnegative": True,
                "weights_sum": float(weights.sum()),
                "best_wp": score["wp"],
            },
        },
        "ensemble": {
            "members": [
                {
                    "name": item["name"],
                    "stage": item["stage"],
                    "trial_number": item["trial_number"],
                    "family": item["family"],
                    "config_sha256": item["config_sha256"],
                    "config_identity_sha256": item["config_identity_sha256"],
                    "config_path": item["config_path"],
                    "weight": float(weight),
                    "trial_result_path": item["trial_result_path"],
                    "trial_result_sha256": item["trial_result_sha256"],
                    "trial_request_path": item["trial_request_path"],
                    "trial_request_sha256": item["trial_request_sha256"],
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    "prediction_manifest_sha256": item["prediction_manifest_sha256"],
                    "producer_identity_sha256": item["producer_identity_sha256"],
                    "objective": item["objective"],
                }
                for item, weight in zip(candidates, weights, strict=True)
            ],
            "control_wp": dict(score),
        },
        "fixed_primary_tl3_reference": {
            "wp": tl3_reference_wp,
            "scope": "full_official_valid_primary_reference",
            "used_for_weight_search": False,
            "directly_comparable_to_inner_proxy": False,
        },
        "interpretation": "Inner-train proxy selection only; freeze weights before any primary-valid evaluation.",
    }
