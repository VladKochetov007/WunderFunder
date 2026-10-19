"""Compact scalar result assembly for the fixed-step sampler pair."""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np

from .importance_math import BATCH_SIZE, DRAW_ROUNDS, UNIFORM_SEED
from .importance_training import EXPECTED_UPDATES, INITIALIZATION_SEED

FIT_ROWS = 31_808
EVAL_ROWS = 59_436


def named_sampler_score(r7_pair_score: Mapping[str, object]) -> dict[str, object]:
    """Rename reused R7 score-helper arms without carrying its variance diagnostic."""
    old_arms = r7_pair_score["arms"]
    if not isinstance(old_arms, Mapping):
        raise ValueError("reused score helper did not return arm mappings")
    uniform = old_arms["control"]
    importance = old_arms["heteroskedastic"]
    if not isinstance(uniform, Mapping) or not isinstance(importance, Mapping):
        raise ValueError("reused score helper returned malformed arm results")
    clean_arms = {
        name: {key: value for key, value in arm.items() if key != "diagnostics"}
        for name, arm in (("uniform", uniform), ("importance", importance))
    }
    return {
        "metric_id": r7_pair_score["metric_id"],
        "components": r7_pair_score["components"],
        "rows": r7_pair_score["rows"],
        "component_sensitivity_is_not_independent_replication": True,
        "arms": clean_arms,
        "primary_delta_importance_minus_uniform_wp": r7_pair_score[
            "primary_delta_heteroskedastic_minus_control_wp"
        ],
    }


def _distribution_summary(
    norms: np.ndarray, probabilities: np.ndarray, corrections: np.ndarray
) -> dict[str, object]:
    """Summarize fixed initial norms, sampling mass, and inverse-probability weights."""
    return {
        "rows": int(len(norms)),
        "probability_sum": float(probabilities.sum(dtype=np.float64)),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "effective_sample_size": float(1.0 / np.square(probabilities).sum(dtype=np.float64)),
        "correction_min": float(corrections.min()),
        "correction_max": float(corrections.max()),
        "gradient_norm_min": float(norms.min()),
        "gradient_norm_median": float(np.median(norms)),
        "gradient_norm_mean": float(norms.mean(dtype=np.float64)),
        "gradient_norm_max": float(norms.max()),
        "zero_gradient_rows": int((norms == 0.0).sum()),
    }


def build_result(
    *,
    identities: dict[str, str],
    preparation_sha256: str,
    mu_w: float,
    initialization: dict[str, str],
    norms: np.ndarray,
    probabilities: np.ndarray,
    corrections: np.ndarray,
    uniform_second_moment: float,
    importance_second_moment: float,
    uniform_fit: dict[str, object],
    importance_fit: dict[str, object],
    score: dict[str, object],
    archive: dict[str, object],
    runtime: dict[str, object],
    phase_seconds: dict[str, float],
    started_wall: float,
    started_cpu: float,
) -> dict[str, object]:
    """Build a JSON-safe outcome with scalar metrics and private archive metadata."""
    return {
        "schema": "nf19.r11_initial_gradient_importance.result.v1",
        "status": "completed",
        **identities,
        "preparation_manifest_sha256": preparation_sha256,
        "objective": {
            "definition": "per-row sum_t abs(clip(y,-2,2))*(mean-clip(y,-2,2))^2 / mu_w",
            "mu_w": mu_w,
            "mu_w_definition": "mean across N fitting rows of the two-target absolute clipped-label weight sum",
            "equivalent_global_weight_denominator": FIT_ROWS * mu_w,
            "sampler_estimator_unbiased_before_clipping_and_adamw_only": True,
            "full_trajectory_or_final_model_unbiasedness_claim": False,
        },
        "model": "FP32 Linear224-64-GELU-Linear64-2 mean plus identical unused zero variance head",
        "initialization_seed": INITIALIZATION_SEED,
        "initial_state_sha256": initialization,
        "fit_rows": FIT_ROWS,
        "evaluation_scored_rows": EVAL_ROWS,
        "samplers": {
            "uniform": "with replacement, p_i=1/N",
            "importance": "with replacement, p_i=0.5/N + 0.5*g_i/sum_j(g_j), g_i fixed at common theta0",
            "importance_weight": "1/(N*p_i), bounded by 2 by the uniform mixture floor",
            "common_uniform_seed": UNIFORM_SEED,
            "common_uniform_draws": DRAW_ROUNDS * FIT_ROWS,
            "common_random_numbers": "same U stream, arm-specific inverse CDF; sampled row sequences differ intentionally",
            "importance_distribution": _distribution_summary(norms, probabilities, corrections),
            "fixed_theta0_uncentered_gradient_norm_second_moment": {
                "uniform": uniform_second_moment,
                "importance": importance_second_moment,
                "importance_to_uniform_ratio": importance_second_moment / uniform_second_moment
                if uniform_second_moment > 0.0
                else None,
                "interpretation": "norm-only uncentered second moment, not variance or training-wide variance reduction",
            },
        },
        "budget": {
            "draw_rounds_per_arm": DRAW_ROUNDS,
            "rows_per_round": FIT_ROWS,
            "with_replacement_draws_per_arm": DRAW_ROUNDS * FIT_ROWS,
            "batch_size": BATCH_SIZE,
            "batches_per_round": 63,
            "batch_partition": "62 batches of 512 and one final batch of 64",
            "updates_per_arm": EXPECTED_UPDATES,
            "equal_examples_and_updates_not_equal_flops_or_wall_time": True,
            "not_unique_row_epochs": True,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0001,
            "foreach": False,
            "fused": False,
            "global_preclip_norm": 1.0,
            "autocast": False,
            "tf32": False,
        },
        "fit": {"uniform": uniform_fit, "importance": importance_fit},
        "final_evaluation": score,
        "archive": archive,
        "runtime": runtime,
        "phase_seconds": {
            **phase_seconds,
            "total_wall": time.perf_counter() - started_wall,
            "total_cpu": time.process_time() - started_cpu,
        },
        "scope_limit": "one exposed R3 F0 panel; conditional fixed-step result, no fresh holdout or efficiency claim",
        "predictions_or_model_parameter_arrays_in_json": False,
        "actual_completion_required": {
            "attempted_updates_per_arm": EXPECTED_UPDATES,
            "successful_updates_per_arm": EXPECTED_UPDATES,
            "nonfinite_count": 0,
        },
    }
