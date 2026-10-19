"""Compact scalar result schema for the fixed R7 endpoint pair."""

from __future__ import annotations

import time

from .model_math import BATCH_SIZE, EPOCHS, LEARNING_RATE

EXPECTED_FIT_ROWS = 31_808
EXPECTED_EVAL_ROWS = 59_436
EXPECTED_UPDATES_PER_ARM = 756


def build_result(
    *,
    identities: dict[str, str],
    preparation_sha256: str,
    initialization_sha256: str,
    control_fit: dict[str, object],
    candidate_fit: dict[str, object],
    score: dict[str, object],
    archive: dict[str, object],
    runtime: dict[str, object],
    source_phase_seconds: dict[str, object],
    evaluation_seconds: float,
    started_wall: float,
    started_cpu: float,
) -> dict[str, object]:
    """Build result-only scalars and references without exposing predictions."""
    return {
        "schema": "nf19.r7_mean_variance.result.v1",
        "status": "completed",
        **identities,
        "preparation_manifest_sha256": preparation_sha256,
        "model": "shared Linear224-64-GELU trunk; separate Linear64-2 mean and logvariance heads",
        "objective_control": "canonical clipped-target |clip(y)| pooled WMSE, common two-target minibatch denominator",
        "objective_candidate": "sum |clip(y)|*[exp(-clamp(s,-4,4))*(mu-clip(y))^2+clamp(s,-4,4)] / sum |clip(y)|; twice weighted Gaussian NLL",
        "initialization_sha256": initialization_sha256,
        "control_variance_head_unchanged": control_fit["variance_head_unchanged"],
        "fit_rows": EXPECTED_FIT_ROWS,
        "evaluation_scored_rows": EXPECTED_EVAL_ROWS,
        "epochs_per_arm": EPOCHS,
        "batch_size": BATCH_SIZE,
        "expected_batches_per_epoch": 63,
        "successful_updates_per_arm": EXPECTED_UPDATES_PER_ARM,
        "total_successful_updates": EXPECTED_UPDATES_PER_ARM * 2,
        "budget_interpretation": "equal examples and optimizer updates, not exact FLOPs or wall time",
        "optimizer": {
            "name": "AdamW",
            "lr": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0001,
            "foreach": False,
            "fused": False,
            "gradient_clip_global_norm": 1.0,
        },
        "fit_curves": {
            "control": control_fit["epochs"],
            "heteroskedastic": candidate_fit["epochs"],
        },
        "final_arm_summaries": {
            "control": {key: value for key, value in control_fit.items() if key != "epochs"},
            "heteroskedastic": {
                key: value for key, value in candidate_fit.items() if key != "epochs"
            },
        },
        "final_evaluation": score,
        "archive": archive,
        "runtime": runtime,
        "phase_seconds": {
            **source_phase_seconds,
            "control_wall": control_fit["wall_seconds"],
            "control_cpu": control_fit["cpu_seconds"],
            "heteroskedastic_wall": candidate_fit["wall_seconds"],
            "heteroskedastic_cpu": candidate_fit["cpu_seconds"],
            "evaluation": evaluation_seconds,
            "total_wall": time.perf_counter() - started_wall,
            "total_cpu": time.process_time() - started_cpu,
        },
        "scope_limit": "single exposed F0 R3 panel; no fresh holdout, replication, full-recipe or public-gain claim",
        "predictions_in_json_or_archive": False,
    }
