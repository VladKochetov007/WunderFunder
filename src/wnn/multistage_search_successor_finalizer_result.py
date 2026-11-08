from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import canonical_json_sha256, file_sha256
from wnn.multistage_search_budget import utc_now


def build_successor_result(
    run_manifest: Mapping[str, object],
    request: Mapping[str, object],
    fit_dir: Path,
    output: Path,
    objective: Mapping[str, object],
    payload: Mapping[str, object],
    identity: Mapping[str, object],
    plots: Mapping[str, object],
    verification: Mapping[str, object],
    prediction_source: Mapping[str, object],
    score_source: Mapping[str, object],
    batch_size: int,
    stage: int,
    trial_number: int,
) -> dict[str, object]:
    """Describe a versioned configured-final endpoint and its exact score artifacts."""
    checkpoint = fit_dir / "model.pt"
    provenance = payload["provenance"]
    return {
        "schema": f"wnn.multistage-stage{stage}-result.v2",
        "stage": stage,
        "trial_number": trial_number,
        "family": request["family"],
        "params": request["params"],
        "pair_id": request.get("pair_id"),
        "pair_role": request.get("pair_role"),
        "pair_mate_trial_number": request.get("pair_mate_trial_number"),
        "initializer_sha256": request.get("initializer_sha256"),
        "teacher_cache_manifest_sha256": request.get("teacher_cache_manifest_sha256"),
        "teacher_pool_id": request.get("teacher_pool_id"),
        "reused_skip_control": request.get("reused_skip_control"),
        "proposal_key": request["proposal_key"],
        "config_sha256": request["config_sha256"],
        "config_identity_sha256": request["config_identity_sha256"],
        "profile_identity_sha256": request["profile_identity_sha256"],
        "training_source_revision": request["source_revision"],
        "training_source_sha256": request["source_sha256"],
        "evaluation_source_revision": run_manifest["evaluation_source_revision"],
        "evaluation_source_sha256": prediction_source["sha256"],
        "evaluation_score_script_sha256": score_source["files"]["scripts/score_sequence.py"],
        "profile_receipt_sha256": verification["native_profile_sha256"],
        "approved_profile_sha256": verification["approved_profile_sha256"],
        "profile_review_revision": verification["review"]["revision"],
        "profile_review_receipt_path": verification["review"]["path"],
        "profile_review_receipt_sha256": verification["review"]["sha256"],
        "profile_review_policy_sha256": verification["review"]["policy_sha256"],
        "profile_reviewer_script_sha256": verification["review"]["checker_sha256"],
        "selected_endpoint": {
            "selection_role": "configured_final",
            "epoch": payload["completed_epochs"],
            "step": provenance["global_updates"],
            "wp": objective["objective"]["value"],
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "epoch_index_sha256": verification["epoch_index_sha256"],
            "epoch_index_file_sha256": verification["epoch_index_file_sha256"],
            "identity_sha256": canonical_json_sha256(identity),
        },
        "objective": objective,
        "fit_output": str(fit_dir.resolve()),
        "fit_model_sha256": file_sha256(checkpoint),
        "fit_process_receipt_path": verification["fit_receipt"]["path"],
        "fit_process_receipt_sha256": verification["fit_receipt"]["sha256"],
        "prediction_manifest_sha256": file_sha256(output / "control_predictions" / "manifest.json"),
        "prediction_batch_size": batch_size,
        "score_receipt_sha256": file_sha256(output / "inner_objective.json"),
        "plots": plots,
        "postprocess_only": True,
        "scientific_fit_performed": False,
        "primary_valid_used": False,
        "status": "COMPLETE",
        "completed_at_utc": utc_now(),
    }


def postprocess_receipt(
    result_path: Path,
    fit_dir: Path,
    objective_path: Path,
    plots: Mapping[str, object],
    stage: int,
    trial_number: int,
) -> dict[str, object]:
    """Bind the postprocessor to the existing fit, native objective, and plots."""
    return {
        "schema": "wnn.multistage-successor-fixed-final-postprocess.v2",
        "status": "complete",
        "stage": stage,
        "trial_number": trial_number,
        "scientific_fit_performed": False,
        "selection_role": "configured_final",
        "checkpoint_sha256": file_sha256(fit_dir / "model.pt"),
        "objective_sha256": file_sha256(objective_path),
        "trial_result_sha256": file_sha256(result_path),
        "plots": plots,
        "completed_at_utc": utc_now(),
    }
