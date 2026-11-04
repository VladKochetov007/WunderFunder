from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_plot import export_stage_trial_plots
from wnn.multistage_search import (
    append_event,
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    read_json,
)
from wnn.multistage_search_adopt_fit_artifacts import (
    complete_preexisting_attempt,
    config_params,
    existing_adoption,
    link_fit_artifacts,
    prediction_batch_size,
)
from wnn.multistage_search_adopt_fit_score import score_adopted_fit
from wnn.multistage_search_adopt_fit_validate import (
    admission_attempt,
    validate_profile_and_fit,
    validate_terminal_owner,
)
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_metrics import select_best_control_epoch, validate_inner_objective
from wnn.training.sequence_config import load_config


def adopt_completed_fit(
    run_dir: Path,
    attempt_id: str,
    *,
    config_path: Path,
    approved_profile_path: Path,
    owner_receipt_path: Path,
    terminal_receipt_path: Path,
    gpu_key: str,
) -> dict[str, object]:
    """Import one terminal fit, score it with the pinned evaluator, and preserve its source receipts."""
    if not attempt_id or len(attempt_id) > 128:
        raise ValueError("preexisting fit attempt id is malformed")
    source_paths = (config_path, approved_profile_path, owner_receipt_path, terminal_receipt_path)
    if any(path.is_symlink() for path in source_paths):
        raise ValueError("completed-fit adoption inputs must be regular non-symlink files")
    run_manifest = read_json(run_dir / "run.json")
    admission = read_json(Path(str(run_manifest["admission"]["path"])))
    attempt = admission_attempt(admission, attempt_id)
    owner_path = owner_receipt_path.resolve(strict=True)
    terminal_path = terminal_receipt_path.resolve(strict=True)
    config_source = config_path.resolve(strict=True)
    approved_path = approved_profile_path.resolve(strict=True)
    owner, terminal = (
        read_json(owner_path),
        read_json(terminal_path),
    )
    config = load_config(config_source)
    native_ref = attempt.get("profile_receipt")
    if not isinstance(native_ref, Mapping):
        raise ValueError("preexisting fit does not bind its original native profile")
    native_raw_path = Path(str(native_ref.get("path")))
    if native_raw_path.is_symlink():
        raise ValueError("preexisting native profile receipt cannot be a symlink")
    native_path = native_raw_path.resolve(strict=True)
    native_sha = file_sha256(native_path)
    native, approved = read_json(native_path), read_json(approved_path)
    output_raw = Path(str(attempt.get("output_dir")))
    if output_raw.is_symlink():
        raise ValueError("preexisting fit output cannot be a symlink")
    output_source = output_raw.resolve(strict=True)
    deadline = read_json(run_dir / "campaign_deadline.json")
    validate_terminal_owner(
        attempt_id, attempt, owner, terminal, owner_path, output_source, deadline
    )
    validate_profile_and_fit(
        attempt,
        config,
        config_source,
        native,
        native_path,
        native_sha,
        approved,
        approved_path,
        output_source,
        admission,
        terminal,
    )
    existing = existing_adoption(run_dir, attempt_id, owner_path, terminal_path)
    if existing is not None:
        complete_preexisting_attempt(run_dir, attempt_id, existing)
        return existing
    family = attempt.get("family")
    trial_number = owner.get("trial_number")
    if (
        not isinstance(family, str)
        or not family
        or type(trial_number) is not int
        or trial_number >= 0
    ):
        raise ValueError("external fit must retain its family and original negative trial id")
    source_revision = attempt.get(
        "training_source_revision", terminal.get("training_source_revision")
    )
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("preexisting fit admission must retain its exact training-source revision")

    trial_dir = run_dir / "stage1" / "preexisting" / attempt_id
    if trial_dir.is_symlink():
        raise ValueError("preexisting adoption trial directory cannot be a symlink")
    trial_dir.mkdir(parents=True, exist_ok=True)
    if not trial_dir.is_dir():
        raise ValueError("preexisting adoption trial path is not a directory")
    config_copy = trial_dir / "config.json"
    if config_copy.is_symlink():
        raise ValueError("adoption config copy cannot be a symlink")
    if config_copy.exists():
        if not config_copy.is_file() or file_sha256(config_copy) != file_sha256(config_source):
            raise ValueError("partial adoption config differs from the verified source config")
    else:
        config_copy.write_bytes(config_source.read_bytes())
    fit_dir = trial_dir / "fit_output"
    if fit_dir.is_symlink():
        raise ValueError("adoption fit directory cannot be a symlink")
    fit_dir.mkdir(exist_ok=True)
    if not fit_dir.is_dir():
        raise ValueError("adoption fit path is not a directory")
    link_fit_artifacts(output_source, fit_dir)
    auxiliary = admission["data"]["auxiliary_manifest"]
    auxiliary_path = Path(str(auxiliary["path"])).resolve(strict=True)
    if file_sha256(auxiliary_path) != auxiliary.get("sha256"):
        raise ValueError("preexisting fit auxiliary split changed after admission")
    primary_sha = str(admission["data"]["primary_split_sha256"])
    selection = select_best_control_epoch(fit_dir, str(auxiliary["sha256"]))
    index = read_json(Path(str(selection["epoch_index_path"])))
    if index.get("identity") != native.get("identity"):
        raise ValueError("selected epoch index differs from the exact native profile identity")
    request = {
        "schema": "wnn.multistage-trial-plan.v1",
        "stage": 1,
        "trial_number": trial_number,
        "family": family,
        "params": config_params(config),
        "config_path": str(config_copy.resolve()),
        "config_sha256": file_sha256(config_copy),
        "profile_identity_sha256": canonical_json_sha256(native["identity"]),
        "profile_receipt_path": str(native_path),
        "profile_receipt_sha256": native_sha,
        "approved_profile_path": str(approved_path),
        "approved_profile_sha256": file_sha256(approved_path),
        "profile_review_sha256": attempt["profile_review_sha256"],
        "primary_split_sha256": primary_sha,
        "auxiliary_split_manifest": str(auxiliary_path),
        "auxiliary_split_sha256": auxiliary["sha256"],
        "source_revision": source_revision,
        "source_sha256": native["identity"]["source_sha256"],
        "objective_scope": "inner_train_proxy",
        "primary_valid_used": False,
        "external_attempt_id": attempt_id,
    }
    request_path = trial_dir / "trial_request.json"
    atomic_write_json(request_path, request, replace=False)
    score_receipts = score_adopted_fit(
        run_dir,
        run_manifest,
        admission,
        auxiliary_path,
        fit_dir,
        selection,
        request,
        gpu_key,
        trial_number,
    )
    objective_path = fit_dir / "inner_objective.json"
    objective = validate_inner_objective(
        objective_path,
        auxiliary_split_sha256=str(auxiliary["sha256"]),
        primary_split_sha256=primary_sha,
        expected_checkpoint_sha256=str(selection["checkpoint_sha256"]),
        expected_identity_sha256=str(selection["identity_sha256"]),
    )
    if abs(float(objective["objective"]["value"]) - float(selection["wp"])) > 1e-7:
        raise ValueError("source4 pooled score differs from the native selected control event")
    plots = export_stage_trial_plots(
        request_path, fit_dir, 1, force=True, plots_dir=trial_dir / "plots"
    )
    result = {
        "schema": "wnn.multistage-stage1-result.v1",
        "stage": 1,
        "trial_number": trial_number,
        "family": family,
        "params": request["params"],
        "proposal_key": canonical_json_sha256(
            {"attempt_id": attempt_id, "config_identity_sha256": canonical_json_sha256(config)}
        ),
        "config_sha256": file_sha256(config_copy),
        "profile_identity_sha256": request["profile_identity_sha256"],
        "training_source_revision": request["source_revision"],
        "training_source_sha256": request["source_sha256"],
        "evaluation_source_revision": run_manifest["evaluation_source_revision"],
        "evaluation_source_sha256": run_manifest["evaluation_source_sha256"],
        "evaluation_score_script_sha256": run_manifest["evaluation_score_script_sha256"],
        "profile_receipt_sha256": native_sha,
        "approved_profile_sha256": file_sha256(approved_path),
        "selected_endpoint": selection,
        "objective": objective,
        "fit_output": str(fit_dir.resolve()),
        "fit_model_sha256": file_sha256(fit_dir / "model.pt"),
        "fit_process_receipt_path": str(owner_path),
        "fit_process_receipt_sha256": file_sha256(owner_path),
        "terminal_receipt_path": str(terminal_path),
        "terminal_receipt_sha256": file_sha256(terminal_path),
        "prediction_manifest_sha256": file_sha256(
            fit_dir / "selected_control_predictions" / "manifest.json"
        ),
        "prediction_batch_size": prediction_batch_size(config),
        "score_receipts": score_receipts,
        "plots": plots,
        "status": "COMPLETE",
        "adopted_external_fit": True,
        "exit_observation": terminal["exit_observation"],
        "process_exit_code_observed": False,
        "completion_evidence": "native_terminal_status_checkpoint_and_four_epoch_index",
        "completed_at_utc": utc_now(),
    }
    result_path = trial_dir / "trial_result.json"
    atomic_write_json(result_path, result, replace=False)
    adoption = {
        "schema": "wnn.multistage-completed-fit-adoption.v1",
        "status": "complete",
        "attempt_id": attempt_id,
        "original_owner_receipt_path": str(owner_path),
        "original_owner_receipt_sha256": file_sha256(owner_path),
        "terminal_receipt_path": str(terminal_path),
        "terminal_receipt_sha256": file_sha256(terminal_path),
        "native_profile_sha256": native_sha,
        "approved_profile_sha256": file_sha256(approved_path),
        "config_sha256": file_sha256(config_copy),
        "fit_model_sha256": result["fit_model_sha256"],
        "epoch_index_sha256": selection["epoch_index_sha256"],
        "epoch_index_file_sha256": selection["epoch_index_file_sha256"],
        "objective_sha256": file_sha256(objective_path),
        "exit_observation": terminal["exit_observation"],
        "process_exit_code_observed": False,
        "trial_result_path": str(result_path.resolve()),
        "trial_result_sha256": file_sha256(result_path),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(trial_dir / "adoption.json", adoption, replace=False)
    complete_preexisting_attempt(run_dir, attempt_id, adoption)
    append_event(run_dir / "events.jsonl", {"event": "completed_fit_adopted", **adoption})
    return adoption
