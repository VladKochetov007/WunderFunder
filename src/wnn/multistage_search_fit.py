from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_plot import export_stage_trial_plots
from wnn.multistage_plot_search import export_stage1_search_plots, export_stage_search_plots
from wnn.multistage_search import (
    append_event,
    atomic_write_json,
    canonical_json_sha256,
    file_lock,
    file_sha256,
    read_json,
)
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_budget_fit import finish_budget_attempt, reserve_fit
from wnn.multistage_search_fit_state import (
    _approved_trial,
    _cache_complete,
    _complete_trial,
    _fail_trial,
)
from wnn.multistage_search_metrics import select_best_control_epoch, validate_inner_objective
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import base_environment, current_process_owner
from wnn.multistage_search_sources import training_source_for_stage
from wnn.multistage_search_study import StudyHandle, open_study


def run_stage1_fit(
    run_dir: Path,
    admission: Mapping[str, object],
    worker_id: str,
    gpu_key: str,
    trial_number: int | None,
    device: Mapping[str, object],
) -> dict[str, object]:
    """Fit one independently reviewed exact-profile trial and score its train-only control."""
    return run_stage_fit(run_dir, admission, worker_id, gpu_key, trial_number, device, stage=1)


def run_stage_fit(
    run_dir: Path,
    admission: Mapping[str, object],
    worker_id: str,
    gpu_key: str,
    trial_number: int | None,
    device: Mapping[str, object],
    *,
    stage: int,
) -> dict[str, object]:
    """Fit one externally reviewed exact-profile trial in stage one, two, or three."""
    if stage not in {1, 2, 3}:
        raise ValueError("fitted Optuna workers support stages one through three")
    run_manifest = read_json(run_dir / "run.json")
    study = open_study(run_dir, worker_id, stage=stage)
    selected = _approved_trial(study, run_dir, trial_number, stage=stage)
    number, trial_dir, request, approval = selected
    claim_path = trial_dir / "fit_claim.lock"
    with file_lock(claim_path, blocking=False):
        return _execute_fit(
            run_dir,
            run_manifest,
            admission,
            study,
            number,
            trial_dir,
            request,
            approval,
            worker_id,
            gpu_key,
            device,
            stage=stage,
        )


def _execute_fit(
    run_dir: Path,
    run_manifest: Mapping[str, object],
    admission: Mapping[str, object],
    study: StudyHandle,
    trial_number: int,
    trial_dir: Path,
    request: Mapping[str, object],
    approval: Mapping[str, object],
    worker_id: str,
    gpu_key: str,
    device: Mapping[str, object],
    *,
    stage: int = 1,
) -> dict[str, object]:
    """Run the exact-profile fit, then select/score one qualified retained epoch."""
    approved_path = Path(str(approval["approved_profile_path"]))
    approved_profile = read_json(approved_path)
    native_path = Path(str(request["profile_receipt_path"]))
    native_profile = read_json(native_path)
    external_review = approved_profile.get("external_review")
    if not isinstance(external_review, Mapping):
        raise ValueError("approved profile copy omits its independent review bindings")
    if (
        approved_profile.get("approved") is not True
        or approved_profile.get("identity") != native_profile.get("identity")
        or external_review.get("native_profile_sha256") != file_sha256(native_path)
        or external_review.get("review_receipt_sha256") != approval.get("review_receipt_sha256")
        or file_sha256(approved_path) != approval.get("approved_profile_sha256")
    ):
        raise ValueError("approved profile copy no longer binds the native exact identity")
    review_path = Path(str(approval["review_receipt_path"]))
    if file_sha256(review_path) != approval.get("review_receipt_sha256"):
        raise ValueError("independent profile review receipt changed after approval")
    anchor_path_value = approval.get("anchor_review_receipt_path")
    if anchor_path_value is not None:
        anchor_path = Path(str(anchor_path_value))
        anchor_sha = file_sha256(anchor_path)
        if anchor_sha != approval.get(
            "anchor_review_receipt_sha256"
        ) or anchor_sha != external_review.get("anchor_review_receipt_sha256"):
            raise ValueError("root anchor review receipt changed after approval")
    elif approval.get("anchor_review_receipt_sha256") is not None:
        raise ValueError("approval record has an unbound anchor receipt hash")
    from wnn.training.sequence_provenance import validate_profile_receipt

    validate_profile_receipt(approved_path, native_profile["identity"])
    config_path = Path(str(request["config_path"]))
    if file_sha256(config_path) != request.get("config_sha256"):
        raise ValueError("trial config changed after its profile")
    config = read_json(config_path)
    monitoring = config.get("monitoring")
    prediction_batch_size = (
        monitoring.get("eval_batch_size", 64) if isinstance(monitoring, Mapping) else 64
    )
    if type(prediction_batch_size) is not int or prediction_batch_size < 1:
        raise ValueError("selected-control prediction batch size is invalid")
    auxiliary = admission["data"]["auxiliary_manifest"]
    auxiliary_path = Path(str(auxiliary["path"])).resolve(strict=True)
    if file_sha256(auxiliary_path) != request.get("auxiliary_split_sha256"):
        raise ValueError("auxiliary split changed after its profile")
    repo_root, source_revision, expected_source_sha, expected_files_sha = (
        training_source_for_stage(run_manifest, stage)
    )
    trainer = repo_root / "scripts" / "train_sequence.py"
    from wnn.training.sequence_provenance import source_identity

    source = source_identity(repo_root, trainer)
    if (
        source.get("sha256") != request.get("source_sha256")
        or source.get("sha256") != expected_source_sha
        or canonical_json_sha256(source.get("files")) != request.get("source_files_sha256")
        or canonical_json_sha256(source.get("files")) != expected_files_sha
    ):
        raise ValueError("pinned training source changed after the exact profile")
    evaluation_root = Path(str(run_manifest["evaluation_source_root"]))
    predictor = evaluation_root / "scripts" / "predict_sequence.py"
    scorer = evaluation_root / "scripts" / "score_sequence.py"
    evaluation_source = source_identity(evaluation_root, predictor)
    score_source = source_identity(evaluation_root, scorer)
    if (
        evaluation_source.get("sha256") != run_manifest.get("evaluation_source_sha256")
        or canonical_json_sha256(evaluation_source.get("files"))
        != run_manifest.get("evaluation_source_files_sha256")
        or score_source.get("files", {}).get("scripts/score_sequence.py")
        != run_manifest.get("evaluation_score_script_sha256")
    ):
        raise ValueError("pinned prediction/scoring source changed after preparation")
    fit_dir = run_dir / f"stage{stage}" / "runs" / f"{trial_number:06d}"
    if fit_dir.exists():
        raise FileExistsError(f"stage{stage} fit output already exists; retries are disabled")
    timeout_limits = run_manifest["limits"]
    stage_policy = admission.get(f"stage{stage}", {})
    if not isinstance(stage_policy, Mapping):
        stage_policy = {}
    score_reserve = int(stage_policy.get("postfit_eval_reserve_seconds", 900))
    deadline = read_json(run_dir / "campaign_deadline.json")
    reserve_start = deadline.get("reserve_start_epoch")
    if not isinstance(reserve_start, (int, float)):
        raise ValueError("campaign deadline has not started")
    remaining_before_reserve = int(reserve_start - time.time())
    fit_budget = min(
        int(timeout_limits["max_fit_wall_seconds"]),
        remaining_before_reserve - score_reserve,
    )
    if fit_budget <= 0:
        raise RuntimeError("fit admission closed to preserve scoring and collection time")
    attempt_id = f"stage{stage}-trial-{trial_number:06d}"
    _, fit_wall_limit = reserve_fit(
        run_dir,
        stage,
        f"stage{stage}-trial-{trial_number:06d}",
        trial_number,
        fit_budget,
    )
    data_root = Path(str(admission["data"]["data_root"])).resolve(strict=True)
    profile_selector = str(device["cuda_visible_devices"])
    env = base_environment(profile_selector, source_root=repo_root)
    fit_command = [
        os.fspath(__import__("sys").executable),
        str(trainer),
        "--config",
        str(config_path),
        "--data-root",
        str(data_root),
        "--split-manifest",
        str(auxiliary_path),
        "--output",
        str(fit_dir),
        "--profile-receipt",
        str(approved_path),
    ]
    initializer = request.get("initializer_path")
    teacher_cache = request.get("teacher_cache_path")
    if isinstance(initializer, str):
        if file_sha256(Path(initializer)) != request.get("initializer_sha256"):
            raise ValueError("stage initializer changed after trial preparation")
        fit_command.extend(["--initializer", initializer])
    if isinstance(teacher_cache, str):
        manifest_path = Path(teacher_cache) / "manifest.json"
        if file_sha256(manifest_path) != request.get("teacher_cache_manifest_sha256"):
            raise ValueError("stage teacher cache changed after trial preparation")
        fit_command.extend(["--teacher-cache", teacher_cache])
    owner = current_process_owner() | {"worker_id": worker_id, "mode": "fit"}
    started = time.time()
    fit_process_receipt = fit_dir.parent / f"{fit_dir.name}.worker_receipt.json"
    last_plot_signature = [0]

    def update_live_plots() -> None:
        """Export after each newly appended real control evaluation event."""
        path = fit_dir / "monitor_evaluations.jsonl"
        if not path.is_file():
            return
        size = path.stat().st_size
        if size <= last_plot_signature[0]:
            return
        last_plot_signature[0] = size
        try:
            export_stage_trial_plots(trial_dir / "trial_request.json", fit_dir, stage)
        except (OSError, ValueError, RuntimeError):
            return

    try:
        code, timed_out, process_record = run_owned_command(
            fit_command,
            cwd=repo_root,
            env=env,
            log_path=fit_dir.parent / f"{fit_dir.name}.fit.log",
            receipt_path=fit_process_receipt,
            owner=owner,
            stage=stage,
            trial_number=trial_number,
            mode=f"stage{stage}_fit",
            gpu_key=gpu_key,
            output_dir=fit_dir,
            timeout_seconds=fit_wall_limit,
            progress=update_live_plots,
        )
        if code != 0 or timed_out:
            raise RuntimeError(f"canonical fit failed with exit={code}, timed_out={timed_out}")
        selection = select_best_control_epoch(fit_dir, str(request["auxiliary_split_sha256"]))
        epoch_index = read_json(Path(str(selection["epoch_index_path"])))
        if epoch_index.get("identity") != native_profile.get("identity"):
            raise ValueError("fit epoch index identity differs from its exact approved profile")
        checkpoint = Path(str(selection["checkpoint_path"]))
        prediction_dir = fit_dir / "selected_control_predictions"
        objective_path = fit_dir / "inner_objective.json"
        predict_script = predictor
        score_script = scorer
        prediction_command = [
            os.fspath(__import__("sys").executable),
            str(predict_script),
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--split-manifest",
            str(auxiliary_path),
            "--output",
            str(prediction_dir),
            "--membership",
            "evaluation",
            "--batch-size",
            str(prediction_batch_size),
            "--epoch-index",
            str(selection["epoch_index_path"]),
            "--epoch-index-sha256",
            str(selection["epoch_index_sha256"]),
        ]
        remaining = int(reserve_start - time.time())
        score_wall = min(int(stage_policy.get("score_wall_seconds", 900)), remaining)
        if score_wall <= 0:
            raise RuntimeError(
                "no campaign time remains for the independent inner objective export"
            )
        prediction_receipt = fit_dir / "prediction_worker_receipt.json"
        code, timed_out, _ = run_owned_command(
            prediction_command,
            cwd=evaluation_root,
            env=base_environment(profile_selector, source_root=evaluation_root),
            log_path=fit_dir / "prediction.log",
            receipt_path=prediction_receipt,
            owner=owner,
            stage=stage,
            trial_number=trial_number,
            mode="prediction",
            gpu_key=gpu_key,
            output_dir=prediction_dir,
            timeout_seconds=score_wall,
        )
        if code != 0 or timed_out:
            raise RuntimeError("selected-control prediction export failed")
        score_command = [
            os.fspath(__import__("sys").executable),
            str(score_script),
            "--data-root",
            str(data_root),
            "--split-manifest",
            str(auxiliary_path),
            "--predictions",
            str(prediction_dir),
            "--output",
            str(objective_path),
        ]
        remaining = int(reserve_start - time.time())
        score_wall = min(score_wall, remaining)
        if score_wall <= 0:
            raise RuntimeError("no campaign time remains for scoring")
        score_receipt = fit_dir / "score_worker_receipt.json"
        code, timed_out, _ = run_owned_command(
            score_command,
            cwd=evaluation_root,
            env=base_environment(profile_selector, source_root=evaluation_root),
            log_path=fit_dir / "score.log",
            receipt_path=score_receipt,
            owner=owner,
            stage=stage,
            trial_number=trial_number,
            mode="score",
            gpu_key=gpu_key,
            output_dir=fit_dir,
            timeout_seconds=score_wall,
        )
        if code != 0 or timed_out:
            raise RuntimeError("inner train-proxy score failed")
        objective = validate_inner_objective(
            objective_path,
            auxiliary_split_sha256=str(request["auxiliary_split_sha256"]),
            primary_split_sha256=str(request["primary_split_sha256"]),
            expected_checkpoint_sha256=str(selection["checkpoint_sha256"]),
            expected_identity_sha256=str(selection["identity_sha256"]),
        )
        if abs(float(objective["objective"]["value"]) - float(selection["wp"])) > 1e-7:
            raise ValueError("standalone pooled score disagrees with the selected monitor event")
        plot_record = export_stage_trial_plots(
            trial_dir / "trial_request.json", fit_dir, stage, force=True
        )
        result = {
            "schema": f"wnn.multistage-stage{stage}-result.v1",
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
            "profile_identity_sha256": request["profile_identity_sha256"],
            "training_source_revision": source_revision,
            "training_source_sha256": request["source_sha256"],
            "evaluation_source_revision": run_manifest["evaluation_source_revision"],
            "evaluation_source_sha256": evaluation_source["sha256"],
            "evaluation_score_script_sha256": run_manifest["evaluation_score_script_sha256"],
            "profile_receipt_sha256": file_sha256(native_path),
            "approved_profile_sha256": file_sha256(approved_path),
            "selected_endpoint": selection,
            "objective": objective,
            "fit_output": str(fit_dir.resolve()),
            "fit_model_sha256": file_sha256(fit_dir / "model.pt"),
            "fit_process_receipt_sha256": file_sha256(fit_process_receipt),
            "prediction_manifest_sha256": file_sha256(prediction_dir / "manifest.json"),
            "prediction_batch_size": prediction_batch_size,
            "score_receipt_sha256": file_sha256(objective_path),
            "plots": plot_record,
            "duration_seconds": time.time() - started,
            "status": "COMPLETE",
            "completed_at_utc": utc_now(),
        }
        result_path = trial_dir / "trial_result.json"
        atomic_write_json(result_path, result, replace=False)
        _complete_trial(
            study,
            run_dir,
            trial_number,
            float(objective["objective"]["value"]),
            result,
            stage=stage,
        )
        _cache_complete(run_dir, str(request["proposal_key"]), result, result_path, trial_number)
        finish_budget_attempt(run_dir, attempt_id, "complete", code)
        if stage == 1:
            export_stage1_search_plots(run_dir)
        else:
            export_stage_search_plots(run_dir, stage)
        if stage == 3:
            try:
                from wnn.multistage_search_stage3_pair import finalize_pair

                finalize_pair(run_dir, request)
            except (OSError, RuntimeError, ValueError) as pair_error:
                append_event(
                    run_dir / "events.jsonl",
                    {
                        "event": "stage3_pair_reconciliation_error",
                        "at_utc": utc_now(),
                        "stage": 3,
                        "trial_number": trial_number,
                        "pair_id": request.get("pair_id"),
                        "error_type": type(pair_error).__name__,
                        "error": str(pair_error),
                    },
                )
        return result
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        finish_budget_attempt(run_dir, attempt_id, "failed", None)
        _fail_trial(study, run_dir, trial_number, error, stage=stage)
        raise
