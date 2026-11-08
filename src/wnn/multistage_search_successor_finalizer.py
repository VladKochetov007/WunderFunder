from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_plot import export_stage_trial_plots
from wnn.multistage_plot_search import export_stage_search_plots
from wnn.multistage_search import (
    append_event,
    atomic_write_json,
    canonical_json_sha256,
    file_lock,
    file_sha256,
    read_json,
)
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_fit_state import _cache_complete, _complete_trial
from wnn.multistage_search_sources import training_source_for_stage
from wnn.multistage_search_study import open_study
from wnn.multistage_search_successor_finalizer_eval import evaluate_successor_fixed_final
from wnn.multistage_search_successor_finalizer_result import (
    build_successor_result,
    postprocess_receipt,
)
from wnn.multistage_search_successor_finalizer_validation import (
    validate_successor_fit,
)
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import source_identity


def finalize_successor_fit(
    run_dir: Path,
    stage: int,
    trial_number: int,
    *,
    batch_size: int = 64,
    timeout_seconds: int = 600,
    python: str | None = None,
) -> dict[str, object]:
    """Score and record a completed Stage2/3 fit using its configured-final checkpoint."""
    if stage not in {2, 3} or trial_number < 0 or batch_size < 1:
        raise ValueError("successor finalization requires stage2/3 and positive ids/batches")
    if not 1 <= timeout_seconds <= 900:
        raise ValueError("postprocess timeout must be between one and 900 seconds")
    run_dir = run_dir.resolve(strict=True)
    trial_dir = run_dir / f"stage{stage}" / "trials" / f"{trial_number:06d}"
    result_path = trial_dir / "trial_result.json"
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError("successor trial already has a terminal result")
    with file_lock(trial_dir / "fixed_final_postprocess.lock", blocking=False):
        return _finalize_locked(
            run_dir,
            trial_dir,
            stage,
            trial_number,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
            python=python,
        )


def _finalize_locked(
    run_dir: Path,
    trial_dir: Path,
    stage: int,
    trial_number: int,
    *,
    batch_size: int,
    timeout_seconds: int,
    python: str | None,
) -> dict[str, object]:
    request_path = trial_dir / "trial_request.json"
    request = read_json(request_path)
    definition = read_json(trial_dir / "trial_definition.json")
    run_manifest = read_json(run_dir / "run.json")
    admission = read_json(Path(str(run_manifest["admission"]["path"])))
    study = open_study(run_dir, "successor-fixed-finalizer", stage=stage)
    optuna_trial = next(
        (row for row in study.get_trials(deepcopy=False) if row.number == trial_number), None
    )
    if optuna_trial is None:
        raise ValueError("successor finalizer requires its exact Optuna trial record")
    optuna_state_before = getattr(getattr(optuna_trial, "state", None), "name", None)
    if optuna_state_before not in {"RUNNING", "FAIL"}:
        raise ValueError("successor finalizer accepts only running or legacy failed trials")
    optuna_snapshot_before = _optuna_trial_snapshot(optuna_trial)
    attrs = getattr(optuna_trial, "user_attrs", {})
    stage_attrs = attrs.get(f"stage{stage}") if isinstance(attrs, Mapping) else None
    if not isinstance(stage_attrs, Mapping) or Path(
        str(stage_attrs.get("trial_dir"))
    ).resolve() != (trial_dir.resolve()):
        raise ValueError("successor Optuna trial does not own this prepared trial directory")
    fit_dir = run_dir / f"stage{stage}" / "runs" / f"{trial_number:06d}"
    if (
        request.get("stage") != stage
        or request.get("trial_number") != trial_number
        or request.get("primary_valid_used") is not False
        or request.get("objective_scope") != "inner_train_proxy"
    ):
        raise ValueError("successor request does not bind this train-only trial")
    data_policy = admission["data"]
    data_root = Path(str(data_policy["data_root"])).resolve(strict=True)
    split_path = Path(str(request["auxiliary_split_manifest"])).resolve(strict=True)
    if (
        split_path != Path(str(data_policy["auxiliary_manifest"]["path"])).resolve(strict=True)
        or file_sha256(split_path) != request.get("auxiliary_split_sha256")
        or request.get("primary_split_sha256") != data_policy.get("primary_split_sha256")
        or Path(str(request["data_root"])).resolve(strict=True) != data_root
    ):
        raise ValueError("successor split/data identity differs from frozen admission")
    config_path = Path(str(request["config_path"])).resolve(strict=True)
    if file_sha256(config_path) != request.get("config_sha256"):
        raise ValueError("successor config bytes changed after exact profiling")
    config = load_config(config_path)
    if canonical_json_sha256(config) != request.get("config_identity_sha256"):
        raise ValueError("successor normalized config differs from exact profile identity")
    source_root, _source_revision, expected_source_sha, expected_files_sha = (
        training_source_for_stage(run_manifest, stage)
    )
    source = source_identity(source_root, source_root / "scripts" / "train_sequence.py")
    if (
        (
            request.get("training_source_root") is not None
            and Path(str(request["training_source_root"])).resolve(strict=True) != source_root
        )
        or request.get("source_revision") != _source_revision
        or source.get("sha256") != request.get("source_sha256")
        or source.get("sha256") != expected_source_sha
        or canonical_json_sha256(source.get("files")) != request.get("source_files_sha256")
        or canonical_json_sha256(source.get("files")) != expected_files_sha
    ):
        raise ValueError("successor fit source differs from its frozen Source5B identity")
    verification = validate_successor_fit(
        run_dir, trial_dir, fit_dir, request, config, split_path, data_root, stage, trial_number
    )
    if optuna_state_before == "FAIL" and verification["review"]["revision"] != "R6":
        raise ValueError("legacy failed Optuna trials require the exact R6 independent re-audit")
    identity, payload = verification["identity"], verification["payload"]
    fixed_final = evaluate_successor_fixed_final(
        run_dir,
        run_manifest,
        request,
        fit_dir,
        identity,
        payload,
        stage,
        trial_number,
        batch_size,
        timeout_seconds,
        python,
    )
    output = fixed_final.output_dir
    objective = fixed_final.objective
    plot_request = dict(request)
    if stage == 2:
        plot_request["reused_skip_control"] = definition.get("reused_skip_control")
    plot_request_path = output / "plot_trial_request.json"
    atomic_write_json(plot_request_path, plot_request, replace=False)
    plots = export_stage_trial_plots(
        plot_request_path, fit_dir, stage, force=True, plots_dir=fit_dir / "plots"
    )
    result_request = dict(request)
    if stage == 2:
        result_request["reused_skip_control"] = definition.get("reused_skip_control")
    result = build_successor_result(
        run_manifest,
        result_request,
        fit_dir,
        output,
        objective,
        payload,
        identity,
        plots,
        verification,
        fixed_final.prediction_source,
        fixed_final.score_source,
        batch_size,
        stage,
        trial_number,
    )
    legacy_after_snapshot: str | None = None
    if optuna_state_before == "FAIL":
        current_trial = next(
            row for row in study.get_trials(deepcopy=False) if row.number == trial_number
        )
        legacy_after_snapshot = _optuna_trial_snapshot(current_trial)
        if (
            getattr(getattr(current_trial, "state", None), "name", None) != "FAIL"
            or legacy_after_snapshot != optuna_snapshot_before
        ):
            raise ValueError("legacy Optuna FAIL trial changed during fixed-final postprocessing")
        result["legacy_engineering_optuna_state"] = {
            "before": "FAIL",
            "after": "FAIL",
            "optuna_value_changed": False,
            "reason": "R2 profile-review rejection; later R6 independent engineering re-audit approved exact profile",
            "trial_snapshot_sha256_before": optuna_snapshot_before,
            "trial_snapshot_sha256_after": legacy_after_snapshot,
        }
    result_path = trial_dir / "trial_result.json"
    atomic_write_json(result_path, result, replace=False)
    atomic_write_json(
        output / "postprocess_receipt.json",
        postprocess_receipt(
            result_path,
            fit_dir,
            output / "inner_objective.json",
            plots,
            stage,
            trial_number,
        ),
        replace=False,
    )
    value = float(objective["objective"]["value"])
    if optuna_state_before == "RUNNING":
        _complete_trial(study, run_dir, trial_number, value, result, stage=stage)
    else:
        current_trial = next(
            row for row in study.get_trials(deepcopy=False) if row.number == trial_number
        )
        after_snapshot = _optuna_trial_snapshot(current_trial)
        if (
            getattr(getattr(current_trial, "state", None), "name", None) != "FAIL"
            or after_snapshot != optuna_snapshot_before
            or after_snapshot != legacy_after_snapshot
        ):
            raise ValueError("legacy Optuna FAIL trial changed during fixed-final postprocessing")
        append_event(
            run_dir / "events.jsonl",
            {
                "event": "trial_scientific_fit_result_recorded_after_review_reaudit",
                "at_utc": utc_now(),
                "stage": stage,
                "trial_number": trial_number,
                "status": "COMPLETE_SCIENTIFIC_FIT_OPTUNA_FAIL",
                "objective_value": value,
                "optuna_state_before": "FAIL",
                "optuna_state_after": "FAIL",
                "trial_snapshot_sha256": after_snapshot,
                "result_path": str(result_path.resolve()),
            },
        )
    _cache_complete(run_dir, str(request["proposal_key"]), result, result_path, trial_number)
    export_stage_search_plots(run_dir, stage)
    if stage == 3:
        from wnn.multistage_search_stage3_pair import finalize_pair

        try:
            finalize_pair(run_dir, request)
        except (OSError, RuntimeError, ValueError) as error:
            append_event(
                run_dir / "events.jsonl",
                {
                    "event": "stage3_pair_reconciliation_error",
                    "at_utc": utc_now(),
                    "stage": 3,
                    "trial_number": trial_number,
                    "pair_id": request.get("pair_id"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
    return result


def _optuna_trial_snapshot(trial: object) -> str:
    """Hash the immutable, JSON-safe state relevant to an existing Optuna trial."""
    state = getattr(getattr(trial, "state", None), "name", None)
    value = getattr(trial, "value", None)
    values = getattr(trial, "values", None)
    snapshot = {
        "number": getattr(trial, "number", None),
        "state": state,
        "value": value,
        "values": values,
        "params": getattr(trial, "params", {}),
        "user_attrs": getattr(trial, "user_attrs", {}),
        "system_attrs": getattr(trial, "system_attrs", {}),
        "intermediate_values": getattr(trial, "intermediate_values", {}),
    }
    return canonical_json_sha256(snapshot)
