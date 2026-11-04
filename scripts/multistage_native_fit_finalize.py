from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multistage_native_fit_receipts import regular_file
from multistage_native_fit_validation import verify_control_events, verify_native_fit

from wnn.multistage_plot import export_stage_trial_plots
from wnn.multistage_primary_report import AUXILIARY_SPLIT_SHA256, PRIMARY_SPLIT_SHA256
from wnn.multistage_search import atomic_write_json, canonical_json_sha256, file_sha256
from wnn.multistage_search_metrics import validate_inner_objective
from wnn.training.sequence_provenance import source_identity


def _run_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    receipt_path: Path,
    deadline: float,
) -> dict[str, Any]:
    """Run one bounded Source6 postprocessing command and persist its observed receipt."""
    started = time.time()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("postprocessing exceeded its shared wall-clock limit")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    return_code: int | None = None
    try:
        with log_path.open("xb") as log:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=remaining,
                check=False,
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    receipt = {
        "schema": "wnn.multistage-postprocess-command.v1",
        "command": argv,
        "command_sha256": canonical_json_sha256(argv),
        "cwd": str(cwd),
        "started_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "duration_seconds": time.time() - started,
        "return_code": return_code,
        "timed_out": timed_out,
        "log_path": str(log_path),
        "log_sha256": file_sha256(log_path) if log_path.is_file() else None,
        "status": "completed" if return_code == 0 and not timed_out else "failed",
    }
    atomic_write_json(receipt_path, receipt, replace=False)
    if receipt["status"] != "completed":
        raise RuntimeError(f"Source6 postprocess command failed; inspect {receipt_path}")
    return receipt


def finalize_native_multistage_fit(args: Any) -> dict[str, Any]:
    """Verify a completed fit, score its fixed final model on auxiliary control, and publish receipts."""
    if args.stage != 1 or args.trial_number < 0 or args.batch_size < 1:
        raise ValueError("this finalizer accepts fixed stage-one fits and positive trial/batch ids")
    if args.timeout_seconds < 1 or args.timeout_seconds > 600:
        raise ValueError("postprocessing wall limit must be between one and 600 seconds")
    registry = regular_file(args.portable_data_manifest, "portable data registry", 4_000_000)
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(registry)
    os.environ["WNN_PROJECT_ROOT"] = str(args.data_root.resolve(strict=True))
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("finalized output must be a fresh regular directory path")
    output.mkdir(parents=True)
    input_receipt, request, verified = verify_native_fit(args, output)
    fit_dir = args.fit_dir.resolve(strict=True)
    identity = verified["identity"]
    epochs = int(verified["payload"]["completed_epochs"])
    if args.epochs != epochs:
        raise ValueError("requested epoch count differs from the verified native fixed-final fit")
    evaluations = verify_control_events(fit_dir, identity, epochs)
    eval_root = args.evaluation_source_root.resolve(strict=True)
    predictor = eval_root / "scripts/predict_sequence.py"
    scorer = eval_root / "scripts/score_sequence.py"
    predict_source = source_identity(eval_root, predictor)
    score_source = source_identity(eval_root, scorer)
    if (
        predict_source.get("sha256") != args.evaluation_source_sha256
        or canonical_json_sha256(predict_source.get("files")) != args.evaluation_files_sha256
        or score_source.get("files", {}).get("scripts/score_sequence.py")
        != args.evaluation_score_script_sha256
    ):
        raise ValueError("Source6 predictor/scorer differs from the pinned evaluation source")

    prediction_dir = output / "fixed_final_control_predictions"
    objective_path = output / "inner_objective.json"
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    env["WNN_PROJECT_ROOT"] = str(args.data_root.resolve(strict=True))
    env["WNN_PORTABLE_DATA_MANIFEST"] = str(registry)
    eval_pythonpath = str(eval_root / "src")
    if env.get("PYTHONPATH"):
        eval_pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = eval_pythonpath
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    python = str(args.python or sys.executable)
    deadline = time.monotonic() + args.timeout_seconds
    prediction_command = [
        python,
        str(predictor),
        "--checkpoint",
        str(fit_dir / "model.pt"),
        "--data-root",
        str(args.data_root.resolve()),
        "--split-manifest",
        str(args.auxiliary_split_manifest.resolve()),
        "--output",
        str(prediction_dir),
        "--membership",
        "evaluation",
        "--batch-size",
        str(args.batch_size),
    ]
    prediction_receipt = _run_command(
        prediction_command,
        cwd=eval_root,
        env=env,
        log_path=output / "predict.log",
        receipt_path=output / "predict_command_receipt.json",
        deadline=deadline,
    )
    score_command = [
        python,
        str(scorer),
        "--data-root",
        str(args.data_root.resolve()),
        "--split-manifest",
        str(args.auxiliary_split_manifest.resolve()),
        "--predictions",
        str(prediction_dir),
        "--output",
        str(objective_path),
    ]
    score_receipt = _run_command(
        score_command,
        cwd=eval_root,
        env=env,
        log_path=output / "score.log",
        receipt_path=output / "score_command_receipt.json",
        deadline=deadline,
    )
    objective = validate_inner_objective(
        objective_path,
        auxiliary_split_sha256=AUXILIARY_SPLIT_SHA256,
        primary_split_sha256=PRIMARY_SPLIT_SHA256,
        expected_checkpoint_sha256=input_receipt["model_sha256"],
        expected_identity_sha256=canonical_json_sha256(identity),
    )
    final_monitor = evaluations[-1]["metrics"]
    if (
        any(
            abs(
                float(objective["objective"][metric]) - float(final_monitor[f"control/wp/{metric}"])
            )
            > 1e-7
            for metric in ("t0", "t1")
        )
        or abs(float(objective["objective"]["value"]) - float(final_monitor["control/wp"])) > 1e-7
    ):
        raise ValueError("fixed-final Source6 score differs from the terminal control event")
    diagnostics = [
        {
            "epoch": event["epoch"],
            "step": event["step"],
            "wp": event["metrics"]["control/wp"],
            "t0": event["metrics"]["control/wp/t0"],
            "t1": event["metrics"]["control/wp/t1"],
        }
        for event in evaluations
    ]
    plot_receipt = export_stage_trial_plots(
        output / "trial_request.json", fit_dir, args.stage, force=True, plots_dir=output / "plots"
    )
    model_sha = input_receipt["model_sha256"]
    result = {
        "schema": "wnn.multistage-stage1-result.v1",
        "stage": args.stage,
        "trial_number": args.trial_number,
        "family": args.family,
        "params": request["params"],
        "proposal_key": canonical_json_sha256(
            {"config_sha256_raw": request["config_sha256"], "trial_number": args.trial_number}
        ),
        "config_sha256": request["config_sha256"],
        "config_identity_sha256": request["config_identity_sha256"],
        "profile_identity_sha256": request["profile_identity_sha256"],
        "training_source_revision": request["source_revision"],
        "training_source_sha256": request["source_sha256"],
        "evaluation_source_revision": args.evaluation_source_revision,
        "evaluation_source_sha256": predict_source["sha256"],
        "evaluation_score_script_sha256": args.evaluation_score_script_sha256,
        "profile_receipt_sha256": input_receipt["native_profile_sha256"],
        "approved_profile_sha256": input_receipt["approved_profile_sha256"],
        "selected_endpoint": {
            "selection_role": "fixed_final",
            "epoch": epochs,
            "step": int(verified["payload"]["provenance"]["global_updates"]),
            "wp": objective["objective"]["value"],
            "checkpoint_path": str(fit_dir / "model.pt"),
            "checkpoint_sha256": model_sha,
            "epoch_index_sha256": input_receipt["epoch_index_internal_sha256"],
            "epoch_index_file_sha256": input_receipt["epoch_index_file_sha256"],
            "identity_sha256": request["profile_identity_sha256"],
        },
        "objective": objective,
        "best_observed_control_diagnostic": {
            "meaning": "diagnostic only; never used to select/export the fixed-final checkpoint",
            "events": diagnostics,
            "maximum_observed_wp": max(float(item["wp"]) for item in diagnostics),
        },
        "fit_output": str(fit_dir),
        "fit_model_sha256": model_sha,
        "fit_process_receipt_path": input_receipt["completion"]["path"],
        "fit_process_receipt_sha256": input_receipt["completion"]["sha256"],
        "prediction_manifest_sha256": file_sha256(prediction_dir / "manifest.json"),
        "prediction_batch_size": args.batch_size,
        "score_receipt_sha256": file_sha256(objective_path),
        "score_receipts": {
            "prediction_command": prediction_receipt,
            "score_command": score_receipt,
        },
        "plots": plot_receipt,
        "parent_config_sha256": request.get("parent_config_sha256"),
        "status": "COMPLETE",
        "adopted_external_fit": True,
        "exit_observation": input_receipt["completion"]["kind"],
        "process_exit_code_observed": input_receipt["completion"]["exit_code"] == 0,
        "completion_evidence": "native_terminal_model_index_and_observed_or_attested_completion",
        "completed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    result_path = output / "trial_result.json"
    atomic_write_json(result_path, result, replace=False)
    postprocess = {
        "schema": "wnn.multistage-native-fit-postprocess.v1",
        "status": "complete",
        "scientific_fit_performed": False,
        "primary_valid_used": False,
        "inputs": input_receipt,
        "monitor_evaluations_sha256": file_sha256(fit_dir / "monitor_evaluations.jsonl"),
        "monitor_events_sha256": file_sha256(fit_dir / "monitor_events.jsonl"),
        "fixed_final_objective_sha256": file_sha256(objective_path),
        "prediction_manifest_sha256": file_sha256(prediction_dir / "manifest.json"),
        "trial_request_sha256": file_sha256(output / "trial_request.json"),
        "trial_result_sha256": file_sha256(result_path),
        "plots": plot_receipt,
    }
    atomic_write_json(output / "postprocess_receipt.json", postprocess, replace=False)
    return result
