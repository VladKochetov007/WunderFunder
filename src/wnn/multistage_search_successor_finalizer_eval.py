from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json
from wnn.multistage_search_metrics import validate_inner_objective
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_reviewed_successor_validation import ensure_gpu_slot_released
from wnn.multistage_search_runtime import current_process_owner, hold_gpu_slot
from wnn.multistage_search_successor_finalizer_validation import validate_terminal_event
from wnn.training.sequence_provenance import source_identity


@dataclass(frozen=True)
class FixedFinalEvaluation:
    output_dir: Path
    objective: Mapping[str, object]
    prediction_source: Mapping[str, object]
    score_source: Mapping[str, object]


def evaluate_successor_fixed_final(
    run_dir: Path,
    run_manifest: Mapping[str, object],
    request: Mapping[str, object],
    fit_dir: Path,
    identity: Mapping[str, object],
    payload: Mapping[str, object],
    stage: int,
    trial_number: int,
    batch_size: int,
    timeout_seconds: int,
    python: str | None,
) -> FixedFinalEvaluation:
    """Run pinned Source6 prediction/scoring on the configured-final checkpoint."""
    evaluation_root = Path(str(run_manifest["evaluation_source_root"])).resolve(strict=True)
    predictor = evaluation_root / "scripts" / "predict_sequence.py"
    scorer = evaluation_root / "scripts" / "score_sequence.py"
    prediction_source = source_identity(evaluation_root, predictor)
    score_source = source_identity(evaluation_root, scorer)
    if (
        prediction_source.get("sha256") != run_manifest.get("evaluation_source_sha256")
        or canonical_json_sha256(prediction_source.get("files"))
        != run_manifest.get("evaluation_source_files_sha256")
        or score_source.get("files", {}).get("scripts/score_sequence.py")
        != run_manifest.get("evaluation_score_script_sha256")
    ):
        raise ValueError("fixed-final predictor/scorer differs from pinned Source6")
    runtime_root = run_dir.parent
    portable = runtime_root / "inputs" / "portable-data-r1.json"
    deps = runtime_root / "python-deps"
    if not portable.is_file() or not deps.is_dir():
        raise FileNotFoundError("pinned portable data registry or private dependencies are missing")
    data_root = Path(str(request["data_root"])).resolve(strict=True)
    split_path = Path(str(request["auxiliary_split_manifest"])).resolve(strict=True)
    gpu_key = request.get("gpu_key")
    selectors = request.get("requested_device_set")
    if not isinstance(gpu_key, str) or not isinstance(selectors, list) or len(selectors) != 1:
        raise ValueError("successor postprocess requires the exact single-GPU request")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(selectors[0]),
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "PYTHONPATH": os.pathsep.join([str(evaluation_root / "src"), str(deps)]),
            "WNN_PORTABLE_DATA_MANIFEST": str(portable),
            "WNN_PROJECT_ROOT": str(data_root),
        }
    )
    owner = current_process_owner() | {"mode": "fixed_final_postprocess"}
    reserve_epoch = read_json(run_dir / "campaign_deadline.json").get("reserve_start_epoch")
    if not isinstance(reserve_epoch, (int, float)):
        raise ValueError("successor postprocess requires the frozen collection-reserve deadline")
    remaining = min(timeout_seconds, int(reserve_epoch - time.time()))
    if remaining <= 0:
        raise RuntimeError(
            "campaign collection reserve has started; postprocess admission is closed"
        )
    deadline = time.monotonic() + remaining
    executable = python or sys.executable
    output = fit_dir / "fixed_final_postprocess_r2"
    with hold_gpu_slot(run_dir, gpu_key):
        ensure_gpu_slot_released(run_dir, gpu_key)
        output.mkdir(parents=True, exist_ok=False)
        prediction_dir = output / "control_predictions"
        objective_path = output / "inner_objective.json"
        predict_command = [
            executable,
            str(predictor),
            "--checkpoint",
            str(fit_dir / "model.pt"),
            "--data-root",
            str(data_root),
            "--split-manifest",
            str(split_path),
            "--output",
            str(prediction_dir),
            "--membership",
            "evaluation",
            "--batch-size",
            str(batch_size),
        ]
        _run_postprocess_command(
            predict_command,
            evaluation_root,
            env,
            owner,
            output / "predict.log",
            output / "predict_worker_receipt.json",
            output,
            deadline,
            stage,
            trial_number,
            gpu_key,
            "fixed_final_prediction",
        )
        score_command = [
            executable,
            str(scorer),
            "--data-root",
            str(data_root),
            "--split-manifest",
            str(split_path),
            "--predictions",
            str(prediction_dir),
            "--output",
            str(objective_path),
        ]
        _run_postprocess_command(
            score_command,
            evaluation_root,
            env,
            owner,
            output / "score.log",
            output / "score_worker_receipt.json",
            output,
            deadline,
            stage,
            trial_number,
            gpu_key,
            "fixed_final_score",
        )
    objective = validate_inner_objective(
        objective_path,
        auxiliary_split_sha256=str(request["auxiliary_split_sha256"]),
        primary_split_sha256=str(request["primary_split_sha256"]),
        expected_checkpoint_sha256=file_sha256(fit_dir / "model.pt"),
        expected_identity_sha256=canonical_json_sha256(identity),
    )
    validate_terminal_event(fit_dir, payload, objective, identity)
    return FixedFinalEvaluation(output, objective, prediction_source, score_source)


def _run_postprocess_command(
    command: list[str],
    cwd: Path,
    env: Mapping[str, str],
    owner: Mapping[str, object],
    log_path: Path,
    receipt_path: Path,
    output: Path,
    deadline: float,
    stage: int,
    trial_number: int,
    gpu_key: str,
    mode: str,
) -> None:
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("fixed-final postprocessing reached its shared deadline")
    code, timed_out, _ = run_owned_command(
        command,
        cwd=cwd,
        env=env,
        log_path=log_path,
        receipt_path=receipt_path,
        owner=owner,
        stage=stage,
        trial_number=trial_number,
        mode=mode,
        gpu_key=gpu_key,
        output_dir=output,
        timeout_seconds=remaining,
    )
    if code != 0 or timed_out:
        raise RuntimeError(f"Source6 {mode} failed with exit={code}, timed_out={timed_out}")
