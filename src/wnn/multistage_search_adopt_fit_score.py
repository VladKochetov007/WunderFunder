from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import file_sha256, read_json
from wnn.multistage_search_adopt_fit_artifacts import prediction_batch_size
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import base_environment, current_process_owner, hold_gpu_slot


def score_adopted_fit(
    run_dir: Path,
    run_manifest: Mapping[str, object],
    admission: Mapping[str, object],
    auxiliary_path: Path,
    fit_dir: Path,
    selection: Mapping[str, object],
    request: Mapping[str, object],
    gpu_key: str,
    trial_number: int,
) -> dict[str, object]:
    """Run only pinned selected-control inference and scoring for a completed checkpoint."""
    evaluation_root = Path(str(run_manifest["evaluation_source_root"])).resolve(strict=True)
    predictor = evaluation_root / "scripts/predict_sequence.py"
    scorer = evaluation_root / "scripts/score_sequence.py"
    from wnn.training.sequence_provenance import source_identity

    predict_identity = source_identity(evaluation_root, predictor)
    score_identity = source_identity(evaluation_root, scorer)
    if predict_identity.get("sha256") != run_manifest.get(
        "evaluation_source_sha256"
    ) or score_identity.get("files", {}).get("scripts/score_sequence.py") != run_manifest.get(
        "evaluation_score_script_sha256"
    ):
        raise ValueError("adopted-fit scoring source differs from the frozen evaluator")
    policy = admission["profile_policy"]
    devices = policy["devices"]
    device = next((value for key, value in devices.items() if str(key) == gpu_key), None)
    if not isinstance(device, Mapping):
        raise ValueError("post-fit scoring GPU is outside the measured admitted device set")
    if gpu_key in {str(key) for key in admission.get("reserved_gpu_keys", [])}:
        raise ValueError("post-fit scoring GPU remains reserved by the frozen campaign admission")
    selector = str(device["cuda_visible_devices"])
    data_root = Path(str(admission["data"]["data_root"])).resolve(strict=True)
    prediction_dir = fit_dir / "selected_control_predictions"
    prediction_dir.mkdir(exist_ok=False)
    objective_path = fit_dir / "inner_objective.json"
    config = read_json(Path(str(request["config_path"])))
    batch_size = prediction_batch_size(config)
    owner = current_process_owner() | {
        "mode": "adopted_fit_scoring",
        "attempt_id": request["external_attempt_id"],
    }
    reserve = read_json(run_dir / "campaign_deadline.json").get("reserve_start_epoch")
    if not isinstance(reserve, (int, float)):
        raise ValueError("campaign collection reserve is missing")
    env = base_environment(selector, source_root=evaluation_root)
    with hold_gpu_slot(run_dir, gpu_key):
        prediction_receipt = fit_dir / "prediction_worker_receipt.json"
        code, timed_out, _ = run_owned_command(
            [
                sys.executable,
                str(predictor),
                "--checkpoint",
                str(selection["checkpoint_path"]),
                "--data-root",
                str(data_root),
                "--split-manifest",
                str(auxiliary_path),
                "--output",
                str(prediction_dir),
                "--membership",
                "evaluation",
                "--batch-size",
                str(batch_size),
                "--epoch-index",
                str(selection["epoch_index_path"]),
                "--epoch-index-sha256",
                str(selection["epoch_index_sha256"]),
            ],
            cwd=evaluation_root,
            env=env,
            log_path=fit_dir / "prediction.log",
            receipt_path=prediction_receipt,
            owner=owner,
            stage=1,
            trial_number=trial_number,
            mode="adopted_prediction",
            gpu_key=gpu_key,
            output_dir=prediction_dir,
            timeout_seconds=_reserve_timeout(reserve),
        )
        if code != 0 or timed_out:
            raise RuntimeError("post-fit selected-control prediction export failed")
        score_receipt = fit_dir / "score_worker_receipt.json"
        code, timed_out, _ = run_owned_command(
            [
                sys.executable,
                str(scorer),
                "--data-root",
                str(data_root),
                "--split-manifest",
                str(auxiliary_path),
                "--predictions",
                str(prediction_dir),
                "--output",
                str(objective_path),
            ],
            cwd=evaluation_root,
            env=env,
            log_path=fit_dir / "score.log",
            receipt_path=score_receipt,
            owner=owner,
            stage=1,
            trial_number=trial_number,
            mode="adopted_score",
            gpu_key=gpu_key,
            output_dir=fit_dir,
            timeout_seconds=_reserve_timeout(reserve),
        )
        if code != 0 or timed_out:
            raise RuntimeError("post-fit inner train-proxy scoring failed")
    return {
        "prediction_receipt_path": str(prediction_receipt.resolve()),
        "prediction_receipt_sha256": file_sha256(prediction_receipt),
        "score_receipt_path": str(score_receipt.resolve()),
        "score_receipt_sha256": file_sha256(score_receipt),
    }


def _reserve_timeout(reserve_epoch: float) -> int:
    """Bound post-fit scoring by the immutable collection-reserve epoch."""
    remaining = int(reserve_epoch - time.time())
    if remaining < 2:
        raise RuntimeError("post-fit scoring is closed by the collection reserve")
    return min(900, remaining)
