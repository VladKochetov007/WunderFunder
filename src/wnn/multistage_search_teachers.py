from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path

from wnn.multistage_search import (
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    read_json,
)
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_runtime import base_environment, current_process_owner
from wnn.multistage_search_successor_prepare import _campaign_inputs
from wnn.training.sequence_config import load_config


def export_teacher(
    run_dir: Path,
    result_path: Path,
    *,
    worker_id: str,
    gpu_key: str,
    device: Mapping[str, object],
    batch_size: int = 64,
) -> dict[str, object]:
    """Export fixed-final teacher targets for auxiliary-fit members only."""
    if batch_size not in {64, 128}:
        raise ValueError("teacher export batch size must be one of the measured 64/128 candidates")
    run_manifest, admission = _campaign_inputs(run_dir)
    result_path = result_path.resolve(strict=True)
    result = read_json(result_path)
    objective = result.get("objective")
    evaluation = objective.get("evaluation") if isinstance(objective, Mapping) else None
    if result.get("status") != "COMPLETE" or result.get("stage") not in {1, 2}:
        raise ValueError("teacher export parent must be a complete stage1/2 fit")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("scope") != "inner_train_proxy"
        or evaluation.get("source") != "train"
        or evaluation.get("is_scored_used") is not False
    ):
        raise ValueError("teacher parent must have a qualified train-only control score")
    fit_dir = Path(str(result["fit_output"])).resolve(strict=True)
    checkpoint = fit_dir / "model.pt"
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != result.get("fit_model_sha256"):
        raise ValueError("teacher fixed-final checkpoint changed after its fit result")
    config_path = result_path.parent / "config.json"
    config_file_sha256 = file_sha256(config_path)
    if config_file_sha256 != result.get("config_sha256"):
        raise ValueError("teacher raw config file changed after its fit result")
    config = load_config(config_path)
    if (
        config.get("precision_mode") != "fp32"
        or config.get("kd_weight") != 0
        or config.get("hint_weight") != 0
        or not isinstance(config.get("model"), Mapping)
        or config["model"].get("hint_dim") != 0
    ):
        raise ValueError("teacher source must be a strict-FP32 label-only fixed-final model")
    import torch

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_config = (
        checkpoint_payload.get("config")
        if isinstance(checkpoint_payload, Mapping)
        else None
    )
    checkpoint_provenance = (
        checkpoint_payload.get("provenance")
        if isinstance(checkpoint_payload, Mapping)
        else None
    )
    checkpoint_identity = (
        checkpoint_provenance.get("identity")
        if isinstance(checkpoint_provenance, Mapping)
        else None
    )
    if (
        checkpoint_config != config
        or not isinstance(checkpoint_identity, Mapping)
        or checkpoint_identity.get("config_sha256") != canonical_json_sha256(config)
    ):
        raise ValueError("teacher checkpoint identity differs from its normalized config")
    evaluation_root = Path(str(run_manifest["evaluation_source_root"])).resolve(strict=True)
    predictor = evaluation_root / "scripts" / "predict_sequence.py"
    from wnn.training.sequence_provenance import source_identity

    source = source_identity(evaluation_root, predictor)
    if source.get("sha256") != run_manifest.get(
        "evaluation_source_sha256"
    ) or canonical_json_sha256(source.get("files")) != run_manifest.get(
        "evaluation_source_files_sha256"
    ):
        raise ValueError("teacher prediction source differs from its pinned evaluation snapshot")
    auxiliary_ref = admission["data"]["auxiliary_manifest"]
    auxiliary = Path(str(auxiliary_ref["path"])).resolve(strict=True)
    if file_sha256(auxiliary) != auxiliary_ref["sha256"]:
        raise ValueError("teacher export auxiliary manifest changed")
    data_root = Path(str(admission["data"]["data_root"])).resolve(strict=True)
    key = f"stage{result['stage']}-trial-{int(result['trial_number']):06d}"
    export_dir = run_dir / "stage3" / "teachers" / key
    export_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = export_dir / "cache"
    command = [
        os.fspath(sys.executable),
        str(predictor),
        "--checkpoint",
        str(checkpoint),
        "--data-root",
        str(data_root),
        "--split-manifest",
        str(auxiliary),
        "--output",
        str(prediction_dir),
        "--membership",
        "fit",
        "--batch-size",
        str(batch_size),
    ]
    deadline = read_json(run_dir / "campaign_deadline.json")
    reserve_start = deadline.get("reserve_start_epoch")
    if not isinstance(reserve_start, (int, float)) or time.time() >= reserve_start:
        raise RuntimeError("teacher export is closed by the collection reserve")
    timeout = min(3600, int(reserve_start - time.time()))
    owner = current_process_owner() | {"worker_id": worker_id, "mode": "teacher_export"}
    from wnn.multistage_search_owned_command import run_owned_command

    receipt = export_dir / "worker_receipt.json"
    code, timed_out, _ = run_owned_command(
        command,
        cwd=evaluation_root,
        env=base_environment(str(device["cuda_visible_devices"]), source_root=evaluation_root),
        log_path=export_dir / "export.log",
        receipt_path=receipt,
        owner=owner,
        stage=3,
        trial_number=int(result["trial_number"]),
        mode="teacher_export",
        gpu_key=gpu_key,
        output_dir=prediction_dir,
        timeout_seconds=timeout,
    )
    if code != 0 or timed_out:
        raise RuntimeError("fitting-only teacher prediction export failed")
    manifest_path = prediction_dir / "manifest.json"
    from wnn.splits import load_campaign_split
    from wnn.training.sequence_cache import TeacherCache
    from wnn.training.sequence_cache_weighted_pool_identity import auxiliary_members

    split = load_campaign_split(auxiliary, data_root)
    fit_members, _ = auxiliary_members(split)
    teacher_cache = TeacherCache(prediction_dir, split, fit_members)
    producer = teacher_cache.metadata["producer"]
    provenance = producer["provenance"]
    record = {
        "schema": "wnn.multistage-teacher-export.v1",
        "status": "complete",
        "trial_id": key,
        "stage": result["stage"],
        "trial_number": result["trial_number"],
        "trial_result_path": str(result_path),
        "trial_result_sha256": file_sha256(result_path),
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": canonical_json_sha256(config),
        "config_file_sha256": config_file_sha256,
        "producer_identity_sha256": canonical_json_sha256(provenance["identity"]),
        "producer_sha256": canonical_json_sha256(producer),
        "provenance_sha256": canonical_json_sha256(provenance),
        "cache_root": str(prediction_dir.resolve()),
        "cache_manifest_sha256": file_sha256(manifest_path),
        "split_sha256": teacher_cache.metadata["split_sha256"],
        "fit_members_count": len(fit_members),
        "batch_size": batch_size,
        "worker_receipt_path": str(receipt.resolve()),
        "worker_receipt_sha256": file_sha256(receipt),
        "completed_at_utc": utc_now(),
    }
    record_path = export_dir / "teacher_export.json"
    atomic_write_json(record_path, record, replace=False)
    return record | {
        "record_path": str(record_path.resolve()),
        "record_sha256": file_sha256(record_path),
    }


def build_teacher_pool(
    run_dir: Path,
    export_paths: Sequence[Path],
    weights: Sequence[float],
    *,
    pool_id: str,
) -> dict[str, object]:
    """Freeze pinned export identities and merge an ordered weighted fit-only cache."""
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", pool_id)
        or len(export_paths) < 2
        or len(export_paths) != len(weights)
    ):
        raise ValueError("teacher pool requires an id and paired exports/weights")
    if (
        any(not isfinite(value) or value <= 0 for value in weights)
        or not isfinite(sum(weights))
        or abs(sum(weights) - 1.0) > 1e-7
    ):
        raise ValueError("teacher pool weights must be positive and sum to one")
    _manifest, admission = _campaign_inputs(run_dir)
    parent_rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for path, weight in zip(export_paths, weights, strict=True):
        export_path = path.resolve(strict=True)
        record = read_json(export_path)
        if record.get("status") != "complete" or record.get("schema") != (
            "wnn.multistage-teacher-export.v1"
        ):
            raise ValueError("teacher pool parent export is incomplete or unsupported")
        trial_id = record.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in seen_ids:
            raise ValueError("teacher pool export parents must have unique trial identities")
        result_path = Path(str(record["trial_result_path"])).resolve(strict=True)
        parent_result = read_json(result_path)
        config_path = result_path.parent / "config.json"
        config_file_sha256 = file_sha256(config_path)
        config = load_config(config_path)
        if (
            file_sha256(result_path) != record.get("trial_result_sha256")
            or parent_result.get("fit_model_sha256") != record.get("checkpoint_sha256")
            or config_file_sha256 != parent_result.get("config_sha256")
            or config_file_sha256 != record.get("config_file_sha256")
            or canonical_json_sha256(config) != record.get("config_sha256")
            or parent_result.get("status") != "COMPLETE"
            or parent_result.get("stage") not in {1, 2}
        ):
            raise ValueError("teacher export parent fit result changed after export")
        seen_ids.add(trial_id)
        parent_rows.append(
            {
                "cache_root": record["cache_root"],
                "trial_id": record["trial_id"],
                "weight": float(weight),
                "cache_manifest_sha256": record["cache_manifest_sha256"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "config_sha256": record["config_sha256"],
                "config_file_sha256": record["config_file_sha256"],
                "producer_identity_sha256": record["producer_identity_sha256"],
                "producer_sha256": record["producer_sha256"],
                "provenance_sha256": record["provenance_sha256"],
            }
        )
    pool_dir = run_dir / "stage3" / "pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    request_path = pool_dir / f"{pool_id}.request.json"
    output = pool_dir / pool_id
    request = {
        "schema": "wnn.multistage-teacher-pool-request.v1",
        "pool_id": pool_id,
        "parents": parent_rows,
    }
    atomic_write_json(request_path, request, replace=False)
    from wnn.splits import load_campaign_split
    from wnn.training.sequence_cache_weighted_pool import (
        DEFAULT_MINIMUM_FREE_BYTES,
        merge_weighted_teacher_caches,
    )
    from wnn.training.sequence_cache_weighted_pool_identity import auxiliary_members

    auxiliary_ref = admission["data"]["auxiliary_manifest"]
    auxiliary = Path(str(auxiliary_ref["path"])).resolve(strict=True)
    data_root = Path(str(admission["data"]["data_root"])).resolve(strict=True)
    split = load_campaign_split(auxiliary, data_root)
    fit_members, _ = auxiliary_members(split)
    import hashlib

    request_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
    cache_path = merge_weighted_teacher_caches(
        [Path(str(row["cache_root"])) for row in parent_rows],
        [{key: value for key, value in row.items() if key != "cache_root"} for row in parent_rows],
        data_root,
        auxiliary,
        output,
        pool_id=pool_id,
        request_sha256=request_sha,
        minimum_free_bytes=DEFAULT_MINIMUM_FREE_BYTES,
    )
    pool_record = {
        "schema": "wnn.multistage-teacher-pool.v1",
        "status": "complete",
        "pool_id": pool_id,
        "request_path": str(request_path.resolve()),
        "request_sha256": request_sha,
        "cache_root": str(cache_path.resolve()),
        "cache_manifest_sha256": file_sha256(cache_path / "manifest.json"),
        "split_sha256": split.sha256,
        "fit_members_count": len(fit_members),
        "parents": parent_rows,
        "completed_at_utc": utc_now(),
    }
    record_path = pool_dir / f"{pool_id}.pool.json"
    atomic_write_json(record_path, pool_record, replace=False)
    return pool_record | {
        "record_path": str(record_path.resolve()),
        "record_sha256": file_sha256(record_path),
    }
