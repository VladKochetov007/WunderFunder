"""Create source-bound sidecars and resource receipts for TL3 F0 replay."""

from __future__ import annotations

import platform
import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow
from f0_replay import ReplayResult
from f0_source import F0Source
from native_runtime import sha256_file, write_json_create_only


def container_cgroup_receipt() -> dict[str, object]:
    """Capture the current container cgroup path and hard limit files."""
    cgroup_root = Path("/sys/fs/cgroup")
    limits: dict[str, str | None] = {}
    for name in ("cpu.max", "memory.max", "memory.current", "memory.peak", "memory.swap.max"):
        path = cgroup_root / name
        try:
            limits[name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            limits[name] = None
    try:
        path = Path("/proc/self/cgroup").read_text(encoding="utf-8").strip()
    except OSError:
        path = "unavailable"
    return {
        "proc_self_cgroup": path,
        "cgroup_files": limits,
        "requested_parent_slice": "wnn-campaign.slice",
        "systemd_parent_measurement": "captured on host while this container scope runs",
    }


def replay_source_hashes() -> dict[str, str]:
    """Hash the exact Python recipe modules that implemented this replay."""
    packet = Path(__file__).resolve().parent
    modules = (
        "score_tl3_f0.py",
        "f0_source.py",
        "f0_replay.py",
        "f0_receipt.py",
        "native_runtime.py",
    )
    return {name: sha256_file(packet / name) for name in modules}


def write_replay_receipts(
    source: F0Source,
    result: ReplayResult,
    archive_path: Path,
    split_path: Path,
    predictions_path: Path,
    sidecar_path: Path,
    receipt_path: Path,
    projection_rows: int,
    resource_profile: str,
) -> dict[str, Any]:
    """Bind cache bytes to the exact archive, split, producer and runtime."""
    data_before = source.data_identity
    data_after = source.data_path.stat()
    stable_identity = (
        int(data_before["device"]) == data_after.st_dev
        and int(data_before["inode"]) == data_after.st_ino
        and int(data_before["bytes"]) == data_after.st_size
        and int(data_before["mtime_ns"]) == data_after.st_mtime_ns
    )
    if not stable_identity:
        predictions_path.unlink(missing_ok=True)
        raise RuntimeError("valid.parquet identity changed during TL3 replay")

    archive_hash = sha256_file(archive_path)
    split_hash = str(source.split.sha256)
    prediction_hash = sha256_file(predictions_path)
    callback_seconds = result.callback_wall_ns / 1e9
    sidecar: dict[str, object] = {
        "schema": "wnn.tl3-f0-prediction-cache.v1",
        "candidate": "team_tl3_original_20261018",
        "zip_sha256": archive_hash,
        "split_manifest_sha256": split_hash,
        "row_groups": list(source.groups),
        "sequence_ids": [source.sequence_by_group[group] for group in source.groups],
        "prediction_sha256": prediction_hash,
        "precision": "team ZIP embedded dynamic-int8/fp16 ensemble",
        "callback_contract": "only seq_ix, step_in_seq, need_prediction and current state[112]; no targets or score mask are read or passed",
        "ancestry": "unknown; potentially trained on F0/validation, so predictions and any local scores are fitted diagnostics, not clean-CV evidence",
    }
    write_json_create_only(sidecar_path, sidecar)
    timing: dict[str, object] = {
        "elapsed_wall_seconds_including_parquet_and_native_calls": result.elapsed_wall_seconds,
        "elapsed_process_cpu_seconds": result.elapsed_cpu_seconds,
        "callback_only_seconds": callback_seconds,
        "callback_only_wall_us_per_row": callback_seconds / result.rows * 1e6,
        "callback_only_full_39400000_row_projection_seconds": callback_seconds
        / result.rows
        * projection_rows,
        "callback_only_projection_rows": projection_rows,
        "complete_competition_runtime_measured": False,
        "projection_scope": "Measured actual F0 callback rows extrapolated by row count only; not complete competition runtime.",
        "reset_check_wall_seconds": result.reset_check_ns / 1e9,
    }
    report: dict[str, Any] = {
        "schema": "wnn.tl3-native-f0-replay.v1",
        "passed": True,
        "candidate": "team_tl3_original_20261018",
        "archive_path": str(archive_path),
        "archive_sha256": archive_hash,
        "native_library": result.native_relative_path,
        "native_library_sha256": result.native_sha256,
        "native_level": result.native_level,
        "cpu_dispatch_level": result.dispatch_level,
        "quant_backend": result.quant_backend,
        "member_count": result.member_count,
        "raw_callback_input_width": len(source.feature_columns),
        "python": sys.version,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
        "resource_profile": resource_profile,
        "container_cgroup": container_cgroup_receipt(),
        "recipe": {
            "row_groups": "all 468 source-qualified F0 evaluation groups in CampaignSplit order",
            "callback": "exact unchanged team ZIP PredictionModel.predict invoked for every row",
            "warmup": "all 99 warmup rows per sequence processed; returned values required to be None",
            "sequence_reset": "native callback receives the source sequence ID and resets on ID change; no state is carried across groups",
            "reset_check_performed_in_full_replay": result.reset_aba_equal is not None,
            "labels_or_scoring_mask_read": False,
        },
        "replay_source_sha256": replay_source_hashes(),
        "split_path": str(split_path),
        "split_manifest_sha256": split_hash,
        "fold": source.split.fold,
        "evaluation_groups_total": len(source.split.evaluation_groups),
        "evaluation_groups_replayed": len(source.groups),
        "row_group_order": "CampaignSplit.evaluation_groups; each producer row group verified against the manifest sequence ID",
        "sequence_receipts": result.sequence_receipts,
        "rows_total": result.rows,
        "feature_width": len(source.feature_columns),
        "warmup_rows_processed_and_suppressed": result.warmup_rows,
        "required_prediction_rows": result.predicted_rows,
        "all_predictions_finite": True,
        "reset_aba_exact_equal": result.reset_aba_equal,
        "predictions_path": str(predictions_path),
        "predictions_sha256": prediction_hash,
        "prediction_sidecar_path": str(sidecar_path),
        "data_path": str(source.data_path),
        "data_identity": data_before,
        "data_identity_unchanged": stable_identity,
        "data_filesystem": "read-only repository bind; output cache on host filesystem",
        "labels_or_scoring_mask_read": False,
        "ancestry_status": "unknown; package metadata calls parents all-data, but exact source/teacher/initializer membership has not been verified",
        "evaluation_exposure": "F0 previously exposed; all local TL3 scores or complementarity values are potentially fitted diagnostics, not clean-CV claims",
        "timing": timing,
        "memory_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json_create_only(receipt_path, report)
    return report
