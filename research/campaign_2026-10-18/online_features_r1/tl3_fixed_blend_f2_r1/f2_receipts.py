"""Write source-bound receipts for exact F2 package replays."""

from __future__ import annotations

import platform
import resource
import sys
from pathlib import Path

import numpy as np
import pyarrow
from f2_candidates import CANDIDATES
from f2_replay_engine import ReplayResult
from f2_source import F2Source
from native_runtime import sha256_file, write_json_create_only

PACKET = Path(__file__).resolve().parent
EXPECTED_ZIPS = CANDIDATES


def _cgroup_receipt() -> dict[str, str | None]:
    """Read current cgroup limits and usage without assuming host RSS is total."""
    return {
        name: _read_text(Path("/sys/fs/cgroup") / name)
        for name in ("cpu.max", "memory.current", "memory.max", "memory.peak", "memory.swap.max")
    }


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def write_receipts(
    source: F2Source,
    result: ReplayResult,
    candidate: str,
    archive_path: Path,
    split_path: Path,
    predictions_path: Path,
    sidecar_path: Path,
    receipt_path: Path,
    resource_profile: str,
) -> dict[str, object]:
    """Bind predictions and execution to frozen model, fold, data and source hashes."""
    archive_hash = sha256_file(archive_path)
    prediction_hash = sha256_file(predictions_path)
    split_hash = str(source.split.sha256)
    data_before = source.data_identity
    data_after = source.data_path.stat()
    stable = all(
        int(data_before[key]) == current
        for key, current in (
            ("bytes", data_after.st_size),
            ("device", data_after.st_dev),
            ("inode", data_after.st_ino),
            ("mtime_ns", data_after.st_mtime_ns),
        )
    )
    if archive_hash != EXPECTED_ZIPS[candidate]["zip_sha256"] or not stable:
        predictions_path.unlink(missing_ok=True)
        raise RuntimeError("a frozen model or source data identity changed during replay")
    sidecar = {
        "schema": "wnn.exact-zip-f2-predictions.v1",
        "candidate": candidate,
        "zip_sha256": archive_hash,
        "split_manifest_sha256": split_hash,
        "fold": 2,
        "row_groups": list(source.groups),
        "sequence_ids": [source.sequence_by_group[group] for group in source.groups],
        "prediction_sha256": prediction_hash,
        "precision": result.precision,
        "ancestry": EXPECTED_ZIPS[candidate]["ancestry"],
        "labels_or_scoring_mask_read": False,
    }
    write_json_create_only(sidecar_path, sidecar)
    callback_seconds = result.callback_wall_ns / 1e9
    source_hashes = {
        name: sha256_file(PACKET / name)
        for name in (
            "f2_candidates.py",
            "f2_source.py",
            "f2_native_runtime.py",
            "f2_replay_engine.py",
            "f2_receipts.py",
            "replay_f2.py",
        )
    }
    source_hashes["tl3_replay/f0_replay.py"] = sha256_file(
        PACKET.parent / "tl3_replay/f0_replay.py"
    )
    source_hashes["tl3_replay/native_runtime.py"] = sha256_file(
        PACKET.parent / "tl3_replay/native_runtime.py"
    )
    report: dict[str, object] = {
        "schema": "wnn.exact-zip-fold2-replay.v1",
        "candidate": candidate,
        "passed": result.reset_aba_equal is not False and stable,
        "archive_path": str(archive_path),
        "archive_sha256": archive_hash,
        "ancestry": EXPECTED_ZIPS[candidate]["ancestry"],
        "native_library": result.native_relative_path,
        "native_library_sha256": result.native_sha256,
        "member_count": result.member_count,
        "precision": result.precision,
        "python": sys.version,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
        "resource_profile": resource_profile,
        "container_cgroup": _cgroup_receipt(),
        "fold": 2,
        "split_path": str(split_path),
        "split_manifest_sha256": split_hash,
        "evaluation_groups_total": len(source.split.evaluation_groups),
        "evaluation_groups_replayed": len(source.groups),
        "sequence_receipts": result.sequences,
        "rows_total": result.rows,
        "feature_width": len(source.feature_columns),
        "warmup_rows_processed_and_suppressed": result.warmup_rows,
        "required_prediction_rows": result.predicted_rows,
        "reset_aba_equal": result.reset_aba_equal,
        "labels_or_scoring_mask_read": False,
        "all_predictions_finite": True,
        "predictions_path": str(predictions_path),
        "predictions_sha256": prediction_hash,
        "prediction_sidecar_path": str(sidecar_path),
        "data_path": str(source.data_path),
        "data_identity": data_before,
        "data_identity_unchanged": stable,
        "source_sha256": source_hashes,
        "timing": {
            "elapsed_wall_seconds": result.elapsed_wall_seconds,
            "elapsed_process_cpu_seconds": result.elapsed_cpu_seconds,
            "callback_only_seconds": callback_seconds,
            "callback_only_wall_us_per_row": callback_seconds / result.rows * 1e6,
            "reset_check_wall_seconds": result.reset_check_ns / 1e9,
        },
        "memory_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "exposure": "F2 has been inspected before; TL3 exact ancestry is unknown, so results are cross-fold robustness diagnostics only",
    }
    write_json_create_only(receipt_path, report)
    return report
