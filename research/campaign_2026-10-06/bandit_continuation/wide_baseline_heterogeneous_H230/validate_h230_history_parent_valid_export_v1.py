from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ATTEMPT_ID = "h230-history-parent-valid-fp32-export-v1-20261008"
CONTRACT_SHA256 = "932a87276c08e2b12bdeed54fac5e81e423e777758e83b9300344d2dd748376e"
AUTH_SHA256 = "12074eb3fe64043e18a3a27e9818b64ea591ed9b1d5964049ff26e62bcdc6be7"
MANIFEST_SHA256 = "01aa939ddb95f91b24859ddf324a66188a3254c11691cd39b40ce77865b3b2f2"
PARENT_RUN = "bd26_h230_all_data_parent_b16_ep3_retry3_v1"
PARENT_CHECKPOINT_SHA256 = "478773bd3d0754d261f8f3c9d46a135d56c3596c269d2a856b760caef05b93c7"
EXPECTED_GROUPS = 1873
EXPECTED_SHAPE = (EXPECTED_GROUPS, 20000, 2)
EXPECTED_ROW_GROUP_SHA256 = "008f12e284e0f72ff423fee68f34c37054759bc4a345e9581620c1fe9b2acdf7"


def sha256_file(path: Path) -> str:
    """Hash one file incrementally without materializing its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction(path: Path) -> dict[str, Any]:
    """Stream the NPY payload once while checking its schema and finiteness."""
    digest = hashlib.sha256()
    finite_values = 0
    with path.open("rb") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(source)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError(f"unexpected NPY version: {version}")
        payload_offset = source.tell()
        if tuple(shape) != EXPECTED_SHAPE or fortran_order or dtype != np.dtype("float32"):
            raise ValueError(f"unexpected prediction header: {(shape, fortran_order, dtype)}")
        source.seek(0)
        digest.update(source.read(payload_offset))
        source.seek(payload_offset)
        expected_payload_bytes = int(np.prod(EXPECTED_SHAPE)) * np.dtype("float32").itemsize
        if path.stat().st_size != payload_offset + expected_payload_bytes:
            raise ValueError("prediction file size does not match its NPY header")
        remaining = expected_payload_bytes
        while remaining:
            chunk = source.read(min(8 * 1024 * 1024, remaining))
            if not chunk or len(chunk) % np.dtype("float32").itemsize:
                raise ValueError("truncated or misaligned prediction payload")
            digest.update(chunk)
            values = np.frombuffer(chunk, dtype=dtype)
            if not np.isfinite(values).all():
                raise ValueError("prediction payload contains nonfinite values")
            finite_values += values.size
            remaining -= len(chunk)
        if source.read(1):
            raise ValueError("unexpected trailing bytes after prediction payload")
    return {
        "sha256": digest.hexdigest(),
        "shape": list(EXPECTED_SHAPE),
        "dtype": "float32",
        "fortran_order": False,
        "finite_values_checked": finite_values,
        "payload_bytes": expected_payload_bytes,
        "header_bytes": payload_offset,
    }


def main() -> int:
    """Validate the sealed valid export and write an append-only receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    campaign = Path("research/campaign_2026-10-06/bandit_continuation")
    h230 = campaign / "wide_baseline_heterogeneous_H230"
    attempt_path = root / campaign / "runtime/attempts" / f"{ATTEMPT_ID}.json"
    contract_path = root / h230 / "H230_HISTORY_PARENT_VALID_EXPORT_PHASE_CONTRACT_V1_20261008.json"
    auth_path = root / h230 / "H230_HISTORY_PARENT_VALID_EXPORT_ROOT_AUTHORIZATION_20261008_R1.json"
    manifest_path = (
        root
        / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/H230_HISTORY_SWAP_SOURCE_MANIFEST_V1_20261008.json"
    )
    prediction_path = root / "cache/pred_h230_history_parent_ep3_fp32_v1_valid.npy"
    sidecar_path = root / "cache/pred_h230_history_parent_ep3_fp32_v1_valid.json"

    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    if (
        attempt.get("id") != ATTEMPT_ID
        or attempt.get("status") != "finished"
        or attempt.get("phase") != "terminal"
        or attempt.get("exit_status") != 0
        or attempt.get("failure") is not None
    ):
        raise ValueError("valid export controller attempt is not a successful terminal")
    attempt_sha256 = sha256_file(attempt_path)
    if sha256_file(contract_path) != CONTRACT_SHA256:
        raise ValueError("valid export phase contract changed")
    if sha256_file(auth_path) != AUTH_SHA256:
        raise ValueError("valid export root authorization changed")
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("sealed source manifest changed")

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    groups = sidecar.get("row_groups")
    if (
        not isinstance(groups, list)
        or len(groups) != EXPECTED_GROUPS
        or any(type(group) is not int for group in groups)
        or len(set(groups)) != EXPECTED_GROUPS
    ):
        raise ValueError("sidecar row-group list is not a unique ordered integer sequence")
    group_digest = hashlib.sha256()
    for group in groups:
        group_digest.update(struct.pack("<q", group))
    row_groups_sha256 = group_digest.hexdigest()
    if row_groups_sha256 != EXPECTED_ROW_GROUP_SHA256:
        raise ValueError("valid row-group list differs from frozen fit-membership digest")
    if (
        sidecar.get("run") != PARENT_RUN
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != 3
        or sidecar.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or sidecar.get("inference_precision") != "fp32"
        or sidecar.get("dtype") != "float32"
        or sidecar.get("shape") != list(EXPECTED_SHAPE)
        or sidecar.get("features_only") is not True
        or sidecar.get("row_groups_sha256") != EXPECTED_ROW_GROUP_SHA256
        or sidecar.get("source_trainer_sha256") != source_hashes.get("scripts/train_rnn.py")
        or sidecar.get("source_exporter_sha256")
        != source_hashes.get("scripts/export_all_data_parent.py")
    ):
        raise ValueError("valid sidecar ancestry, split, precision, order, or schema mismatch")

    prediction = validate_prediction(prediction_path)
    if prediction["sha256"] != sidecar.get("prediction_sha256"):
        raise ValueError("streamed prediction hash differs from exporter sidecar")
    sidecar_sha256 = sha256_file(sidecar_path)
    worker_delta = attempt["final_memory_event_deltas"]
    parent_local_delta = attempt["final_parent_memory_event_deltas"]
    parent_hier_delta = attempt["final_parent_hierarchical_memory_event_deltas"]
    if any(
        delta.get(key) != 0
        for delta in (worker_delta, parent_local_delta, parent_hier_delta)
        for key in ("oom", "oom_kill", "oom_group_kill")
    ):
        raise ValueError("current export or parent memory-scope OOM counter delta is nonzero")

    timing = sidecar["timing"]
    receipt = {
        "schema": "h230-history-parent-valid-export-terminal-validation-v1",
        "record_id": "H230_HISTORY_PARENT_VALID_EXPORT_TERMINAL_VALIDATION_V1_20261008",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "passed_valid_feature_export_validation_no_score",
        "attempt": {
            "id": ATTEMPT_ID,
            "path": str(attempt_path.relative_to(root)),
            "sha256": attempt_sha256,
            "exit_status": 0,
            "started_utc": attempt.get("started_utc"),
            "finished_utc": attempt.get("finished_utc"),
            "elapsed_seconds": attempt.get("elapsed_seconds"),
            "worker_peak_ram_bytes": attempt.get("peak_ram_bytes"),
            "parent_peak_ram_bytes": attempt.get("peak_parent_ram_bytes"),
            "peak_observed_vram_bytes": attempt.get("peak_observed_vram_bytes"),
            "worker_memory_event_deltas": worker_delta,
            "parent_local_memory_event_deltas": parent_local_delta,
            "parent_hierarchical_baseline": attempt.get(
                "initial_parent_hierarchical_memory_events"
            ),
            "parent_hierarchical_memory_event_deltas": parent_hier_delta,
        },
        "root_authorization_sha256": AUTH_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "source_manifest_sha256": MANIFEST_SHA256,
        "parent_run": PARENT_RUN,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "prediction_path": str(prediction_path.relative_to(root)),
        "prediction_sha256": prediction["sha256"],
        "sidecar_path": str(sidecar_path.relative_to(root)),
        "sidecar_sha256": sidecar_sha256,
        "schema_and_finiteness": prediction,
        "groups": {
            "count": EXPECTED_GROUPS,
            "row_groups_sha256": row_groups_sha256,
            "sequence_order": "sidecar's exact ordered valid target-index IDs; int64 digest matches frozen parent fit-membership digest",
        },
        "sidecar_timing": timing,
        "first_threshold_interpretation": {
            "sidecar_field": "first_100_sequences_wall_seconds",
            "sidecar_seconds": timing["first_100_sequences_wall_seconds"],
            "actual_sequences_at_threshold": 112,
            "reason": "B16 exporter checks after each complete batch; the seventh batch first reaches at least100.",
            "measured_threshold_sequences_per_second": 112
            / timing["first_100_sequences_wall_seconds"],
        },
        "scope": {
            "features_only": True,
            "targets_or_masks_read": False,
            "fit_performed": False,
            "metric_computed": False,
            "heldout_claim": False,
            "cache_build_and_student_fit": "require separate root authorization and fresh admission",
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "attempt_sha256": attempt_sha256,
                "prediction_sha256": prediction["sha256"],
                "sidecar_sha256": sidecar_sha256,
                "elapsed_seconds": attempt.get("elapsed_seconds"),
                "total_export_wall_seconds": timing["total_export_wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
