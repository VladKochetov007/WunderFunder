#!/usr/bin/env python3
"""Stage the small, hash-pinned R4 fit artifacts needed by export-only R5."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
SOURCE_WORKTREE = Path("/home/vlad/development/wunder/worktrees/luna-runner-20261008")
R4_ROOT = (
    SOURCE_WORKTREE
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    / "cloud_delivery/gpu_loss_pair_r4"
)
OUTPUT = R4_ROOT / "terminal_collect_r1/attempt_001/outputs"
DATASET = "vladkochetov07/wnn-loss-pair-r4-recovery-20261009-r1"
STAGE = HERE / "source_asset_stage"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            hasher.update(block)
    return hasher.hexdigest()


def source_rows() -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = [
        ("out/bootstrap_receipt.json", OUTPUT / "out/bootstrap_receipt.json"),
        ("out/gpu_inventory.json", OUTPUT / "out/gpu_inventory.json"),
        ("boot_marker.json", OUTPUT / "boot_marker.json"),
        ("failure.json", OUTPUT / "failure.json"),
        ("final_status.json", OUTPUT / "final_status.json"),
        ("heartbeat.json", OUTPUT / "heartbeat.json"),
        ("phase.jsonl", OUTPUT / "phase.jsonl"),
        ("root_fit_review.json", REPO / "research/campaign_2026-10-06/bandit_continuation"
         "/diversity_stack_20261009/ROOT_GPU_R4_COMPLETED_FITS_REVIEW_R1.json"),
        ("terminal_operator_receipt.json", R4_ROOT / "GPU_R4_TERMINAL_OPERATOR_RECEIPT_R1.json"),
        ("out/shared/cache_validation_runtime.json", OUTPUT / "out/shared/cache_validation_runtime.json"),
        ("out/shared/cache_reader_profile_runtime.json", OUTPUT / "out/shared/cache_reader_profile_runtime.json"),
        ("out/shared/device_profile_gate.json", OUTPUT / "out/shared/device_profile_gate.json"),
        ("out/shared/fold0_assets_validation.json", OUTPUT / "out/shared/fold0_assets_validation.json"),
        ("out/logs/cache_profile_verify.log", OUTPUT / "out/logs/cache_profile_verify.log"),
        ("out/logs/device_shape_update_gate.log", OUTPUT / "out/logs/device_shape_update_gate.log"),
        ("out/logs/fit_huber_d1.log", OUTPUT / "out/logs/fit_huber_d1.log"),
        ("out/logs/fit_wmse.log", OUTPUT / "out/logs/fit_wmse.log"),
        ("out/logs/predict_wmse.log", OUTPUT / "out/logs/predict_wmse.log"),
    ]
    arm_files = (
        "artifact_collection.json",
        "model/model.pt",
        "model/model_ep1.pt",
        "model/provenance.json",
        "model/history.json",
        "model/checkpoint_timing.jsonl",
        "prefit/initial_state.npz",
        "prefit/normalization.npz",
        "prefit/normalization.json",
        "prefit/prefit_receipt.json",
        "prefit/sample_orders.json",
        "schedule_trace.json",
        "update_trace.jsonl",
    )
    for arm in ("wmse", "huber_d1"):
        for relative in arm_files:
            path = f"out/arms/{arm}/{relative}"
            rows.append((path, OUTPUT / path))
    rows.append(
        (
            "out/arms/wmse/prefit/profile_inputs.npz",
            OUTPUT / "out/arms/wmse/prefit/profile_inputs.npz",
        )
    )
    return rows


def main() -> None:
    if STAGE.exists():
        raise SystemExit(f"staging directory already exists: {STAGE}")
    source = source_rows()
    missing = [str(path) for _, path in source if not path.is_file() or path.is_symlink()]
    if missing:
        raise SystemExit(f"recovery source artifacts are missing or non-regular: {missing}")
    if (OUTPUT / "out/training_contract.json").exists():
        raise SystemExit("R4 unexpectedly has a training contract; preserve its actual state")
    STAGE.mkdir(parents=True)
    files = []
    for relative, path in source:
        sha = digest(path)
        size = path.stat().st_size
        opaque = f"{sha}.wnn"
        shutil.copyfile(path, STAGE / opaque)
        if digest(STAGE / opaque) != sha or (STAGE / opaque).stat().st_size != size:
            raise SystemExit(f"staged payload verification failed: {relative}")
        files.append({"filename": opaque, "path": relative, "sha256": sha, "size": size})
    total_bytes = sum(row["size"] for row in files)
    if total_bytes >= 100_000_000:
        raise SystemExit(f"recovery asset exceeds its 100 MB staging cap: {total_bytes}")
    manifest = {
        "dataset": DATASET,
        "files": files,
        "private": True,
        "version": 1,
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    (STAGE / "assets-manifest.json").write_bytes(manifest_bytes)
    metadata = {
        "id": DATASET,
        "isPrivate": True,
        "licenses": [{"name": "other"}],
        "title": "WNN Loss Pair R4 Recovery 20261009 R1",
    }
    (STAGE / "dataset-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema": "wnn.loss-pair.r4-export-recovery-stage.v1",
        "dataset": DATASET,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "file_count": len(files),
        "payload_bytes": total_bytes,
        "files": files,
    }
    (STAGE / "stage_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: receipt[key] for key in ("dataset", "manifest_sha256", "file_count", "payload_bytes")}, sort_keys=True))


if __name__ == "__main__":
    main()
