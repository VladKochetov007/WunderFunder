"""Packet identity verification and bounded actual-shape profile admission."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from journal import Journal, write_json_fsync
from policy import (
    ALLOCATOR_BYTES,
    DISK_FREE_MIN,
    PROFILE_TIMEOUT_SECONDS,
    sha256_file,
    validate_profile_log,
)
from runtime import run_single
from runtime_gates import cgroup_snapshot, parse_process_envelope, require_gpus

FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "CUDA error:",
)


def verify_packet(packet: Path, wave: dict[str, Any]) -> dict[str, str]:
    """Check every frozen packet file hash before interpreting or executing it."""
    verified = {}
    for relative, expected in wave["file_hashes"].items():
        path = (packet / relative).resolve(strict=True)
        if not path.is_relative_to(packet.resolve(strict=True)):
            raise ValueError(f"wave packet path escapes its root: {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"frozen packet file hash mismatch: {relative}")
        verified[relative] = digest
    return verified


def actor_environment(root: Path, run_dir: Path, uuid: str) -> dict[str, str]:
    """Build the per-process thread, temp, GPU, and actor workspace envelope."""
    temporary = root / "tmp" / run_dir.name
    temporary.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": uuid,
        "WNN_RUN_DIR": str(run_dir),
        "WNN_GPU_MEMORY_BYTES": str(ALLOCATOR_BYTES),
        "WNN_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "NVIDIA_TF32_OVERRIDE": "0",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(temporary),
    }


def _validate_profile(
    job: dict[str, Any], result: dict[str, Any], run_dir: Path, log: Path
) -> dict[str, Any]:
    """Check profile process, producer, controller-log and exact-shape evidence."""
    if result["status"] != "completed" or result["returncode"] != 0:
        raise RuntimeError(f"{job['arm']} profile process failed: {result}")
    text = log.read_text(encoding="utf-8", errors="replace")
    if any(marker in text for marker in FAILURE_MARKERS):
        raise RuntimeError(f"{job['arm']} profile log contains a CUDA/OOM/runtime failure marker")
    timings, evidence = validate_profile_log(text)
    producer = json.loads((run_dir / "producer_receipt.json").read_text(encoding="utf-8"))
    if (
        producer.get("schema") != "tabdata-b300-profile-producer.v1"
        or producer.get("status") != "profile_completed_check_controller_log"
        or producer.get("arm") != job["arm"]
        or producer.get("fold") != job["fold"]
        or producer.get("seed") != job["seed"]
        or producer.get("expected_log_records") != ["PROFILE", "PROFILE_UPDATE_EVIDENCE"]
    ):
        raise RuntimeError("profile producer receipt does not bind the admitted shape")
    resource = parse_process_envelope(
        run_dir / "train_cell_resource_receipt.json", job["gpu_uuid"], job["cpu_ids_initial"]
    )
    elapsed = producer.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
    ):
        raise RuntimeError("profile producer elapsed time is missing or nonfinite")
    if result["peak_rss_bytes"] <= 0:
        raise RuntimeError("profile process RSS measurement is missing")
    return {
        "schema": "tabdata-b300-shape-profile-gate.v1",
        "status": "passed",
        "job_id": job["id"],
        "arm": job["arm"],
        "fold": job["fold"],
        "seed": job["seed"],
        "gpu_uuid": job["gpu_uuid"],
        "cpu_ids": job["cpu_ids_initial"],
        "timings": timings,
        "update_evidence": evidence,
        "process_result": result,
        "producer_receipt_sha256": sha256_file(run_dir / "producer_receipt.json"),
        "resource_receipt": resource,
        "log_sha256": sha256_file(log),
    }


def run_profile(
    *,
    job: dict[str, Any],
    root: Path,
    state: Path,
    journal: Journal,
    wave_deadline: float,
) -> tuple[dict[str, Any], int]:
    """Run one fresh actual-shape profile with exact L4, affinity, and timeout gates."""
    require_gpus([job], job["gpu_uuid"])
    run_dir = state / "profiles" / job["arm"]
    log = state / "logs" / f"profile-{job['arm']}.log"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    argv = list(job["argv"])
    if argv[-1] != "fit":
        raise ValueError(f"wave command for {job['id']} is not a fit invocation")
    argv[-1] = "profile"
    env = actor_environment(root, run_dir, job["gpu_uuid"])
    before = cgroup_snapshot()
    result = run_single(
        name=f"profile-{job['arm']}",
        argv=argv,
        environment={**env, "WNN_CWD": str(root)},
        cpus=tuple(job["cpu_ids_initial"]),
        log_path=log,
        timeout_seconds=PROFILE_TIMEOUT_SECONDS,
        journal=journal,
        root=root,
        global_deadline=wave_deadline,
        owned_paths=[state],
    )
    gated = _validate_profile(job, result, run_dir, log)
    gated["cgroup_before"] = before
    gated["cgroup_after"] = cgroup_snapshot()
    gated["disk_free_bytes"] = shutil.disk_usage(root).free
    if gated["disk_free_bytes"] < DISK_FREE_MIN:
        raise RuntimeError("disk free-space floor fell below 8 GiB after profile")
    journal.record(
        "profile_pass",
        arm=job["arm"],
        receipt=str(state / "receipts" / f"profile-{job['arm']}.json"),
    )
    write_json_fsync(state / "receipts" / f"profile-{job['arm']}.json", gated)
    return gated, int(result["peak_rss_bytes"])
