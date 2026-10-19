"""Export recovered R01 endpoints and grade an original wide/deep pair."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from r01_recovery_common import read_json, sha256_file, write_json_new
from r01_recovery_fit import EXPECTED_PANEL_SHA256

GRADER_SHA256 = "56ffc693da52668726bab1f7008e31753a76aed3ff5d79089167785079d9eb31"
CACHE_FILES = ("chains.npz", "chain_folds.npz", "train_targets.npz", "valid_targets.npz")


def _verified_receipt(
    recovery_root: Path, packet_root: Path, job_id: str
) -> tuple[Path, dict[str, Any]]:
    """Load a successful fit-recovery receipt and verify its separate fit view."""
    panel_path = packet_root / "PANEL.json"
    if sha256_file(panel_path) != EXPECTED_PANEL_SHA256:
        raise ValueError("R01 panel differs from the frozen source panel")
    panel = read_json(panel_path)
    matches = [job for job in panel.get("jobs", []) if job.get("id") == job_id]
    if len(matches) != 1:
        raise ValueError(f"job is not present exactly once in the frozen R01 panel: {job_id}")
    if recovery_root.is_symlink():
        raise ValueError("recovery root cannot be a symlink")
    recovery_root = recovery_root.resolve(strict=True)
    path = recovery_root / "receipts" / f"{job_id}.json"
    if path.is_symlink():
        raise ValueError("fit recovery receipt cannot be a symlink")
    receipt = read_json(path)
    view_path = Path(receipt.get("fit_view", ""))
    if view_path.is_symlink():
        raise ValueError("recovery fit view cannot itself be a symlink")
    view = view_path.resolve(strict=True)
    if (
        receipt.get("schema") != "tabdata-r01-fit-recovery.v2"
        or receipt.get("status") != "existing_fit_validated_no_refit"
        or receipt.get("job", {}).get("id") != job_id
        or receipt.get("job") != matches[0]
        or view != recovery_root.resolve(strict=True) / "fit_views" / job_id
        or not (view / "producer_receipt.json").is_file()
        or (view / "producer_receipt.json").is_symlink()
        or sha256_file(view / "producer_receipt.json") != receipt.get("recovery_fit_receipt_sha256")
    ):
        raise ValueError(f"fit recovery receipt/view is incomplete or changed: {job_id}")
    return path, receipt


def export_existing_fit(
    project_root: Path,
    packet_root: Path,
    source_root: Path,
    recovery_root: Path,
    job_id: str,
) -> dict[str, Any]:
    """Run only the pinned R01 epoch-5 predictor for one previously validated fit."""
    root = project_root.resolve(strict=True)
    source = source_root.resolve(strict=True)
    _, recovery = _verified_receipt(recovery_root, packet_root, job_id)
    job = recovery["job"]
    code = root / "runtime/b300-r01-v1/code"
    grader = root / "runtime/b300-r01-v1/grader/grade_endpoint.py"
    entrypoint = code / "gpu_entrypoint.py"
    if sha256_file(grader) != GRADER_SHA256:
        raise ValueError("official R01 grader differs from the frozen SHA-256")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices.startswith("GPU-") or "," in visible_devices:
        raise ValueError("R01 export requires one externally assigned B300 UUID")
    output = recovery_root / "exports" / job_id
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"recovery export attempt already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        str(entrypoint),
        str(code / "export_cell.py"),
        "--source-root",
        str(source),
        "--run-dir",
        recovery["fit_view"],
        "--export-dir",
        str(output),
        "--split-manifest",
        str(root / "splits" / f"fold{job['fold']}.json"),
        "--run-name",
        job_id,
        "--fold",
        str(job["fold"]),
        "--seed",
        str(job["seed"]),
        "--grader-script",
        str(grader),
    ]
    actor_log = recovery_root / "logs" / f"{job_id}.export.log"
    actor_log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=visible_devices,
        WNN_RUN_DIR=str(output),
        PYTHONDONTWRITEBYTECODE="1",
    )
    with actor_log.open("x", encoding="utf-8") as log:
        process = subprocess.run(
            argv,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=3600,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"frozen R01 export actor exited {process.returncode}; output/logs retained"
        )
    command_receipt = read_json(output / "export_receipt.json")
    if (
        command_receipt.get("status") != "completed_fixed_epoch5_feature_only_fp32"
        or command_receipt.get("run_name") != job_id
        or command_receipt.get("fold") != job["fold"]
        or command_receipt.get("seed") != job["seed"]
        or command_receipt.get("epoch") != 5
        or command_receipt.get("checkpoint_sha256")
        != recovery["checkpoint_identity"]["fixed_file_sha256"]
        or command_receipt.get("feature_only") is not True
        or command_receipt.get("inference_precision") != "fp32"
        or command_receipt.get("target_reads_during_export") is not False
    ):
        raise ValueError("R01 predictor did not produce the validated fixed-epoch endpoint")
    receipt_path = output / "export_receipt.json"
    saved_receipt = read_json(receipt_path)
    resource_path = output / "export_cell_resource_receipt.json"
    resource = read_json(resource_path)
    if (
        resource.get("schema") != "tabdata-b300-process-envelope.v1"
        or resource.get("status") != "actor_returned"
        or resource.get("script") != "export_cell.py"
        or resource.get("visible_devices") != visible_devices
        or "B300" not in str(resource.get("gpu", ""))
        or resource.get("allocator_budget_bytes") != 6 * 1024**3
        or resource.get("torch_threads") != 2
        or type(resource.get("cuda_peak_allocated_bytes")) is not int
        or type(resource.get("cuda_peak_reserved_bytes")) is not int
        or not 0 <= resource["cuda_peak_allocated_bytes"] <= 6 * 1024**3
        or not 0 <= resource["cuda_peak_reserved_bytes"] <= 6 * 1024**3
    ):
        raise ValueError("B300 export process envelope is missing or outside the frozen cap")
    if saved_receipt != command_receipt or saved_receipt.get("fit_receipt_sha256") != sha256_file(
        Path(recovery["fit_view"]) / "producer_receipt.json"
    ):
        raise ValueError("persisted R01 export receipt differs from the exact fit recovery view")
    recovery_export = {
        "schema": "tabdata-r01-export-recovery.v2",
        "status": "exported_fixed_epoch5_no_refit",
        "job_id": job_id,
        "fit_recovery_receipt_sha256": sha256_file(recovery_root / "receipts" / f"{job_id}.json"),
        "export_receipt_path": str(receipt_path),
        "export_receipt_sha256": sha256_file(receipt_path),
        "export_command_sha256": sha256_file(output / "export_command.json"),
        "export_resource_sha256": sha256_file(resource_path),
        "actor_log_sha256": sha256_file(actor_log),
        "actor_argv": argv,
        "visible_gpu_uuid": visible_devices,
        "checkpoint_sha256": command_receipt["checkpoint_sha256"],
        "prediction_sha256": command_receipt["prediction_sha256"],
        "sidecar_sha256": command_receipt["sidecar_sha256"],
        "old_fit_outputs_modified": False,
    }
    write_json_new(recovery_root / "receipts" / f"{job_id}.export.json", recovery_export)
    return recovery_export


def _link(source: Path, destination: Path, *, directory: bool = False) -> None:
    """Create one fresh symlink in the new, isolated grading view."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"grading view path already exists: {destination}")
    destination.symlink_to(source.resolve(strict=True), target_is_directory=directory)


def grade_pair(
    project_root: Path,
    packet_root: Path,
    source_root: Path,
    recovery_root: Path,
    wide_id: str,
    deep_id: str,
    *,
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """Run the original CPU-only official R4 paired grade for one same-seed fold."""
    root = project_root.resolve(strict=True)
    source = source_root.resolve(strict=True)
    _, wide = _verified_receipt(recovery_root, packet_root, wide_id)
    _, deep = _verified_receipt(recovery_root, packet_root, deep_id)
    wide_job, deep_job = wide["job"], deep["job"]
    if (
        wide_job.get("arm") != "wide"
        or deep_job.get("arm") != "deep"
        or (wide_job.get("fold"), wide_job.get("seed"))
        != (deep_job.get("fold"), deep_job.get("seed"))
    ):
        raise ValueError("official R01 grading requires the exact paired wide/deep cell")
    fold, seed = wide_job["fold"], wide_job["seed"]
    export_dirs = {
        "wide": recovery_root / "exports" / wide_id,
        "deep": recovery_root / "exports" / deep_id,
    }
    for arm, job_id in (("wide", wide_id), ("deep", deep_id)):
        output = export_dirs[arm]
        receipt = read_json(output / "export_receipt.json")
        recovery_export = read_json(recovery_root / "receipts" / f"{job_id}.export.json")
        if (
            recovery_export.get("schema") != "tabdata-r01-export-recovery.v2"
            or recovery_export.get("status") != "exported_fixed_epoch5_no_refit"
            or recovery_export.get("job_id") != job_id
            or recovery_export.get("fit_recovery_receipt_sha256")
            != sha256_file(recovery_root / "receipts" / f"{job_id}.json")
            or recovery_export.get("export_receipt_sha256")
            != sha256_file(output / "export_receipt.json")
            or recovery_export.get("export_command_sha256")
            != sha256_file(output / "export_command.json")
            or recovery_export.get("export_resource_sha256")
            != sha256_file(output / "export_cell_resource_receipt.json")
            or recovery_export.get("actor_log_sha256")
            != sha256_file(recovery_root / "logs" / f"{job_id}.export.log")
            or recovery_export.get("checkpoint_sha256")
            != (wide if arm == "wide" else deep)["checkpoint_identity"]["fixed_file_sha256"]
            or recovery_export.get("prediction_sha256") != receipt.get("prediction_sha256")
            or recovery_export.get("sidecar_sha256") != receipt.get("sidecar_sha256")
            or receipt.get("status") != "completed_fixed_epoch5_feature_only_fp32"
            or receipt.get("run_name") != job_id
            or receipt.get("checkpoint_sha256")
            != (wide if arm == "wide" else deep)["checkpoint_identity"]["fixed_file_sha256"]
        ):
            raise ValueError(f"export receipt is invalid for {arm} endpoint")

    attempt = recovery_root / "grades" / f"seed{seed}" / f"fold{fold}"
    attempt.mkdir(parents=True, exist_ok=False)
    view = attempt / "view"
    view.mkdir()
    data = root / "data"
    _link(data / "datasets", view / "datasets", directory=True)
    cache = view / "cache"
    cache.mkdir()
    for filename in CACHE_FILES:
        _link(data / "cache" / filename, cache / filename)
    artifact_runs = view / "artifacts/runs"
    artifact_runs.mkdir(parents=True)
    endpoint_jobs = {"wide": wide_job, "deep": deep_job}
    for arm, job_id in (("wide", wide_id), ("deep", deep_id)):
        fit_run = Path(endpoint_jobs[arm]["run_dir"])
        _link(fit_run / "artifacts/runs" / job_id, artifact_runs / job_id, directory=True)
        for suffix in ("npy", "json"):
            filename = f"pred_{job_id}_valid.{suffix}"
            _link(export_dirs[arm] / "cache" / filename, cache / filename)

    grader = root / "runtime/b300-r01-v1/grader/grade_endpoint.py"
    if sha256_file(grader) != GRADER_SHA256:
        raise ValueError("official R01 grader differs from the frozen SHA-256")
    scratch = attempt / "scratch"
    scratch.mkdir()
    temp = scratch / "tmp"
    temp.mkdir()
    output = attempt / "grade.json"
    split = root / "splits" / f"fold{fold}.json"
    argv = [
        sys.executable,
        str(grader),
        "--root",
        str(view),
        "--run",
        deep_id,
        "--split",
        str(split),
        "--epoch",
        "5",
        "--reference",
        f"{wide_id}@5",
        "--output",
        str(output),
        "--bootstrap",
        "2000",
        "--scratch-dir",
        str(scratch),
    ]
    command = {
        "schema": "tabdata-r01-delivery-grade-command.v2",
        "argv": argv,
        "candidate": deep_id,
        "reference": wide_id,
        "fold": fold,
        "seed": seed,
        "epoch": 5,
        "grader_sha256": GRADER_SHA256,
        "wide_fit_recovery_sha256": sha256_file(recovery_root / "receipts" / f"{wide_id}.json"),
        "deep_fit_recovery_sha256": sha256_file(recovery_root / "receipts" / f"{deep_id}.json"),
        "wide_export_receipt_sha256": sha256_file(export_dirs["wide"] / "export_receipt.json"),
        "deep_export_receipt_sha256": sha256_file(export_dirs["deep"] / "export_receipt.json"),
        "split_sha256": sha256_file(split),
        "cuda_disabled": True,
    }
    write_json_new(attempt / "grade_command.json", command)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(temp),
        PYTHONPATH=str(source / "src"),
    )
    started = time.time()
    with (
        (attempt / "stdout.txt").open("x", encoding="utf-8") as stdout,
        (attempt / "stderr.txt").open("x", encoding="utf-8") as stderr,
    ):
        process = subprocess.run(
            argv,
            cwd=root,
            env=env,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout_seconds,
            check=False,
        )
    if process.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"official R01 grader failed with exit {process.returncode}; logs retained"
        )
    grade = read_json(output)
    if (
        grade.get("metric_id") != "global_wp_clip2_v1"
        or grade.get("fold") != fold
        or grade.get("split_manifest_sha256") != sha256_file(split)
        or grade.get("endpoint", {}).get("run") != deep_id
        or grade.get("endpoint", {}).get("epoch") != 5
        or grade.get("reference", {}).get("run") != wide_id
        or grade.get("reference", {}).get("epoch") != 5
        or grade.get("paired_component_bootstrap", {}).get("resamples") != 2000
    ):
        raise ValueError("official grade does not bind the requested R01 pair and fixed endpoints")
    scores = {
        "deep_wp": grade.get("endpoint", {}).get("score", {}).get("wp"),
        "wide_wp": grade.get("reference", {}).get("score", {}).get("wp"),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in scores.values()
    ):
        raise ValueError("official paired grade contains a nonfinite or malformed WP score")
    receipt = {
        "schema": "tabdata-r01-delivery-grade.v2",
        "status": "graded",
        "candidate": deep_id,
        "reference": wide_id,
        "fold": fold,
        "seed": seed,
        "metric_id": grade["metric_id"],
        "grade_path": str(output),
        "grade_sha256": sha256_file(output),
        "command_sha256": sha256_file(attempt / "grade_command.json"),
        "stdout_sha256": sha256_file(attempt / "stdout.txt"),
        "stderr_sha256": sha256_file(attempt / "stderr.txt"),
        "elapsed_seconds": time.time() - started,
        "score": {
            **scores,
        },
        "old_fit_outputs_modified": False,
    }
    write_json_new(attempt / "grade_receipt.json", receipt)
    return receipt
