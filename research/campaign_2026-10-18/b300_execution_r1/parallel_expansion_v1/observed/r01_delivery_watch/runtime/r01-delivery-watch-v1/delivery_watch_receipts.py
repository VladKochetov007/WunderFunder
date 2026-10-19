"""Independent validators for completed R01 fits, exports, and grades."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from delivery_watch_core import read_object, sha256_file


def verify_original_actor_binding(
    project_root: Path,
    job: dict[str, Any],
    receipt: dict[str, Any],
    placement_verifier: Any,
) -> dict[str, Any]:
    """Rehash the original actor resource and check its pinned placement authority."""
    actor_phase = receipt.get("original_actor_phase")
    run_dir = Path(job["run_dir"]).resolve(strict=True)
    resource_path = run_dir / "train_cell_resource_receipt.json"
    if (
        not isinstance(actor_phase, dict)
        or resource_path.is_symlink()
        or not resource_path.is_file()
        or actor_phase.get("resource_receipt_sha256") != sha256_file(resource_path)
    ):
        raise ValueError("original actor resource receipt changed after fit validation")
    resource = read_object(resource_path)
    placement = placement_verifier(project_root, job, resource)
    recorded = actor_phase.get("placement_authorization")
    if recorded is None:
        if placement.get("authority") != "frozen_panel_initial_placement":
            raise ValueError("legacy fit receipt lacks changed-placement authorization")
    elif recorded != placement:
        raise ValueError("original actor placement authorization differs from pinned evidence")
    return placement


def verify_fit_receipt(
    helper: Any,
    recovery_root: Path,
    packet_root: Path,
    project_root: Path,
    job: dict[str, Any],
    placement_verifier: Any,
) -> dict[str, Any]:
    """Verify an existing recovery receipt and rehash every original fit output."""
    _, receipt = helper._verified_receipt(recovery_root, packet_root, str(job["id"]))
    run_dir = Path(job["run_dir"]).resolve(strict=True)
    verify_gradient_history(run_dir, str(job["id"]))
    verify_original_actor_binding(project_root, job, receipt, placement_verifier)
    if (
        Path(receipt["original_run_dir"]).resolve(strict=True) != run_dir
        or receipt.get("original_command_sha256") != sha256_file(run_dir / "command.json")
        or receipt.get("original_preflight_sha256")
        != sha256_file(run_dir / "adapter_preflight.json")
    ):
        raise ValueError("existing fit receipt no longer binds its original command/preflight")
    outputs = receipt.get("recovery_fit_receipt", {}).get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("existing fit receipt has no output hash map")
    for relative, meta in outputs.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe fit output path: {relative}")
        path = run_dir / rel
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != meta.get("bytes")
            or sha256_file(path) != meta.get("sha256")
        ):
            raise ValueError(f"completed fit artifact changed after validation: {relative}")
    return receipt


def verify_gradient_history(run_dir: Path, job_id: str) -> str:
    """Require all six recorded gradient windows to match optimizer updates."""
    history_path = run_dir / "artifacts" / "runs" / job_id / "history.json"
    if history_path.is_symlink() or not history_path.is_file():
        raise ValueError("R01 history is not a regular-file artifact")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != 6:
        raise ValueError("R01 history is not the six-epoch regular-file artifact")
    for index, row in enumerate(history):
        if not isinstance(row, dict):
            raise ValueError(f"R01 history epoch {index} is malformed")
        diagnostics = row.get("update_diagnostics", {})
        epoch = diagnostics.get("epoch", {})
        gradients = diagnostics.get("grad_stats", {})
        completed = epoch.get("optimizer_completed_updates")
        steps = gradients.get("steps")
        nonfinite = gradients.get("nonfinite_steps")
        if (
            type(row.get("epoch")) is not int
            or row["epoch"] != index
            or type(completed) is not int
            or completed <= 0
            or type(steps) is not int
            or steps != completed
            or type(nonfinite) is not int
            or nonfinite != 0
        ):
            raise ValueError(f"R01 gradient diagnostics fail the epoch {index} update gate")
    return sha256_file(history_path)


def verify_export_receipt(
    recovery_root: Path,
    project_root: Path,
    job_id: str,
    fit_receipt: dict[str, Any],
    gpu_uuid: str,
) -> dict[str, Any]:
    """Verify an existing recovery export receipt and its endpoint files."""
    record_path = recovery_root / "receipts" / f"{job_id}.export.json"
    record = read_object(record_path)
    output = recovery_root / "exports" / job_id
    export_path = output / "export_receipt.json"
    command_path = output / "export_command.json"
    resource_path = output / "export_cell_resource_receipt.json"
    log_path = recovery_root / "logs" / f"{job_id}.export.log"
    exported = read_object(export_path)
    command = read_object(command_path)
    resource = read_object(resource_path)
    argv = command.get("argv")
    fit_job = fit_receipt.get("job")
    if not isinstance(fit_job, dict):
        raise ValueError(f"fit receipt lacks its frozen panel job: {job_id}")
    expected_actor = (project_root / "runtime/b300-r01-v1/code/export_cell.py").resolve(strict=True)
    if (
        record.get("schema") != "tabdata-r01-export-recovery.v2"
        or record.get("status") != "exported_fixed_epoch5_no_refit"
        or record.get("job_id") != job_id
        or record.get("old_fit_outputs_modified") is not False
        or record.get("fit_recovery_receipt_sha256")
        != sha256_file(recovery_root / "receipts" / f"{job_id}.json")
        or record.get("export_receipt_sha256") != sha256_file(export_path)
        or record.get("export_command_sha256") != sha256_file(command_path)
        or record.get("export_resource_sha256") != sha256_file(resource_path)
        or record.get("actor_log_sha256") != sha256_file(log_path)
        or record.get("visible_gpu_uuid") != gpu_uuid
        or resource.get("schema") != "tabdata-b300-process-envelope.v1"
        or resource.get("status") != "actor_returned"
        or resource.get("script") != "export_cell.py"
        or resource.get("visible_devices") != gpu_uuid
        or "B300" not in str(resource.get("gpu", ""))
        or resource.get("allocator_budget_bytes") != 6 * 1024**3
        or resource.get("torch_threads") != 2
        or type(resource.get("cuda_peak_allocated_bytes")) is not int
        or type(resource.get("cuda_peak_reserved_bytes")) is not int
        or not 0 <= resource["cuda_peak_allocated_bytes"] <= 6 * 1024**3
        or not 0 <= resource["cuda_peak_reserved_bytes"] <= 6 * 1024**3
        or not isinstance(argv, list)
        or len(argv) < 3
        or not Path(argv[2]).is_absolute()
        or Path(argv[2]).resolve(strict=True) != expected_actor
        or exported.get("status") != "completed_fixed_epoch5_feature_only_fp32"
        or exported.get("run_name") != job_id
        or exported.get("fold") != fit_job.get("fold")
        or exported.get("seed") != fit_job.get("seed")
        or exported.get("epoch") != 5
        or exported.get("feature_only") is not True
        or exported.get("inference_precision") != "fp32"
        or exported.get("target_reads_during_export") is not False
        or exported.get("checkpoint_sha256")
        != fit_receipt["checkpoint_identity"]["fixed_file_sha256"]
        or exported.get("prediction_sha256") != record.get("prediction_sha256")
        or exported.get("sidecar_sha256") != record.get("sidecar_sha256")
    ):
        raise ValueError(f"existing export receipt is invalid: {job_id}")
    for suffix, expected in (
        ("npy", record["prediction_sha256"]),
        ("json", record["sidecar_sha256"]),
    ):
        path = output / "cache" / f"pred_{job_id}_valid.{suffix}"
        if sha256_file(path) != expected:
            raise ValueError(f"existing export payload changed: {path}")
    return record


def verify_grade_receipt(
    recovery_root: Path,
    project_root: Path,
    wide_id: str,
    deep_id: str,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    """Verify a completed official wide/deep grade without rerunning it."""
    attempt = recovery_root / "grades" / f"seed{seed}" / f"fold{fold}"
    receipt = read_object(attempt / "grade_receipt.json")
    grade_path, command_path = attempt / "grade.json", attempt / "grade_command.json"
    command = read_object(command_path)
    grade = read_object(grade_path)
    wide_fit = recovery_root / "receipts" / f"{wide_id}.json"
    deep_fit = recovery_root / "receipts" / f"{deep_id}.json"
    wide_export = recovery_root / "exports" / wide_id / "export_receipt.json"
    deep_export = recovery_root / "exports" / deep_id / "export_receipt.json"
    split_path = project_root / "splits" / f"fold{fold}.json"
    argv = command.get("argv")
    endpoint_score = grade.get("endpoint", {}).get("score", {}).get("wp")
    reference_score = grade.get("reference", {}).get("score", {}).get("wp")
    if (
        receipt.get("schema") != "tabdata-r01-delivery-grade.v2"
        or receipt.get("status") != "graded"
        or receipt.get("candidate") != deep_id
        or receipt.get("reference") != wide_id
        or receipt.get("fold") != fold
        or receipt.get("seed") != seed
        or receipt.get("metric_id") != "global_wp_clip2_v1"
        or receipt.get("grade_sha256") != sha256_file(grade_path)
        or receipt.get("command_sha256") != sha256_file(command_path)
        or receipt.get("stdout_sha256") != sha256_file(attempt / "stdout.txt")
        or receipt.get("stderr_sha256") != sha256_file(attempt / "stderr.txt")
        or command.get("candidate") != deep_id
        or command.get("reference") != wide_id
        or command.get("schema") != "tabdata-r01-delivery-grade-command.v2"
        or command.get("epoch") != 5
        or command.get("wide_fit_recovery_sha256") != sha256_file(wide_fit)
        or command.get("deep_fit_recovery_sha256") != sha256_file(deep_fit)
        or command.get("wide_export_receipt_sha256") != sha256_file(wide_export)
        or command.get("deep_export_receipt_sha256") != sha256_file(deep_export)
        or command.get("split_sha256") != sha256_file(split_path)
        or not isinstance(argv, list)
        or len(argv) < 9
        or Path(argv[1]).resolve(strict=True)
        != (project_root / "runtime/b300-r01-v1/grader/grade_endpoint.py").resolve(strict=True)
        or Path(argv[3]).resolve(strict=True)
        != (recovery_root / "grades" / f"seed{seed}" / f"fold{fold}" / "view").resolve(strict=True)
        or Path(argv[7]).resolve(strict=True) != split_path.resolve(strict=True)
        or command.get("cuda_disabled") is not True
        or command.get("grader_sha256")
        != "56ffc693da52668726bab1f7008e31753a76aed3ff5d79089167785079d9eb31"
        or grade.get("metric_id") != "global_wp_clip2_v1"
        or grade.get("fold") != fold
        or grade.get("endpoint", {}).get("run") != deep_id
        or grade.get("endpoint", {}).get("epoch") != 5
        or grade.get("reference", {}).get("run") != wide_id
        or grade.get("reference", {}).get("epoch") != 5
        or grade.get("split_manifest_sha256") != command.get("split_sha256")
        or grade.get("paired_component_bootstrap", {}).get("resamples") != 2000
        or isinstance(endpoint_score, bool)
        or not isinstance(endpoint_score, (int, float))
        or not math.isfinite(endpoint_score)
        or isinstance(reference_score, bool)
        or not isinstance(reference_score, (int, float))
        or not math.isfinite(reference_score)
        or receipt.get("score") != {"deep_wp": endpoint_score, "wide_wp": reference_score}
    ):
        raise ValueError(f"existing official grade receipt is invalid: {wide_id}/{deep_id}")
    return receipt
