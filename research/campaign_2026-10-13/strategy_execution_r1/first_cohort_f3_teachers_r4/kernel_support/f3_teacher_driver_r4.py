"""Reuse the sealed R2 T4 profile, then fit and export the F3 seed-3 teacher."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PACKET = Path("research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r4")
FIT = "fit_clean_t3_f3_muonall_h384_e3_s3_r4"
EXPORTS = ["export_clean_t3_f3_muonall_h384_e3_s3_r4_fit_train"]
PENDING_EXPORTS = [
    "export_clean_t3_f3_muonall_h384_e3_s3_r4_fit_valid",
    "export_clean_t3_f3_muonall_h384_e3_s3_r4_full_valid",
]
EXPORT_BUDGETS = {EXPORTS[0]: 6500}
FIT_CAP_SECONDS = 8500
FINALIZATION_RESERVE_SECONDS = 900
PROVIDER_SESSION_CAP_SECONDS = 43_200
MAX_QUOTA_SNAPSHOT_AGE_SECONDS = 3600
QUOTA_SAFETY_RESERVE_GPU_SECONDS = 1800
CONSERVATIVE_BILLABLE_GPU_COUNT = 2


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load sealed helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def quota_admission(root: Path, helpers, provider_started: float, provider_deadline: float) -> dict:
    """Convert the live frozen quota snapshot to a conservative wall deadline."""
    packet = root / PACKET
    quota = json.loads((packet / "KAGGLE_QUOTA_GUARD_R4.json").read_text())
    snapshot = quota["quota_snapshot"]
    checked = datetime.fromisoformat(snapshot["checked_utc"].replace("Z", "+00:00"))
    age = (datetime.now(UTC) - checked).total_seconds()
    if age < 0 or age > MAX_QUOTA_SNAPSHOT_AGE_SECONDS:
        raise RuntimeError("Kaggle quota snapshot is stale; refresh and reseal before dispatch")
    remaining_gpu = float(snapshot["gpu_remaining_seconds"])
    reserved_gpu = float(snapshot["gpu_reserved_seconds"])
    reserve_gpu = int(quota["billing_assumption"]["quota_safety_reserve_gpu_seconds"])
    devices = int(quota["billing_assumption"]["billable_gpu_count"])
    usable_gpu = remaining_gpu - reserved_gpu - reserve_gpu
    if usable_gpu <= 0 or devices != CONSERVATIVE_BILLABLE_GPU_COUNT:
        raise RuntimeError("quota snapshot has no safe two-device billing allowance")
    window_seconds = int(usable_gpu // devices)
    if (
        provider_deadline <= provider_started
        or provider_deadline - provider_started > PROVIDER_SESSION_CAP_SECONDS + 1
    ):
        raise RuntimeError("Kaggle bootstrap supplied an invalid provider session window")
    quota_deadline = provider_started + window_seconds
    effective_deadline = min(provider_deadline, quota_deadline)
    budgets = {
        "fit": FIT_CAP_SECONDS,
        "fit_train_export": sum(EXPORT_BUDGETS.values()),
        "finalization": FINALIZATION_RESERVE_SECONDS,
    }
    total_required = sum(budgets.values())
    available = helpers.remaining_seconds(effective_deadline)
    if available <= total_required:
        raise TimeoutError(
            "quota-derived wall window cannot reserve fit, fit-train export, and finalization"
        )
    return {
        "schema": "wnn.strategy.f3-teacher-quota-admission.r4",
        "status": "passed",
        "quota_snapshot_checked_utc": snapshot["checked_utc"],
        "quota_snapshot_age_seconds": age,
        "gpu_seconds_remaining_at_snapshot": remaining_gpu,
        "gpu_seconds_reserved_at_snapshot": reserved_gpu,
        "gpu_safety_reserve_seconds": reserve_gpu,
        "conservative_billable_gpu_count": devices,
        "quota_derived_wall_window_seconds": window_seconds,
        "provider_session_cap_seconds": PROVIDER_SESSION_CAP_SECONDS,
        "provider_started_monotonic": provider_started,
        "available_effective_wall_seconds_at_admission": available,
        "stage_budgets_seconds": budgets,
        "maximum_planned_stage_plus_finalization_seconds": total_required,
        "quota_deadline_monotonic": quota_deadline,
        "effective_deadline_monotonic": effective_deadline,
        "interpretation": "Two T4 billable devices are assumed conservatively until actual quota billing is verified; worker execution remains serialized on one visible T4 device.",
    }


def stage(root: Path, runner, output: Path, name: str, timeout_seconds: int | None = None) -> dict:
    sealed_spec = root / "kaggle_specs" / f"{name}.json"
    spec_path = sealed_spec
    if timeout_seconds is not None:
        spec = json.loads(sealed_spec.read_text())
        declared = spec["limits"]["timeout_seconds"]
        if timeout_seconds < 1 or timeout_seconds > declared:
            raise ValueError(f"runtime worker budget is outside sealed cap: {name}")
        spec["limits"]["timeout_seconds"] = timeout_seconds
        spec_path = output / "runtime_specs" / f"{name}.json"
        write_json(spec_path, spec)
    result_dir = output / name
    runner.run(root, [spec_path], result_dir)
    summary = json.loads((result_dir / "summary.json").read_text())
    rows = summary.get("finished", [])
    if summary.get("incident") is not None or len(rows) != 1 or rows[0].get("name") != name:
        raise RuntimeError(f"stage failed or did not finish exactly one worker: {name}")
    result = json.loads((result_dir / name / "result.json").read_text())
    if result.get("status") != "complete" or result.get("returncode") != 0:
        raise RuntimeError(f"worker did not complete cleanly: {name}")
    if result.get("artifacts", {}).get("missing"):
        raise RuntimeError(f"worker omitted a declared output: {name}")
    return result


def main() -> None:
    root = Path("/kaggle/working/experiment")
    output = Path("/kaggle/working/results")
    output.mkdir(parents=True, exist_ok=True)
    pipeline_path = output / "f3_teacher_pipeline.json"
    pipeline = {
        "schema": "wnn.strategy.f3-teacher-pipeline.r4",
        "status": "admission_pending",
        "stages": [],
    }
    write_json(pipeline_path, pipeline)
    try:
        sys.path.insert(0, str(root / "scripts"))
        runner = load_module("f3_kaggle_runner_r4", root / "scripts/kaggle_gpu_runner.py")
        helpers = load_module("f3_kaggle_kernel_r4", root / "scripts/kaggle_kernel.py")
        profile_reuse = load_module(
            "f3_profile_reuse_r4", root / PACKET / "kernel_support/profile_reuse.py"
        )
        deadline_value = os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC")
        started_value = os.environ.get("WNN_KERNEL_STARTED_MONOTONIC")
        if deadline_value is None or started_value is None:
            raise RuntimeError("Kaggle bootstrap did not pass its provider-session window")
        provider_deadline = float(deadline_value)
        provider_started = float(started_value)
        quota = quota_admission(root, helpers, provider_started, provider_deadline)
        effective_deadline = quota["effective_deadline_monotonic"]
        write_json(output / "quota_admission.json", quota)
        pipeline["quota_admission"] = quota
        pipeline["status"] = "running"
        write_json(pipeline_path, pipeline)

        reuse_receipt = profile_reuse.validate_profile_reuse(root)
        write_json(output / "profile_reuse_receipt.json", reuse_receipt)
        pipeline["profile_reuse"] = reuse_receipt
        pipeline["stages"].append(
            {"name": "profile_reuse_r2", "status": "complete", "result": reuse_receipt}
        )
        write_json(pipeline_path, pipeline)
        remaining_pipeline_need = (
            FIT_CAP_SECONDS + sum(EXPORT_BUDGETS.values()) + FINALIZATION_RESERVE_SECONDS
        )
        if helpers.remaining_seconds(effective_deadline) <= remaining_pipeline_need:
            raise TimeoutError(
                "profile reuse left insufficient quota-derived wall time for fit, fit-train export, and finalization"
            )
        pipeline["stages"].append(
            {"name": FIT, "status": "running", "timeout_seconds": FIT_CAP_SECONDS}
        )
        write_json(pipeline_path, pipeline)
        fit_result = stage(root, runner, output / "fit", FIT, FIT_CAP_SECONDS)
        pipeline["stages"][-1] = {"name": FIT, "status": "complete", "result": fit_result}
        write_json(pipeline_path, pipeline)

        remaining_exports = list(EXPORTS)
        for name in EXPORTS:
            required = (
                sum(EXPORT_BUDGETS[item] for item in remaining_exports)
                + FINALIZATION_RESERVE_SECONDS
            )
            if helpers.remaining_seconds(effective_deadline) <= required:
                raise TimeoutError(
                    f"quota-derived wall time cannot preserve required output: {remaining_exports}"
                )
            result = stage(root, runner, output / "exports", name, EXPORT_BUDGETS[name])
            remaining_exports.remove(name)
            pipeline["stages"].append({"name": name, "status": "complete", "result": result})
            write_json(pipeline_path, pipeline)
        pipeline["pending_exports"] = PENDING_EXPORTS
        pipeline["status"] = "fit_and_fit_train_complete_exports_pending"
        pipeline["effective_completed_elapsed_seconds"] = time.monotonic() - provider_started
        write_json(pipeline_path, pipeline)
    except BaseException as error:
        pipeline["status"] = "blocked_or_failed"
        pipeline["error_type"] = type(error).__name__
        pipeline["error"] = str(error)[:1000]
        write_json(pipeline_path, pipeline)
        raise


if __name__ == "__main__":
    main()
