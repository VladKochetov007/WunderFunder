"""Profile the actual T4 allocation, then run the seed-3 F3 teacher and exports."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

PROFILE = "profile_clean_t3_f3_muonall_h384_e3_s3_t4_r1"
FIT = "fit_clean_t3_f3_muonall_h384_e3_s3_r1"
EXPORTS = [f"export_clean_t3_f3_muonall_h384_e3_s3_r1_{kind}" for kind in ("fit_train", "fit_valid", "full_valid")]
FINALIZATION_RESERVE_SECONDS = 300
GPU_HEADROOM_RESERVE_BYTES = 1_000_000_000


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


def stage(root: Path, runner, output: Path, name: str) -> dict:
    spec = root / "kaggle_specs" / f"{name}.json"
    result_dir = output / name
    runner.run(root, [spec], result_dir)
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


def profile_guard(root: Path, output: Path, result: dict) -> dict:
    spec = json.loads((root / "kaggle_specs" / f"{PROFILE}.json").read_text())
    log = output / PROFILE / PROFILE / "worker.log"
    matches = [
        line.removeprefix("PROFILE_UPDATE_COUNTS ").strip()
        for line in log.read_text(errors="replace").splitlines()
        if line.startswith("PROFILE_UPDATE_COUNTS ")
    ]
    if len(matches) != 1:
        raise RuntimeError("profile must produce exactly one update-count record")
    counts = json.loads(matches[0]).get("epoch", {})
    attempts = counts.get("optimizer_attempts")
    completed = counts.get("optimizer_completed_updates")
    skipped = counts.get("amp_skipped_updates")
    batches = counts.get("completed_batches")
    if (
        batches != 10
        or not all(isinstance(value, int) for value in (attempts, completed, skipped))
        or attempts <= 0
        or completed <= 0
        or completed + skipped != attempts
    ):
        raise RuntimeError("profile update counters failed the sealed 5+5-batch check")
    device = result.get("device", {})
    total_gpu = device.get("total_bytes")
    peak_gpu = result.get("peak_gpu_bytes")
    peak_ram = result.get("peak_ram_bytes")
    if (
        "T4" not in device.get("name", "")
        or not device.get("uuid")
        or not isinstance(total_gpu, int)
        or not isinstance(peak_gpu, int)
        or peak_gpu + GPU_HEADROOM_RESERVE_BYTES > total_gpu
    ):
        raise RuntimeError("profile did not retain 1 GB below measured physical T4 capacity")
    if not isinstance(peak_ram, int) or peak_ram > spec["limits"]["ram_bytes"]:
        raise RuntimeError("profile exceeded its worker RAM reservation")
    receipt = {
        "schema": "wnn.strategy.f3-teacher-profile-guard.r1",
        "status": "passed",
        "device": device,
        "peak_gpu_bytes": peak_gpu,
        "physical_gpu_total_bytes": total_gpu,
        "gpu_headroom_reserve_bytes": GPU_HEADROOM_RESERVE_BYTES,
        "peak_worker_ram_bytes": peak_ram,
        "profile_batches": batches,
        "optimizer_attempts": attempts,
        "optimizer_completed_updates": completed,
        "amp_skipped_updates": skipped,
        "profile_wall_seconds": result.get("wall_seconds"),
        "interpretation": "Measured profile guard; the reserve is physical-fit headroom, not a model VRAM ceiling.",
    }
    write_json(output / "profile_guard_receipt.json", receipt)
    return receipt


def main() -> None:
    root = Path("/kaggle/working/experiment")
    output = Path("/kaggle/working/results")
    output.mkdir(parents=True, exist_ok=True)
    pipeline_path = output / "f3_teacher_pipeline.json"
    pipeline = {"schema": "wnn.strategy.f3-teacher-pipeline.r1", "status": "running", "stages": []}
    write_json(pipeline_path, pipeline)
    try:
        sys.path.insert(0, str(root / "scripts"))
        runner = load_module("f3_kaggle_runner", root / "scripts/kaggle_gpu_runner.py")
        helpers = load_module("f3_kaggle_kernel", root / "scripts/kaggle_kernel.py")
        deadline_value = os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC")
        if deadline_value is None:
            raise RuntimeError("Kaggle bootstrap did not pass its provider-session deadline")
        deadline = float(deadline_value)
        profile_result = stage(root, runner, output / "profile", PROFILE)
        guard = profile_guard(root, output / "profile", profile_result)
        pipeline["profile_guard"] = guard
        pipeline["stages"].append({"name": PROFILE, "status": "complete"})
        write_json(pipeline_path, pipeline)
        needed = [FIT, *EXPORTS]
        timeout_sum = sum(
            json.loads((root / "kaggle_specs" / f"{name}.json").read_text())["limits"]["timeout_seconds"]
            for name in needed
        )
        required_seconds = timeout_sum + FINALIZATION_RESERVE_SECONDS
        remaining = helpers.remaining_seconds(deadline)
        if remaining <= required_seconds:
            raise TimeoutError(
                "profile left insufficient kernel lifetime for all declared fit/export worker caps "
                "and finalization reserve"
            )
        pipeline["stages"].append({"name": FIT, "status": "running"})
        write_json(pipeline_path, pipeline)
        fit_result = stage(root, runner, output / "fit", FIT)
        pipeline["stages"][-1] = {"name": FIT, "status": "complete", "result": fit_result}
        write_json(pipeline_path, pipeline)
        for name in EXPORTS:
            if helpers.remaining_seconds(deadline) <= FINALIZATION_RESERVE_SECONDS:
                raise TimeoutError("kernel window cannot finish remaining FP32 exports safely")
            result = stage(root, runner, output / "exports", name)
            pipeline["stages"].append({"name": name, "status": "complete", "result": result})
            write_json(pipeline_path, pipeline)
        pipeline["status"] = "complete"
        pipeline["fit_and_export_timeout_sum_seconds"] = timeout_sum
        pipeline["completed_elapsed_seconds"] = 43_200 - helpers.remaining_seconds(deadline)
        write_json(pipeline_path, pipeline)
    except BaseException as error:
        pipeline["status"] = "blocked_or_failed"
        pipeline["error_type"] = type(error).__name__
        pipeline["error"] = str(error)[:1000]
        write_json(pipeline_path, pipeline)
        raise


if __name__ == "__main__":
    main()
