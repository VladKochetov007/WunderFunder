"""Run the frozen teacher-fit stage, verify it, then run its prediction exports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PACKET_RELATIVE = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_fold0_h390_r2_teacher_cohort"
)
FIT_NAMES = [
    "fit_clean_t3_f0_muonall_h384_e3_s3_r2",
    "fit_clean_t3_f0_muonall_h384_e3_s4_r2",
    "fit_clean_t3_f0_muonall_h384_e3_s5_r2",
]
EXPORT_NAMES = [
    f"export_clean_t3_f0_muonall_h384_e3_s{seed}_r2_{kind}"
    for seed in (3, 4, 5)
    for kind in ("fit_train", "fit_valid", "full_valid")
]
FINALIZATION_RESERVE_SECONDS = 300
KERNEL_WALL_CEILING_SECONDS = 14_400


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_bootstrap_helpers(root: Path):
    path = root / "scripts/kaggle_kernel.py"
    spec = importlib.util.spec_from_file_location("r2_kaggle_kernel_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import the bundled Kaggle bootstrap helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_stage(output: Path, specs: list[Path]) -> None:
    expected = {path.stem for path in specs}
    summary = json.loads((output / "summary.json").read_text())
    finished = summary.get("finished", [])
    observed = {row.get("name") for row in finished}
    if summary.get("incident") is not None or summary.get("not_started") or observed != expected:
        raise RuntimeError("stage summary is incomplete or reports a worker incident")
    if any(row.get("status") != "complete" or row.get("returncode") != 0 for row in finished):
        raise RuntimeError("stage contains a failed worker")
    for name in expected:
        record = json.loads((output / name / "result.json").read_text())
        if record.get("status") != "complete" or record.get("returncode") != 0:
            raise RuntimeError(f"worker result is not complete: {name}")
        if record.get("artifacts", {}).get("missing"):
            raise RuntimeError(f"worker did not produce every declared artifact: {name}")
        declared = {item["path"] for item in record.get("artifacts", {}).get("files", [])}
        spec = json.loads(
            (output.parent.parent / "experiment" / "kaggle_specs" / f"{name}.json").read_text()
        )
        if not set(spec["artifacts"]).issubset(declared):
            raise RuntimeError(f"worker artifact receipt is incomplete: {name}")


def run_stage(
    root: Path, output_root: Path, stage: str, names: list[str], deadline: float, helpers
) -> dict:
    output = output_root / stage
    spec_paths = [root / "kaggle_specs" / f"{name}.json" for name in names]
    runtime_paths: list[Path] = []
    command = [
        sys.executable,
        str(root / "scripts/kaggle_gpu_runner.py"),
        "--root",
        str(root),
        "--output",
        str(output),
    ]
    for source in spec_paths:
        runtime = Path("/kaggle/working/runtime_specs") / source.name
        helpers.write_runtime_spec(source, runtime, deadline)
        runtime_paths.append(runtime)
        command.extend(["--spec", str(runtime)])
    remaining = helpers.remaining_seconds(deadline)
    if remaining <= FINALIZATION_RESERVE_SECONDS:
        raise TimeoutError(f"insufficient wall budget to begin {stage}")
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=root, env=os.environ.copy(), start_new_session=True)
    try:
        returncode = process.wait(timeout=max(1, int(remaining - FINALIZATION_RESERVE_SECONDS)))
    except subprocess.TimeoutExpired as error:
        helpers.terminate_process_tree(process)
        raise TimeoutError(f"{stage} exceeded the remaining bounded cohort wall time") from error
    elapsed = time.monotonic() - started
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    validate_stage(output, runtime_paths)
    return {
        "name": stage,
        "status": "complete",
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "specs": names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    raw_deadline = os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC")
    if raw_deadline is None:
        raise RuntimeError("bootstrap did not pass its monotonic deadline")
    deadline = float(raw_deadline)
    packet = root / PACKET_RELATIVE
    expected_gate = json.loads((packet / "CONDITIONAL_FIT_ADMISSION_R2.json").read_text())
    if (
        expected_gate.get("status") != "admitted"
        or expected_gate.get("release_authorized") is not True
    ):
        raise PermissionError("R2 full-fit admission is still pending")
    helpers = load_bootstrap_helpers(root)
    output_root = Path("/kaggle/working/results")
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_path = output_root / "teacher_cohort_pipeline.json"
    pipeline = {
        "schema": "wnn.kaggle.teacher_cohort_pipeline.r2",
        "status": "running",
        "stages": [],
    }
    write_json(pipeline_path, pipeline)
    try:
        pipeline["stages"].append(
            run_stage(root, output_root, "teacher_fits", FIT_NAMES, deadline, helpers)
        )
        write_json(pipeline_path, pipeline)
        pipeline["stages"].append(
            run_stage(root, output_root, "teacher_exports", EXPORT_NAMES, deadline, helpers)
        )
        pipeline["status"] = "complete"
        pipeline["completed_elapsed_seconds"] = (
            KERNEL_WALL_CEILING_SECONDS - helpers.remaining_seconds(deadline)
        )
        write_json(pipeline_path, pipeline)
        if helpers.remaining_seconds(deadline) <= 0:
            raise TimeoutError("cohort completed outside its wall ceiling")
    except BaseException as error:
        pipeline["status"] = "failed"
        pipeline["error_type"] = type(error).__name__
        pipeline["error"] = str(error)[:1000]
        write_json(pipeline_path, pipeline)
        raise


if __name__ == "__main__":
    main()
