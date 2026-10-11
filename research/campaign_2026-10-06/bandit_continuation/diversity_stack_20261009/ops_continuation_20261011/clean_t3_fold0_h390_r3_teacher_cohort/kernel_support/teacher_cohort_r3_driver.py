"""Run paired T4 profile guards before any teacher fit, then export all outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

PACKET_RELATIVE = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort"
)
PROFILE_NAMES = [f"profile_clean_t3_f0_muonall_h384_e3_s{seed}_paired_t4_r3" for seed in (3, 4)]
FIT_NAMES = [f"fit_clean_t3_f0_muonall_h384_e3_s{seed}_r3" for seed in (3, 4, 5)]
EXPORT_NAMES = [
    f"export_clean_t3_f0_muonall_h384_e3_s{seed}_r3_{kind}"
    for seed in (3, 4, 5)
    for kind in ("fit_train", "fit_valid", "full_valid")
]
FINALIZATION_RESERVE_SECONDS = 300
KERNEL_WALL_CEILING_SECONDS = 14_400
MAX_PROFILE_RAM_BYTES = 6_000_000_000
MAX_COMBINED_PROFILE_RAM_BYTES = 12_000_000_000
MAX_PROFILE_GPU_BYTES = 5_000_000_000


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load bundled helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return False


def parse_marker(log: Path, marker: str) -> dict:
    matches = [
        line[len(marker) :].strip()
        for line in log.read_text(errors="replace").splitlines()
        if line.startswith(marker)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {marker.strip()} line in {log.name}")
    value = json.loads(matches[0])
    if not isinstance(value, dict) or not finite_tree(value):
        raise RuntimeError(f"non-finite or malformed {marker.strip()} record")
    return value


def load_bootstrap_helpers(root: Path):
    return load_module("r3_kaggle_kernel_helpers", root / "scripts/kaggle_kernel.py")


def load_runner(root: Path):
    sys.path.insert(0, str(root / "scripts"))
    return load_module("r3_kaggle_gpu_runner", root / "scripts/kaggle_gpu_runner.py")


def validate_stage(root: Path, output: Path, names: list[str]) -> dict[str, dict]:
    summary = read_json(output / "summary.json")
    expected = set(names)
    rows = summary.get("finished", [])
    if summary.get("incident") is not None or summary.get("not_started"):
        raise RuntimeError("stage summary has an incident or unstarted workers")
    if {row.get("name") for row in rows} != expected:
        raise RuntimeError("stage summary does not contain exactly the requested workers")
    results = {}
    for name in names:
        row = next(item for item in rows if item.get("name") == name)
        result = read_json(output / name / "result.json")
        if (
            row.get("status") != "complete"
            or row.get("returncode") != 0
            or result.get("status") != "complete"
            or result.get("returncode") != 0
            or result.get("artifacts", {}).get("missing")
        ):
            raise RuntimeError(f"worker did not complete cleanly: {name}")
        declared = {item["path"] for item in result.get("artifacts", {}).get("files", [])}
        spec = read_json(root / "kaggle_specs" / f"{name}.json")
        if not set(spec.get("artifacts", [])).issubset(declared):
            raise RuntimeError(f"worker did not produce all declared artifacts: {name}")
        results[name] = result
    return results


def profile_guard(root: Path, output: Path, preflight: dict) -> dict:
    results = validate_stage(root, output, PROFILE_NAMES)
    devices = []
    workers = []
    for name in PROFILE_NAMES:
        result = results[name]
        device = result.get("device", {})
        if "T4" not in device.get("name", "") or not device.get("uuid"):
            raise RuntimeError(f"paired profile did not use a measured T4: {name}")
        devices.append(device["uuid"])
        ram = result.get("peak_ram_bytes")
        gpu = result.get("peak_gpu_bytes")
        if not isinstance(ram, int) or ram > MAX_PROFILE_RAM_BYTES:
            raise RuntimeError(f"profile RAM guard failed: {name}")
        if not isinstance(gpu, int) or gpu > MAX_PROFILE_GPU_BYTES:
            raise RuntimeError(f"profile GPU-memory guard failed: {name}")
        log = output / name / "worker.log"
        counts = parse_marker(log, "PROFILE_UPDATE_COUNTS ")
        epoch = counts.get("epoch", {})
        attempts = epoch.get("optimizer_attempts")
        completed = epoch.get("optimizer_completed_updates")
        skipped = epoch.get("amp_skipped_updates")
        batches = epoch.get("completed_batches")
        if (
            batches != 10
            or not isinstance(attempts, int)
            or not isinstance(completed, int)
            or not isinstance(skipped, int)
            or attempts <= 0
            or completed <= 0
            or completed + skipped != attempts
        ):
            raise RuntimeError(f"profile update counters failed the sealed guard: {name}")
        grad = parse_marker(log, "GRAD_STATS ")
        if grad.get("steps") != attempts or grad.get("nonfinite_steps", attempts + 1) > skipped:
            raise RuntimeError(
                f"gradient summary is inconsistent with skipped-update counts: {name}"
            )
        profile = parse_marker(log, "PROFILE ")
        if not isinstance(profile.get("step_wall"), (int, float)) or profile["step_wall"] <= 0:
            raise RuntimeError(f"profile batch timing is missing or nonpositive: {name}")
        workers.append(
            {
                "name": name,
                "device_uuid": device["uuid"],
                "peak_ram_bytes": ram,
                "peak_gpu_bytes": gpu,
                "completed_batches": batches,
                "optimizer_attempts": attempts,
                "optimizer_completed_updates": completed,
                "amp_skipped_updates": skipped,
                "gradient_nonfinite_steps": grad["nonfinite_steps"],
                "profile": profile,
            }
        )
    if len(set(devices)) != 2:
        raise RuntimeError("paired profiles were not placed on two distinct physical T4 UUIDs")
    expected_devices = {row["uuid"] for row in preflight["devices"]}
    if set(devices) != expected_devices:
        raise RuntimeError("paired profile devices differ from the preflight inventory")
    combined_ram = sum(row["peak_ram_bytes"] for row in workers)
    if combined_ram > MAX_COMBINED_PROFILE_RAM_BYTES:
        raise RuntimeError("combined per-worker peak RAM exceeded the paired-profile guard")
    host = preflight["host"]
    budget = min(
        int(host["capacity_bytes"] * 0.8),
        host["available_bytes"] - int(host["capacity_bytes"] * 0.2),
    )
    if budget <= 0 or combined_ram > budget:
        raise RuntimeError("observed paired RAM peaks exceed the initial host operational budget")
    receipt = {
        "schema": "wnn.kaggle.paired_t4_profile_guard_receipt.r1",
        "status": "pass",
        "runner_concurrency": 2,
        "profile_workers": workers,
        "distinct_t4_uuids": sorted(set(devices)),
        "sum_worker_peak_ram_bytes": combined_ram,
        "initial_host_budget_bytes": budget,
        "host_inventory": host,
        "device_inventory": preflight["devices"],
        "fit_release": "still requires embedded root conditional admission; this receipt alone cannot authorize fits",
    }
    return receipt


def ensure_pair_admission(root: Path, profile_names: list[str], output_root: Path) -> dict:
    runner = load_runner(root)
    helpers = load_bootstrap_helpers(root)
    host, devices = runner.host_inventory(), runner.gpu_inventory()
    if len(devices) != 2 or any("T4" not in d.get("name", "") for d in devices):
        raise RuntimeError("R3 paired profile gate requires exactly two visible Tesla T4 devices")
    specs = [runner.read_spec(root / "kaggle_specs" / f"{name}.json") for name in profile_names]
    admitted = runner.admit_specs(specs, host, devices)
    if admitted != 2:
        raise RuntimeError(
            "runner host/CPU admission would serialize the paired profile; refusing to launch"
        )
    preflight = {"host": host, "devices": devices, "concurrency_admitted": admitted}
    write_json(output_root / "profile_preflight_inventory.json", preflight)
    return {"runner": runner, "helpers": helpers, "specs": specs, "preflight": preflight}


def run_stage(
    root: Path, output_root: Path, stage: str, names: list[str], deadline: float, helpers
) -> dict:
    output = output_root / stage
    command = [
        sys.executable,
        str(root / "scripts/kaggle_gpu_runner.py"),
        "--root",
        str(root),
        "--output",
        str(output),
    ]
    runtime_specs = []
    for name in names:
        source = root / "kaggle_specs" / f"{name}.json"
        runtime = Path("/kaggle/working/runtime_specs") / source.name
        helpers.write_runtime_spec(source, runtime, deadline)
        runtime_specs.append(runtime)
        command.extend(["--spec", str(runtime)])
    remaining = helpers.remaining_seconds(deadline)
    if remaining <= FINALIZATION_RESERVE_SECONDS:
        raise TimeoutError(f"insufficient kernel wall budget to begin {stage}")
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=root, env=os.environ.copy(), start_new_session=True)
    try:
        returncode = process.wait(timeout=max(1, int(remaining - FINALIZATION_RESERVE_SECONDS)))
    except subprocess.TimeoutExpired as error:
        helpers.terminate_process_tree(process)
        raise TimeoutError(f"{stage} exceeded remaining bounded kernel wall time") from error
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    result = {
        "name": stage,
        "status": "complete",
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "workers": names,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    packet = root / PACKET_RELATIVE
    gate = read_json(packet / "CONDITIONAL_FIT_ADMISSION_R3.json")
    if gate.get("status") != "admitted" or gate.get("release_authorized") is not True:
        raise PermissionError("R3 root conditional admission is pending")
    release_sha = gate.get("root_release_receipt_sha256")
    if (
        not isinstance(release_sha, str)
        or len(release_sha) != 64
        or any(character not in "0123456789abcdef" for character in release_sha)
    ):
        raise PermissionError("R3 root release receipt is not bound")
    if gate.get("profile_reference_sha256") != sha256_file(packet / "PROFILE_REFERENCE_R3.json"):
        raise PermissionError("R3 root gate does not bind the actual single-T4 profile reference")
    if gate.get("profile_guard_contract_sha256") != sha256_file(
        packet / "PROFILE_GUARD_CONTRACT_R3.json"
    ):
        raise PermissionError("R3 root gate does not bind the paired-T4 profile guard")
    deadline_value = os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC")
    if deadline_value is None:
        raise RuntimeError("Kaggle bootstrap did not pass its monotonic deadline")
    deadline = float(deadline_value)
    output_root = Path("/kaggle/working/results")
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_path = output_root / "teacher_cohort_pipeline.json"
    pipeline = {
        "schema": "wnn.kaggle.teacher_cohort_pipeline.r3",
        "status": "running",
        "stages": [],
    }
    write_json(pipeline_path, pipeline)
    try:
        pair = ensure_pair_admission(root, PROFILE_NAMES, output_root)
        pipeline["paired_profile_preflight"] = pair["preflight"]
        pipeline["stages"].append(
            run_stage(
                root, output_root, "paired_profiles", PROFILE_NAMES, deadline, pair["helpers"]
            )
        )
        profile_output = output_root / "paired_profiles"
        guard_receipt = profile_guard(root, profile_output, pair["preflight"])
        write_json(output_root / "paired_profile_guard_receipt.json", guard_receipt)
        pipeline["paired_profile_guard"] = guard_receipt
        pipeline["status"] = "profile_guard_passed_pending_fit_stage"
        write_json(pipeline_path, pipeline)

        # The embedded root receipt authorizes a conditional continuation. This
        # exact branch runs only after the paired profiles pass every guard.
        if gate.get("profile_guard_contract_sha256") != sha256_file(
            packet / "PROFILE_GUARD_CONTRACT_R3.json"
        ):
            raise PermissionError("root gate does not bind the frozen paired-profile contract")
        runner = pair["runner"]
        host, devices = runner.host_inventory(), runner.gpu_inventory()
        fit_specs = [runner.read_spec(root / "kaggle_specs" / f"{name}.json") for name in FIT_NAMES]
        if runner.admit_specs(fit_specs, host, devices) != 2:
            raise RuntimeError(
                "paired fit admission no longer permits two workers; no full fits were started"
            )
        helpers = pair["helpers"]
        pipeline["stages"].append(
            run_stage(root, output_root, "teacher_fits", FIT_NAMES, deadline, helpers)
        )
        write_json(pipeline_path, pipeline)
        validate_stage(root, output_root / "teacher_fits", FIT_NAMES)
        pipeline["stages"].append(
            run_stage(root, output_root, "teacher_exports", EXPORT_NAMES, deadline, helpers)
        )
        pipeline["status"] = "complete"
        pipeline["completed_elapsed_seconds"] = (
            KERNEL_WALL_CEILING_SECONDS - helpers.remaining_seconds(deadline)
        )
        write_json(pipeline_path, pipeline)
    except BaseException as error:
        pipeline["status"] = "blocked_or_failed"
        pipeline["error_type"] = type(error).__name__
        pipeline["error"] = str(error)[:1000]
        write_json(pipeline_path, pipeline)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
