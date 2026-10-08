from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
from kaggle_common import read_spec, relative_path, sha256_file, verify_files, write_json

WORKER_TIMEOUT_CEILING_SECONDS = 5_200
WORKER_TIMEOUT_FINALIZATION_RESERVE_SECONDS = 300
E540_PHASE_RECEIPT_BYTES_MAX = 64 * 1024
E540_PHASE_RECEIPT_AGE_SECONDS_MAX = 6_000
E540_PHASE_METRIC_SECONDS_MAX = 5_200
E540_PHASE_PROJECTION_SECONDS_MAX = 10_000
E540_PHASE_UPDATE_COUNT_MAX = 72_996
E540_PHASE_STEP_WALL_MS_MAX = 100_000
E540_V7_SPEC_NAME = "e540-v7-fp32-20261008"
E540_V7_PACKET_ID = "E540_FP32_V7"
E540_PHASES = {
    "input_identity_preflight_completed_by_kaggle_gpu_runner": ("input_identity", None),
    "membership_verified_before_profile_and_fit": ("membership", None),
    "profile_linear": ("profile", "linear"),
    "profile_linear_failed": ("profile", "linear"),
    "profile_ple": ("profile", "ple"),
    "profile_ple_failed": ("profile", "ple"),
    "profile_gate_rejected": ("profile_gate", None),
    "fit_linear": ("fit", "linear"),
    "fit_linear_failed": ("fit", "linear"),
    "fit_ple": ("fit", "ple"),
    "fit_ple_failed": ("fit", "ple"),
    "prediction_linear": ("prediction", "linear"),
    "prediction_linear_failed": ("prediction", "linear"),
    "prediction_ple": ("prediction", "ple"),
    "prediction_ple_failed": ("prediction", "ple"),
    "complete": ("complete", None),
}
E540_COHORT_STATUSES = frozenset(
    {
        "profile_pending",
        "profile_failed",
        "profile_gate_rejected",
        "fit_running",
        "fit_failed",
        "prediction_failed",
        "complete_ungraded",
        "failed",
    }
)
E540_ARM_STATUSES = frozenset(
    {
        "not_started",
        "profile_running",
        "profile_pass",
        "profile_failed",
        "not_started_due_to_cohort_failure",
        "fit_running",
        "fit_pass",
        "fit_failed",
        "prediction_running",
        "prediction_pass",
        "prediction_failed",
    }
)


def resolve_worker_timeout_seconds(
    declared_seconds: int,
    kernel_deadline_value: str | None,
    *,
    now_monotonic: float | None = None,
) -> int:
    """Recompute an E540 worker timeout at launch; legacy specs retain their limit."""
    if (
        isinstance(declared_seconds, bool)
        or not isinstance(declared_seconds, int)
        or declared_seconds < 1
    ):
        raise ValueError("declared worker timeout must be a positive integer")
    if kernel_deadline_value is None:
        return declared_seconds
    try:
        deadline = float(kernel_deadline_value)
    except (TypeError, ValueError) as error:
        raise ValueError("kernel deadline must be a finite monotonic timestamp") from error
    now = time.monotonic() if now_monotonic is None else now_monotonic
    if (
        isinstance(now, bool)
        or not isinstance(now, (int, float))
        or not math.isfinite(deadline)
        or not math.isfinite(now)
        or now < 0
    ):
        raise ValueError("kernel deadline and current monotonic time must be finite")
    available = math.floor(deadline - now - WORKER_TIMEOUT_FINALIZATION_RESERVE_SECONDS)
    resolved = min(declared_seconds, WORKER_TIMEOUT_CEILING_SECONDS, available)
    if resolved < 1:
        raise TimeoutError("kernel deadline leaves no worker time after finalization reserve")
    return resolved


def gpu_inventory() -> list[dict]:
    """Measure physical-device total and occupied memory independently of PyTorch."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    records = []
    for line in output.strip().splitlines():
        index, uuid, name, total, used = (part.strip() for part in line.split(","))
        records.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "total_bytes": int(total) * 1024**2,
                "used_bytes": int(used) * 1024**2,
            }
        )
    return records


def host_inventory() -> dict:
    """Respect container memory limits as well as visible physical host capacity."""
    memory = psutil.virtual_memory()
    capacity = memory.total
    available = memory.available
    path = Path("/sys/fs/cgroup/memory.max")
    if path.exists() and path.read_text().strip() != "max":
        capacity = min(capacity, int(path.read_text().strip()))
        current = Path("/sys/fs/cgroup/memory.current")
        if current.exists():
            available = min(available, capacity - int(current.read_text().strip()))
    versions = {}
    for name in ("torch", "numpy", "pyarrow", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    return {
        "capacity_bytes": capacity,
        "available_bytes": min(available, capacity),
        "affinity": sorted(os.sched_getaffinity(0)),
        "versions": versions,
        "python": sys.version,
    }


def admit_specs(specs: list[dict], host: dict, devices: list[dict]) -> int:
    """Reserve host capacity for at most two concurrent workers before launch."""
    if not devices or len(devices) > 2 or any("T4" not in d["name"] for d in devices):
        raise ValueError("This runner admits only one or two measured T4 devices")
    for spec in specs:
        limits = spec["limits"]
        if limits["cpus"] > len(host["affinity"]):
            raise ValueError("Worker CPU reservation exceeds available affinity")
        if any(limits["gpu_stop_bytes"] >= device["total_bytes"] for device in devices):
            raise ValueError("Worker stop limit must fit physical device")
    for device in devices:
        if device["used_bytes"] > 128 * 1024**2:
            raise ValueError("GPU has unexplained preexisting memory use")
    host_budget = min(
        int(host["capacity_bytes"] * 0.8),
        host["available_bytes"] - int(host["capacity_bytes"] * 0.2),
    )
    if host_budget <= 0 or any(s["limits"]["ram_bytes"] > host_budget for s in specs):
        raise ValueError("Worker reservation exceeds host operational budget")
    # Concurrent combinations can differ from input order; reserve worst two.
    memories = sorted((s["limits"]["ram_bytes"] for s in specs), reverse=True)
    cpus = sorted((s["limits"]["cpus"] for s in specs), reverse=True)
    if sum(memories[:2]) <= host_budget and sum(cpus[:2]) <= len(host["affinity"]):
        return len(devices)
    return 1


def process_tree_rss(pid: int) -> int:
    """Count a worker's live descendants without counting shared file caches."""
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def stop_group(process: subprocess.Popen) -> None:
    """Terminate only the admitted worker process group, including descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def export_artifacts(root: Path, spec: dict, output: Path) -> dict:
    """Record declared outputs and missing files; never export undeclared directories."""
    import shutil

    result = {"files": [], "missing": []}
    for name in spec["artifacts"]:
        path = root / relative_path(name)
        if not path.exists():
            result["missing"].append(name)
            continue
        entries = sorted(path.rglob("*")) if path.is_dir() else [path]
        for entry in entries:
            if not entry.is_file():
                continue
            if not entry.resolve().is_relative_to(root.resolve()):
                raise ValueError("Output artifact escapes worker root")
            relative = relative_path(entry.relative_to(root).as_posix())
            destination = output / "artifacts" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry, destination)
            result["files"].append(
                {
                    "path": relative,
                    "size": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
    return result


def _read_bounded_json_object(path: Path) -> dict[str, object] | None:
    """Read a small atomic receipt without following links or exposing errors."""
    if path.is_symlink() or path.parent.is_symlink() or path.parent.parent.is_symlink():
        return None
    try:
        with path.open("rb") as source:
            payload = source.read(E540_PHASE_RECEIPT_BYTES_MAX + 1)
    except OSError:
        return None
    if len(payload) > E540_PHASE_RECEIPT_BYTES_MAX:
        return None

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON number: {value}")

    try:
        value = json.loads(payload, parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _finite_float(value: object) -> float | None:
    """Convert only finite, non-boolean JSON numbers without overflowing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _bounded_float(value: object, upper_bound: float) -> float | None:
    """Accept only finite non-boolean values in a small nonnegative range."""
    number = _finite_float(value)
    if number is None or not 0 <= number <= upper_bound:
        return None
    return round(number, 3)


def _bounded_update_count(value: object) -> int | None:
    """Accept only integer optimizer counters within the frozen E540 scale."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= E540_PHASE_UPDATE_COUNT_MAX
    ):
        return None
    return value


def _e540_worker_phase(root: Path, *, now_unix: float | None = None) -> dict[str, object]:
    """Project the atomic E540 V7 receipts onto a strict live-safe field allowlist."""
    cohort = _read_bounded_json_object(root / "e540_cohort" / "cohort.json")
    if (
        cohort is None
        or cohort.get("packet_id") != E540_V7_PACKET_ID
        or cohort.get("precision") != "fp32"
    ):
        return {"status": "unavailable"}

    phase = cohort.get("phase")
    cohort_status = cohort.get("status")
    started_unix = cohort.get("started_unix")
    current_unix = time.time() if now_unix is None else now_unix
    started_number = _finite_float(started_unix)
    current_number = _finite_float(current_unix)
    if (
        not isinstance(phase, str)
        or phase not in E540_PHASES
        or not isinstance(cohort_status, str)
        or cohort_status not in E540_COHORT_STATUSES
        or started_number is None
        or current_number is None
    ):
        return {"status": "unavailable"}
    elapsed = _bounded_float(
        current_number - started_number,
        E540_PHASE_RECEIPT_AGE_SECONDS_MAX,
    )
    if elapsed is None:
        return {"status": "unavailable"}

    stage, arm_id = E540_PHASES[phase]
    summary: dict[str, object] = {
        "status": "available",
        "stage": stage,
        "arm_id": arm_id,
        "cohort_status": cohort_status,
        "worker_elapsed_seconds": elapsed,
    }
    if arm_id is None:
        return _add_e540_profile_gate_metrics(root, summary)

    arm = _read_bounded_json_object(root / "e540_cohort" / "receipts" / f"{arm_id}.json")
    if arm is None or arm.get("mode") != arm_id:
        return _add_e540_profile_gate_metrics(root, summary)
    arm_status = arm.get("status")
    if isinstance(arm_status, str) and arm_status in E540_ARM_STATUSES:
        summary["arm_status"] = arm_status

    phase_wall: object = None
    updates: object = None
    step_wall_ms: object = None
    if stage == "profile":
        profile = arm.get("profile")
        if isinstance(profile, dict):
            phase_wall = profile.get("profile_wall_seconds")
            step_wall_ms = profile.get("step_wall_ms")
            evidence = profile.get("profile_update_evidence")
            if isinstance(evidence, dict):
                updates = evidence.get("optimizer_completed_updates")
    elif stage == "fit":
        phase_wall = arm.get("fit_wall_seconds")
        diagnostics = arm.get("fit_optimizer_diagnostics")
        if isinstance(diagnostics, dict):
            updates = diagnostics.get("optimizer_completed_updates")
    elif stage == "prediction":
        phase_wall = arm.get("prediction_wall_seconds")

    bounded_wall = _bounded_float(phase_wall, E540_PHASE_METRIC_SECONDS_MAX)
    if bounded_wall is not None:
        summary["phase_wall_seconds"] = bounded_wall
    bounded_updates = _bounded_update_count(updates)
    if bounded_updates is not None:
        summary["optimizer_completed_updates"] = bounded_updates
    bounded_step_wall = _bounded_float(step_wall_ms, E540_PHASE_STEP_WALL_MS_MAX)
    if bounded_step_wall is not None:
        summary["step_wall_ms"] = bounded_step_wall
    return _add_e540_profile_gate_metrics(root, summary)


def _add_e540_profile_gate_metrics(root: Path, summary: dict[str, object]) -> dict[str, object]:
    """Add only finite, bounded profile-gate measurements when that receipt exists."""
    gate = _read_bounded_json_object(root / "e540_cohort" / "profile_gate.json")
    if gate is None:
        return summary
    status = gate.get("status")
    if isinstance(status, str) and status in {"pass", "reject"}:
        summary["profile_gate_status"] = status
    for field, limit in (
        ("projected_total_worker_seconds", E540_PHASE_PROJECTION_SECONDS_MAX),
        ("maximum_worker_seconds", E540_PHASE_METRIC_SECONDS_MAX),
        ("worker_elapsed_seconds_at_gate", E540_PHASE_RECEIPT_AGE_SECONDS_MAX),
    ):
        value = _bounded_float(gate.get(field), limit)
        if value is not None:
            summary[field] = value
    return summary


def run(root: Path, spec_paths: list[Path], output: Path, stagger_seconds: float = 30) -> None:
    """Run fixed jobs with one process per device and a fail-closed external watchdog."""
    specs = [read_spec(path) for path in spec_paths]
    if len({s["name"] for s in specs}) != len(specs):
        raise ValueError("Duplicate experiment identity")
    output.mkdir(parents=True, exist_ok=False)
    host, devices = host_inventory(), gpu_inventory()
    write_json(output / "inventory.json", {"host": host, "devices": devices})
    concurrency = admit_specs(specs, host, devices)
    print(
        json.dumps(
            {
                "event": "validating_inputs",
                "jobs": [spec["name"] for spec in specs],
                "concurrency_admitted": concurrency,
            }
        ),
        flush=True,
    )
    # Validate sealed code locally. Explicitly deferred mounted data receive a
    # streaming portable-registry check in the recipe worker before Torch loads.
    for spec in specs:
        raw_deferred = spec["environment"].get("WNN_DEFERRED_IDENTITY_PATHS", "")
        deferred_paths = {relative_path(value) for value in raw_deferred.split(",") if value}
        input_paths = {item["path"] for item in spec["inputs"]}
        code_paths = {item["path"] for item in spec["code"]}
        if deferred_paths & code_paths or not deferred_paths.issubset(input_paths):
            raise ValueError("deferred identity paths must name declared mounted inputs only")
        verify_files(
            root,
            spec["code"] + [item for item in spec["inputs"] if item["path"] not in deferred_paths],
            identity_receipt=output / f"{spec['name']}.input_identity.json",
        )
        if deferred_paths:
            receipt_path = output / f"{spec['name']}.input_identity.json"
            receipt = json.loads(receipt_path.read_text())
            for name in sorted(deferred_paths):
                descriptor = next(item for item in spec["inputs"] if item["path"] == name)
                path = root / name
                observed_size = path.stat().st_size if path.is_file() else None
                matches = path.is_symlink() and observed_size == descriptor["size"]
                receipt["observations"].append(
                    {
                        "path": name,
                        "expected": {"size": descriptor["size"], "sha256": descriptor["sha256"]},
                        "observed": {"size": observed_size, "sha256": None},
                        "status": "deferred_to_worker_preflight" if matches else "mount_mismatch",
                    }
                )
                if not matches:
                    receipt["status"] = "failed"
                    write_json(receipt_path, receipt)
                    raise ValueError(f"Deferred mounted input size mismatch: {name}")
            receipt["status"] = "verified_internal_deferred_external"
            write_json(receipt_path, receipt)
    active, finished = [], []
    queue = list(specs)
    last_launch = -float("inf")
    last_heartbeat = -float("inf")
    incident = None
    try:
        while queue or active:
            now = time.monotonic()
            if queue and len(active) < concurrency and now - last_launch >= stagger_seconds:
                occupied = {job["device"]["index"] for job in active}
                device = next(d for d in devices if d["index"] not in occupied)
                spec = queue.pop(0)
                job_output = output / spec["name"]
                job_output.mkdir()
                env = os.environ.copy()
                env.update(spec["environment"])
                count = spec["limits"]["cpus"]
                used_cpus = {cpu for job in active for cpu in job["cpus"]}
                cpus = [cpu for cpu in host["affinity"] if cpu not in used_cpus][:count]
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": device["uuid"],
                        "OMP_NUM_THREADS": str(count),
                        "MKL_NUM_THREADS": str(count),
                        "OPENBLAS_NUM_THREADS": str(count),
                        "WNN_THREADS": str(count),
                        "NUMBA_NUM_THREADS": str(count),
                        "POLARS_MAX_THREADS": str(count),
                        "ARROW_NUM_THREADS": str(count),
                        "WNN_GPU_MEMORY_BYTES": str(spec["limits"]["gpu_memory_bytes"]),
                        "WNN_GPU_FRACTION": str(
                            spec["limits"]["gpu_memory_bytes"] / device["total_bytes"]
                        ),
                        "PYTHONPATH": str(root / "src"),
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONHASHSEED": spec["environment"].get("PYTHONHASHSEED", "0"),
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    }
                )
                argv = [sys.executable, *spec["argv"][1:]]
                log = (job_output / "worker.log").open("w")
                declared_timeout = spec["limits"]["timeout_seconds"]
                deadline_value = os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC")
                launch_monotonic = time.monotonic()
                resolved_timeout = resolve_worker_timeout_seconds(
                    declared_timeout,
                    deadline_value,
                    now_monotonic=launch_monotonic,
                )
                spec["limits"]["timeout_seconds"] = resolved_timeout
                timeout_resolution = {
                    "status": "resolved_before_worker_launch",
                    "declared_spec_timeout_seconds": declared_timeout,
                    "resolved_worker_timeout_seconds": resolved_timeout,
                    "kernel_deadline_monotonic": deadline_value,
                    "computed_at_monotonic": launch_monotonic,
                    "finalization_reserve_seconds": (
                        WORKER_TIMEOUT_FINALIZATION_RESERVE_SECONDS
                        if deadline_value is not None
                        else None
                    ),
                    "scope": (
                        "inherited kernel-process deadline"
                        if deadline_value is not None
                        else "legacy spec timeout; no kernel deadline inherited"
                    ),
                }
                process = subprocess.Popen(
                    argv,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    preexec_fn=lambda cpus=cpus: os.sched_setaffinity(0, cpus),
                )
                active.append(
                    {
                        "process": process,
                        "spec": spec,
                        "device": device,
                        "cpus": cpus,
                        "output": job_output,
                        "log": log,
                        "e540_phase_root": (root if spec["name"] == E540_V7_SPEC_NAME else None),
                        "started": launch_monotonic,
                        "worker_timeout_seconds": resolved_timeout,
                        "peak_ram_bytes": 0,
                        "peak_gpu_bytes": 0,
                    }
                )
                # Persist after Popen so the timeout computation stays adjacent to
                # launch; the active watchdog already holds this same resolved value.
                write_json(job_output / "spec.json", spec)
                write_json(job_output / "timeout_resolution.json", timeout_resolution)
                last_launch = launch_monotonic
            samples = {device["index"]: device for device in gpu_inventory()}
            total_rss = 0
            for job in list(active):
                process, spec = job["process"], job["spec"]
                rss = process_tree_rss(process.pid)
                used = samples[job["device"]["index"]]["used_bytes"]
                total_rss += rss
                job["peak_ram_bytes"] = max(job["peak_ram_bytes"], rss)
                job["peak_gpu_bytes"] = max(job["peak_gpu_bytes"], used)
                limits = spec["limits"]
                if rss > limits["ram_bytes"] or used > limits["gpu_stop_bytes"]:
                    incident = f"Resource stop for {spec['name']}"
                # Allocator bytes exclude CUDA context/runtime memory. Preserve a
                # separate 12GiB observed operating target below the hard stop.
                elif used > min(12 * 1024**3, limits["gpu_stop_bytes"] - 1024**2) + 1024**2:
                    incident = f"GPU operating target exceeded for {spec['name']}"
                elif now - job["started"] > limits["timeout_seconds"]:
                    incident = f"Timeout for {spec['name']}"
                if incident:
                    raise RuntimeError(incident)
                if process.poll() is not None:
                    job["log"].close()
                    record = {
                        "name": spec["name"],
                        "returncode": process.returncode,
                        "wall_seconds": now - job["started"],
                        "worker_timeout_seconds": spec["limits"]["timeout_seconds"],
                        "device": job["device"],
                        "peak_ram_bytes": job["peak_ram_bytes"],
                        "peak_gpu_bytes": job["peak_gpu_bytes"],
                        "artifacts": export_artifacts(root, spec, job["output"]),
                    }
                    record["status"] = (
                        "complete"
                        if process.returncode == 0 and not record["artifacts"]["missing"]
                        else "failed"
                    )
                    write_json(job["output"] / "result.json", record)
                    finished.append(record)
                    active.remove(job)
                    if record["status"] == "failed":
                        raise RuntimeError(f"Worker failed: {spec['name']}")
            if total_rss > host["capacity_bytes"] * 0.9:
                raise RuntimeError("Aggregate host stop threshold exceeded")
            if now - last_heartbeat >= 30:
                heartbeat = resource_heartbeat(active, finished, queue, now)
                write_json(output / "heartbeat.json", heartbeat)
                print(json.dumps(heartbeat), flush=True)
                last_heartbeat = now
            time.sleep(0.25)
    except BaseException as exc:
        incident = f"{type(exc).__name__}: {exc}"
        for job in active:
            stop_group(job["process"])
            job["log"].close()
            write_json(
                job["output"] / "failure.json",
                {
                    "incident": incident,
                    "peak_ram_bytes": job["peak_ram_bytes"],
                    "peak_gpu_bytes": job["peak_gpu_bytes"],
                    "worker_timeout_seconds": job["spec"]["limits"]["timeout_seconds"],
                    "artifacts": export_artifacts(root, job["spec"], job["output"]),
                },
            )
        raise
    finally:
        write_json(
            output / "summary.json",
            {
                "finished": finished,
                "incident": incident,
                "not_started": [spec["name"] for spec in queue],
                "concurrency_admitted": concurrency,
                "watchdog_interval_seconds": 0.25,
                "limitation": "GPU/RSS external watchdog is sampled; it is not a provider cgroup hard cap",
            },
        )


def resource_heartbeat(
    active: list[dict], finished: list[dict], queue: list[dict], now: float
) -> dict:
    """Expose only whitelisted operational fields to private provider stdout."""
    active_rows = []
    for job in active:
        row = {
            "name": job["spec"]["name"],
            "gpu_index": job["device"]["index"],
            "gpu_uuid": job["device"]["uuid"],
            "elapsed_seconds": round(max(0, now - job["started"]), 3),
            "worker_timeout_seconds": job["worker_timeout_seconds"],
            "peak_ram_bytes": job["peak_ram_bytes"],
            "peak_gpu_bytes": job["peak_gpu_bytes"],
        }
        phase_root = job.get("e540_phase_root")
        if isinstance(phase_root, Path):
            row["worker_phase"] = _e540_worker_phase(phase_root)
        active_rows.append(row)
    return {
        "event": "campaign_resource_heartbeat",
        "active": active_rows,
        "finished": [{"name": job["name"], "status": job["status"]} for job in finished],
        "not_started": [spec["name"] for spec in queue],
        "scope": "measured worker resource activity; not proof of epoch progress or scientific success",
    }


def main() -> None:
    """Execute an admitted, private two-T4 dependency bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.spec, args.output)


if __name__ == "__main__":
    main()
