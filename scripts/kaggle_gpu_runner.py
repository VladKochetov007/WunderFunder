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

# Provider wall time is the outer bound. A worker's declared timeout remains
# authoritative, preserving historical specs while allowing new fits to use the
# current Kaggle session window instead of the withdrawn 5,200-second cap.
WORKER_TIMEOUT_CEILING_SECONDS = 43_200
WORKER_TIMEOUT_FINALIZATION_RESERVE_SECONDS = 300
HOST_OPERATIONAL_HEADROOM_BYTES = 1_000_000_000


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


def _read_cgroup_integer(path: Path) -> int | None:
    """Read a finite cgroup-v2 memory value, returning None when unavailable."""
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _read_cgroup_memory_stat(path: Path) -> dict[str, int]:
    """Read only bounded memory.stat counters needed to identify reclaimable cache."""
    wanted = {"anon", "file", "shmem", "active_file", "inactive_file"}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError:
        return {}
    stats: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2 or parts[0] not in wanted:
            continue
        try:
            value = int(parts[1])
        except ValueError:
            continue
        if value >= 0:
            stats[parts[0]] = value
    return stats


def _memory_capacity_snapshot(
    physical_capacity_bytes: int,
    mem_available_bytes: int,
    cgroup_limit_bytes: int | None,
    cgroup_current_bytes: int | None,
    cgroup_memory_stat: dict[str, int],
) -> dict[str, int | None]:
    """Combine Linux available RAM with cgroup headroom after inactive-file reclaim."""
    capacity = physical_capacity_bytes
    cgroup_free: int | None = None
    cgroup_reclaimable_available: int | None = None
    if cgroup_limit_bytes is not None:
        capacity = min(capacity, cgroup_limit_bytes)
        if cgroup_current_bytes is not None:
            used = min(cgroup_current_bytes, capacity)
            cgroup_free = max(0, capacity - used)
            reclaimable_file = min(used, cgroup_memory_stat.get("inactive_file", 0))
            cgroup_reclaimable_available = min(capacity, cgroup_free + reclaimable_file)

    available = min(mem_available_bytes, capacity)
    if cgroup_reclaimable_available is not None:
        available = min(available, cgroup_reclaimable_available)
    return {
        "capacity_bytes": capacity,
        "mem_available_bytes": mem_available_bytes,
        "cgroup_limit_bytes": cgroup_limit_bytes,
        "cgroup_current_bytes": cgroup_current_bytes,
        "cgroup_free_bytes": cgroup_free,
        "cgroup_reclaimable_available_bytes": cgroup_reclaimable_available,
        "cgroup_anon_bytes": cgroup_memory_stat.get("anon"),
        "cgroup_file_bytes": cgroup_memory_stat.get("file"),
        "cgroup_shmem_bytes": cgroup_memory_stat.get("shmem"),
        "cgroup_active_file_bytes": cgroup_memory_stat.get("active_file"),
        "cgroup_inactive_file_bytes": cgroup_memory_stat.get("inactive_file"),
        "available_bytes": available,
    }


def host_inventory() -> dict:
    """Measure MemAvailable and cgroup usage, separating reclaimable file cache."""
    memory = psutil.virtual_memory()
    cgroup = Path("/sys/fs/cgroup")
    cgroup_limit = _read_cgroup_integer(cgroup / "memory.max")
    cgroup_current = _read_cgroup_integer(cgroup / "memory.current")
    cgroup_stat = _read_cgroup_memory_stat(cgroup / "memory.stat")
    memory_snapshot = _memory_capacity_snapshot(
        memory.total,
        memory.available,
        cgroup_limit,
        cgroup_current,
        cgroup_stat,
    )
    versions = {}
    for name in ("torch", "numpy", "pyarrow", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    return {
        **memory_snapshot,
        "affinity": sorted(os.sched_getaffinity(0)),
        "versions": versions,
        "python": sys.version,
    }


def admit_specs(specs: list[dict], host: dict, devices: list[dict]) -> int:
    """Reserve measured host RAM, one GB headroom, and at most two T4 workers."""
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
        host["available_bytes"] - HOST_OPERATIONAL_HEADROOM_BYTES,
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
    return {
        "event": "campaign_resource_heartbeat",
        "active": [
            {
                "name": job["spec"]["name"],
                "gpu_index": job["device"]["index"],
                "gpu_uuid": job["device"]["uuid"],
                "elapsed_seconds": round(max(0, now - job["started"]), 3),
                "worker_timeout_seconds": job["worker_timeout_seconds"],
                "peak_ram_bytes": job["peak_ram_bytes"],
                "peak_gpu_bytes": job["peak_gpu_bytes"],
            }
            for job in active
        ],
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
