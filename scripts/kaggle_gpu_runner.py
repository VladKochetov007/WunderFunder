from __future__ import annotations

import argparse
import importlib.metadata
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
from kaggle_common import read_spec, relative_path, sha256_file, verify_files, write_json


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


def run(root: Path, spec_paths: list[Path], output: Path, stagger_seconds: float = 30) -> None:
    """Run fixed jobs with one process per device and a fail-closed external watchdog."""
    specs = [read_spec(path) for path in spec_paths]
    if len({s["name"] for s in specs}) != len(specs):
        raise ValueError("Duplicate experiment identity")
    output.mkdir(parents=True, exist_ok=False)
    host, devices = host_inventory(), gpu_inventory()
    write_json(output / "inventory.json", {"host": host, "devices": devices})
    concurrency = admit_specs(specs, host, devices)
    # Validate the complete batch before any worker can start.
    for spec in specs:
        verify_files(root, spec["code"] + spec["inputs"])
    active, finished = [], []
    queue = list(specs)
    last_launch = -float("inf")
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
                        "WNN_GPU_FRACTION": str(
                            spec["limits"]["gpu_memory_bytes"] / device["total_bytes"]
                        ),
                        "PYTHONPATH": str(root / "src"),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    }
                )
                log = (job_output / "worker.log").open("w")
                argv = [sys.executable, *spec["argv"][1:]]
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
                        "started": now,
                        "peak_ram_bytes": 0,
                        "peak_gpu_bytes": 0,
                    }
                )
                write_json(job_output / "spec.json", spec)
                last_launch = now
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
                elif used > limits["gpu_memory_bytes"]:
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
