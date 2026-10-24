"""Observe immutable epoch checkpoints without changing live training or selection."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    """Atomically publish observer state outside the scientific source."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def capture(source: Path, directory: Path) -> Path | None:
    """Hardlink the completed atomic checkpoint inode, never its mutable filename."""
    if not source.is_file():
        return None
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < 225 * (1 << 30):
        write_json(
            directory.parent.parent / "DISK_ADMISSION_STOP.json",
            {
                "free_bytes": shutil.disk_usage(directory).free,
                "time_unix": time.time(),
                "policy": "stop retaining new checkpoints; leave training untouched",
            },
        )
        return None
    temporary = directory / f"capture-{time.time_ns()}.tmp"
    try:
        os.link(source, temporary)
    except FileNotFoundError:
        return None
    stat = temporary.stat()
    target = directory / f"inode-{stat.st_ino}-mtime-{stat.st_mtime_ns}.pt"
    if target.exists():
        temporary.unlink()
    else:
        temporary.replace(target)
    return target


def prepare_jobs(plan: dict, root: Path) -> list[dict]:
    """Freeze a hash-selected fitting probe before observing validation outcomes."""
    jobs = []
    for index, original in enumerate(plan["jobs"]):
        job = dict(original)
        split = json.loads(Path(job["split_manifest"]).read_text())
        fit = [(source, group) for source, groups in split["fit"].items() for group in groups]
        ordered = sorted(
            fit,
            key=lambda pair: hashlib.sha256(
                f"pr32-train-probe-v1:{job['split_sha256']}:{pair[0]}:{pair[1]}".encode()
            ).digest(),
        )
        job.update(
            source_root=plan["source"]["root"],
            source_commit=plan["source"]["commit"],
            data_root=plan["data_root"],
            cell=job["name"],
            train_probe_members=[list(member) for member in ordered[:64]],
            selection_policy="diagnostic only; original fixed epoch30 endpoint unchanged",
            initial_capture_manifest=str(root / "INITIAL_CAPTURE.json"),
        )
        job_path = root / f"{job['name']}-job.json"
        if job_path.exists() and json.loads(job_path.read_text()) != job:
            raise ValueError("observer job changed after freezing")
        write_json(job_path, job)
        job["job_path"] = str(job_path)
        job["observer_gpu"] = job["gpu_uuids"][0]
        job["observer_cpus"] = f"{64 + index * 2},{65 + index * 2}"
        jobs.append(job)
    return jobs


def launch(job: dict, checkpoint: Path, output: Path, runtime: Path, seconds: int) -> dict:
    """Start an inference-only process with its own memory and lifetime bounds."""
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=job["observer_gpu"],
        PYTHONPATH=f"{runtime}:{job['source_root']}/src",
        CUBLAS_EMULATE_SINGLE_PRECISION="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        TORCHINDUCTOR_CACHE_DIR=str(output.parent / "compile-cache"),
        TMPDIR=str(output.parent / "scratch"),
        PYTHONUNBUFFERED="1",
    )
    Path(environment["TMPDIR"]).mkdir(exist_ok=True)
    command = [
        "taskset",
        "-c",
        job["observer_cpus"],
        sys.executable,
        "-m",
        "validation_monitor.evaluate",
        "--job",
        job["job_path"],
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--batch-size",
        "2",
        "--max-gpu-gib",
        "16",
        "--max-seconds",
        str(seconds),
    ]
    with (output / "worker.log").open("wb") as stream:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {
        "process": process,
        "started": time.monotonic(),
        "output": output,
        "checkpoint": checkpoint,
        "command": command,
    }


def stop_worker(worker: dict) -> None:
    """Terminate only the observer's own still-live child group."""
    process = worker["process"]
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def run(plan: dict, root: Path, runtime: Path, hours: float, seconds: int) -> None:
    """Capture every new safe epoch and evaluate each cell serially, cells in parallel."""
    jobs = prepare_jobs(plan, root)
    initial = json.loads((root / "INITIAL_CAPTURE.json").read_text())
    for item in initial["captured"]:
        capture(Path(item["path"]), root / "snapshots" / item["cell"])
    workers = {}
    deadline = time.monotonic() + hours * 3600
    try:
        while time.monotonic() < deadline:
            status = []
            all_final = True
            for job in jobs:
                cell = job["name"]
                directory = root / "snapshots" / cell
                capture(Path(plan["run_root"]) / cell / "fit/training_state.pt", directory)
                worker = workers.get(cell)
                if worker is not None:
                    process = worker["process"]
                    if (
                        process.poll() is None
                        and time.monotonic() - worker["started"] > seconds + 60
                    ):
                        stop_worker(worker)
                    if process.poll() is not None:
                        write_json(
                            worker["output"] / "WORKER_EXIT.json",
                            {
                                "exit_code": process.returncode,
                                "checkpoint": str(worker["checkpoint"]),
                                "command": worker["command"],
                                "ended_unix": time.time(),
                            },
                        )
                        workers.pop(cell)
                pending = []
                for snapshot in sorted(
                    directory.glob("inode-*.pt"), key=lambda p: p.stat().st_mtime_ns
                ):
                    output = root / "evaluations" / cell / snapshot.stem
                    if not output.exists():
                        pending.append((snapshot, output))
                if cell not in workers and pending:
                    snapshot, output = pending[0]
                    workers[cell] = launch(job, snapshot, output, runtime, seconds)
                final_results = list((root / "evaluations" / cell).glob("*/RESULT.json"))
                complete30 = any(
                    json.loads(path.read_text()).get("completed_epochs") == 30
                    and json.loads(path.read_text()).get("status") == "complete"
                    for path in final_results
                )
                all_final = all_final and complete30
                status.append(
                    {
                        "cell": cell,
                        "pid": workers[cell]["process"].pid if cell in workers else None,
                        "pending_snapshots": len(pending),
                        "completed_results": len(final_results),
                        "final30_evaluated": complete30,
                    }
                )
            write_json(
                root / "WATCH_STATE.json",
                {
                    "pid": os.getpid(),
                    "updated_unix": time.time(),
                    "cells": status,
                    "status": "complete" if all_final else "watching",
                    "policy": "diagnostic; no early stopping or checkpoint selection changes",
                },
            )
            if all_final:
                return
            time.sleep(30)
    finally:
        for worker in workers.values():
            stop_worker(worker)


def main() -> None:
    """Claim one cron-owned observer for a bounded 36-hour watch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=36)
    parser.add_argument("--worker-seconds", type=int, default=3600)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "WATCH.lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (args.root / "WATCH_FINISHED.json").exists():
            return
        run(
            json.loads(args.plan.read_text()),
            args.root,
            Path(__file__).resolve().parent.parent,
            args.hours,
            args.worker_seconds,
        )
        write_json(args.root / "WATCH_FINISHED.json", {"ended_unix": time.time()})


if __name__ == "__main__":
    main()
