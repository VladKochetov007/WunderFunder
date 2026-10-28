"""Own one bounded real-source engineering profile per fixed panel GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def publish(path: Path, value: dict[str, object]) -> None:
    """Atomically retain profile phase state and exact owned process identities."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def start_ticks(pid: int) -> str:
    """Identify one Linux process incarnation rather than trusting a recycled PID."""
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def parse_cpus(value: str) -> set[int]:
    """Parse the panel's explicit contiguous eight-physical-core range."""
    first, last = map(int, value.split("-"))
    if last - first != 7 or not 0 <= first <= last < 96:
        raise ValueError("Expected eight physical CPU cores in the fixed panel")
    return set(range(first, last + 1))


def main() -> None:
    """Wait for verified shared inputs and launch eight discard-only profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state: dict[str, object] = {
        "schema": "tabdata.real-source-profile-cohort.v1",
        "pid": os.getpid(),
        "status": "waiting_for_data_preparation",
        "scientific_fits_started": 0,
        "source_root": str(source),
        "jobs": [],
    }
    receipt = output / "cohort.json"
    publish(receipt, state)
    deadline = time.monotonic() + 14_400
    preparation = args.preparation_root / "preparation.json"
    while True:
        prepared = json.loads(preparation.read_text())
        if prepared["status"] == "failed":
            raise RuntimeError(f"Preparation failed: {prepared.get('failed_phase')}")
        if prepared["status"] == "ready_for_actual_data_profiles":
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Data preparation did not complete within four hours")
        time.sleep(15)
    split = Path(str(prepared["split_manifest"]))
    if hashlib.sha256(split.read_bytes()).hexdigest() != prepared["split_sha256"]:
        raise ValueError("Prepared chronological split changed")
    packet = source / "research/experiment_drafts/galam-l4-quality8-20261028/launch"
    panel = json.loads((packet / "PANEL.json").read_text())
    cells = panel["cells"]
    if len(cells) != 8 or [c["gpu"] for c in cells] != list(range(8)):
        raise ValueError("Expected exactly the fixed eight ordered GPUs")
    processes: list[tuple[subprocess.Popen[bytes], dict[str, object]]] = []
    jobs: list[dict[str, object]] = []
    for cell in cells:
        config = (packet / cell["config"]).resolve()
        if (
            not config.is_relative_to(packet)
            or hashlib.sha256(config.read_bytes()).hexdigest() != cell["config_file_sha256"]
        ):
            raise ValueError("Panel configuration changed or escaped packet")
        raw = json.loads(config.read_text())
        cadence = raw["monitoring"]["diagnostics_every_steps"]
        updates = max(
            cadence, math.ceil(math.ceil(20_000 / raw["chunk_rows"]) / raw["accumulate_chunks"]) + 1
        )
        if updates != cell["profile_updates"] or raw.get("input_prefetch_batches") != 1:
            raise ValueError("Profile must consume a prefetched successor and sample diagnostics")
        environment = dict(
            os.environ,
            PYTHONPATH=str(source / "src"),
            CUDA_VISIBLE_DEVICES=str(cell["gpu"]),
            OMP_NUM_THREADS="8",
            MKL_NUM_THREADS="8",
            OPENBLAS_NUM_THREADS="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        command = [
            sys.executable,
            "scripts/train_sequence.py",
            "--config",
            str(config),
            "--data-root",
            str(args.data_root.resolve()),
            "--split-manifest",
            str(split),
            "--output",
            str(output / cell["id"]),
            "--profile-updates",
            str(updates),
        ]
        cpus = parse_cpus(cell["cpu_affinity"])
        with (output / f"{cell['id']}.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=source,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=lambda cpus=cpus: os.sched_setaffinity(0, cpus),
            )
        job: dict[str, object] = {
            "id": cell["id"],
            "gpu": cell["gpu"],
            "pid": process.pid,
            "start_ticks": start_ticks(process.pid),
            "command": command,
            "config_sha256": cell["config_file_sha256"],
            "status": "running",
        }
        jobs.append(job)
        processes.append((process, job))
        state.update(status="profiles_running", jobs=jobs)
        publish(receipt, state)
    deadline = time.monotonic() + 3_600
    while any(process.poll() is None for process, _ in processes):
        for process, job in processes:
            code = process.poll()
            if code is not None:
                job.update(status="exited", exit_code=code)
            elif time.monotonic() > deadline and start_ticks(process.pid) == job["start_ticks"]:
                os.killpg(process.pid, signal.SIGTERM)
                job.update(status="profile_timeout_terminated")
        publish(receipt, state)
        time.sleep(10)
    for process, job in processes:
        job["exit_code"] = process.wait()
        job["status"] = "exited"
    state["status"] = "profiles_finished_pending_review"
    publish(receipt, state)


if __name__ == "__main__":
    main()
