#!/usr/bin/env python3
"""Run E528-A2 under explicit limits and retain an external RSS/exit receipt."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
LINEAGE = SCRIPT.parent
ROOT = SCRIPT.parents[4]
BENCHMARK = LINEAGE / "benchmark_feature_reuse_v2.py"
PYTHON = ROOT / ".venv-py311-profile/bin/python"
WALL_CAP_SECONDS = 180
RSS_CAP_BYTES = 1_000_000_000
AS_CAP_BYTES = 1_500_000_000
RSS_POLL_SECONDS = 0.1


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_status(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = (Path("/proc") / str(pid) / "status").read_text().splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return values
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key in {"VmRSS", "VmHWM"}:
            parts = value.split()
            if parts:
                values[key] = int(parts[0]) * 1024
    return values


def descendants(root_pid: int) -> list[int]:
    found: list[int] = []
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        try:
            child_text = (Path("/proc") / str(pid) / "task" / str(pid) / "children").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        children = [int(value) for value in child_text.split()]
        found.extend(children)
        pending.extend(children)
    return found


def kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_attempt(phase: str) -> int:
    if phase not in {"freeze", "run"}:
        raise ValueError(f"unsupported phase: {phase}")
    executable_paths = ["/usr/bin/timeout", "/usr/bin/taskset", "/usr/bin/prlimit"]
    for executable in executable_paths:
        if not Path(executable).is_file():
            raise FileNotFoundError(executable)
    if not PYTHON.is_file() or not BENCHMARK.is_file():
        raise FileNotFoundError("profile Python or versioned benchmark helper is absent")

    command = [
        "/usr/bin/timeout", "--signal=TERM", "--kill-after=10s", f"{WALL_CAP_SECONDS}s",
        "/usr/bin/taskset", "-c", "0",
        "/usr/bin/prlimit", f"--as={AS_CAP_BYTES}", f"--cpu={WALL_CAP_SECONDS}",
        f"--rss={RSS_CAP_BYTES}", "--", str(PYTHON), str(BENCHMARK), f"--{phase}",
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LC_ALL": "C.UTF-8",
    })
    log_path = LINEAGE / f"E528_{phase.upper()}_ATTEMPT_A2.log"
    receipt_path = LINEAGE / f"E528_{phase.upper()}_ATTEMPT_A2.json"
    started = utc_now()
    started_clock = time.monotonic()
    max_worker_rss = 0
    max_worker_hwm = 0
    max_process_tree_rss = 0
    sampled_pids: set[int] = set()
    rss_limit_terminated = False
    wall_watchdog_terminated = False

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            current = [process.pid, *descendants(process.pid)]
            sampled_pids.update(current)
            rss_values = [read_status(pid) for pid in current]
            worker_rss_values = [values.get("VmRSS", 0) for values in rss_values]
            worker_hwm_values = [values.get("VmHWM", 0) for values in rss_values]
            tree_rss = sum(worker_rss_values)
            max_worker_rss = max(max_worker_rss, max(worker_rss_values, default=0))
            max_worker_hwm = max(max_worker_hwm, max(worker_hwm_values, default=0))
            max_process_tree_rss = max(max_process_tree_rss, tree_rss)
            elapsed = time.monotonic() - started_clock
            if max_worker_hwm >= RSS_CAP_BYTES:
                rss_limit_terminated = True
                kill_process_group(process)
                break
            if elapsed >= WALL_CAP_SECONDS + 2:
                wall_watchdog_terminated = True
                kill_process_group(process)
                break
            time.sleep(RSS_POLL_SECONDS)
        exit_status = process.wait()

    elapsed_seconds = time.monotonic() - started_clock
    receipt = {
        "experiment_id": "E528",
        "attempt_id": "E528-A2",
        "phase": phase,
        "status": "finished" if exit_status == 0 else "failed_or_guard_terminated",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "elapsed_wall_seconds": elapsed_seconds,
        "argv": command,
        "working_directory": str(ROOT),
        "log_path": str(log_path.relative_to(ROOT)),
        "exit_status": exit_status,
        "rss_guard_terminated": rss_limit_terminated,
        "wall_watchdog_terminated": wall_watchdog_terminated,
        "resource_limits": {
            "cpu_affinity": [0],
            "rlimit_cpu_seconds": WALL_CAP_SECONDS,
            "rlimit_rss_bytes": RSS_CAP_BYTES,
            "rlimit_address_space_bytes": AS_CAP_BYTES,
            "wall_timeout_seconds": WALL_CAP_SECONDS,
            "rss_poll_interval_seconds": RSS_POLL_SECONDS,
            "observed_max_process_vm_rss_bytes": max_worker_rss,
            "observed_max_process_vm_hwm_bytes": max_worker_hwm,
            "observed_max_tree_vm_rss_bytes_sum": max_process_tree_rss,
            "rss_tree_sum_note": "Conservative sum of sampled RSS across supervisor descendants; VmHWM is the kernel per-process high-water mark sampled while alive.",
            "observed_descendant_pids": sorted(sampled_pids),
        },
        "device": "CPU only; CUDA_VISIBLE_DEVICES is empty",
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps({
        "phase": phase,
        "exit_status": exit_status,
        "elapsed_wall_seconds": elapsed_seconds,
        "max_process_vm_hwm_bytes": max_worker_hwm,
        "max_tree_rss_sum_bytes": max_process_tree_rss,
        "log": str(log_path),
        "receipt": str(receipt_path),
    }, indent=2))
    return exit_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "run"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_attempt(args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
