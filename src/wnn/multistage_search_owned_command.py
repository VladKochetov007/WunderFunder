from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from wnn.multistage_search import atomic_write_json
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_runtime import is_same_process, process_identity, terminate_owned_process


def run_owned_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    receipt_path: Path,
    owner: Mapping[str, object],
    stage: int,
    trial_number: int,
    mode: str,
    gpu_key: str,
    output_dir: Path,
    timeout_seconds: int,
    progress: Callable[[], None] | None = None,
) -> tuple[int | None, bool, dict[str, object]]:
    """Run a bounded command in an isolated process group and retain its ownership receipt."""
    if os.name != "posix":
        raise RuntimeError("remote worker ownership requires POSIX process groups and procfs")
    if timeout_seconds <= 0:
        raise ValueError("owned command timeout must be positive")
    encoded_command = json.dumps(list(argv), separators=(",", ":")).encode()
    record: dict[str, object] = {
        "schema": "wnn.multistage-owned-process.v1",
        "stage": stage,
        "trial_number": trial_number,
        "mode": mode,
        "gpu_key": gpu_key,
        "owner": dict(owner),
        "command": list(argv),
        "command_sha256": hashlib.sha256(encoded_command).hexdigest(),
        "cwd": str(cwd.resolve()),
        "output_dir": str(output_dir.resolve()),
        "started_at_utc": utc_now(),
        "started_at_epoch": time.time(),
        "timeout_seconds": timeout_seconds,
        "effective_thread_environment": {
            key: env[key]
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            if key in env
        },
        "child_environment": {
            key: env[key]
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            if key in env
        },
        "cpu_thread_env": {
            key: env[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS") if key in env
        },
        "status": "launching",
    }
    hard_deadline = time.monotonic() + timeout_seconds
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_stream:
        proc = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        child = process_identity(proc.pid)
        if child is None or child.get("pgid") != proc.pid:
            proc.kill()
            proc.wait()
            raise RuntimeError("could not establish owned child PID/birth/PGID")
        record["child"] = child
        record["status"] = "running"
        try:
            atomic_write_json(receipt_path, record)
        except (OSError, TypeError, ValueError):
            terminate_owned_process(child, grace_seconds=0.0)
            proc.wait()
            raise
        timed_out = False
        term_sent = False
        grace = min(10.0, max(0.1, timeout_seconds * 0.05))
        terminate_at = hard_deadline - grace
        while proc.poll() is None:
            if progress is not None:
                try:
                    progress()
                except Exception as error:
                    remaining = max(0.0, hard_deadline - time.monotonic())
                    terminate_owned_process(child, grace_seconds=remaining)
                    try:
                        proc.wait(timeout=max(0.0, hard_deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        if is_same_process(child):
                            os.killpg(int(child["pgid"]), signal.SIGKILL)
                        proc.kill()
                        proc.wait()
                    record.update(
                        {
                            "status": "progress_callback_failed",
                            "error": f"{type(error).__name__}: {error}",
                            "exit_code": proc.returncode,
                            "finished_at_utc": utc_now(),
                            "finished_at_epoch": time.time(),
                        }
                    )
                    atomic_write_json(receipt_path, record)
                    raise
            now = time.monotonic()
            if now >= hard_deadline:
                timed_out = True
                if is_same_process(child):
                    os.killpg(int(child["pgid"]), signal.SIGKILL)
                break
            if now >= terminate_at and not term_sent:
                timed_out = True
                if is_same_process(child):
                    os.killpg(int(child["pgid"]), signal.SIGTERM)
                term_sent = True
            sleep_until = hard_deadline if term_sent else terminate_at
            time.sleep(min(0.1, max(0.01, sleep_until - now)))
        if proc.poll() is None:
            remaining = max(0.0, hard_deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                if is_same_process(child):
                    os.killpg(int(child["pgid"]), signal.SIGKILL)
        try:
            code = proc.wait(timeout=max(0.1, hard_deadline - time.monotonic() + 0.1))
        except subprocess.TimeoutExpired:
            if is_same_process(child):
                os.killpg(int(child["pgid"]), signal.SIGKILL)
            proc.kill()
            code = proc.wait()
    record.update(
        {
            "status": "timed_out" if timed_out else "completed" if code == 0 else "failed",
            "exit_code": code,
            "timed_out": timed_out,
            "finished_at_utc": utc_now(),
            "finished_at_epoch": time.time(),
        }
    )
    atomic_write_json(receipt_path, record)
    return code, timed_out, record
