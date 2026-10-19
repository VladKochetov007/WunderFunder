"""Durable watcher state and bounded helper subprocess utilities."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from delivery_watch_core import append_event, read_object, sha256_file, write_atomic


def _cell_write(state: Path, cell: dict[str, Any], phase: str, **details: Any) -> None:
    """Persist a watcher-owned phase transition before and after work."""
    now = time.time()
    record = {**cell, phase: {"updated_unix": now, **details}}
    write_atomic(state / "cells" / f"{cell['job_id']}.json", record)
    append_event(
        state / "events.jsonl",
        {"time_unix": now, "job_id": cell["job_id"], "phase": phase, **details},
    )
    cell.update(record)


def _cell_load(state: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Load one cell's durable state or return a fresh pending record."""
    path = state / "cells" / f"{job['id']}.json"
    if path.exists():
        cell = read_object(path)
        if cell.get("job_id") != job["id"] or cell.get("job") != job:
            raise ValueError(f"watcher state has a different panel binding: {job['id']}")
        return cell
    return {
        "job_id": job["id"],
        "job": job,
        "fit": {"status": "pending"},
        "export": {"status": "pending"},
    }


def _run_child(
    argv: list[str], env: dict[str, str], log_path: Path, timeout: int
) -> dict[str, Any]:
    """Run a bounded child process and terminate only its new process group."""
    if log_path.exists() or log_path.is_symlink():
        raise FileExistsError(f"attempt log already exists: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(
            argv,
            cwd=env["WNN_PROJECT_ROOT"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124
        log.flush()
        os.fsync(log.fileno())
    return {"returncode": code, "log_sha256": sha256_file(log_path), "argv": argv}


def _base_cli(config: dict[str, Any], recovery: Path) -> list[str]:
    """Build helper arguments for only the three permitted delivery phases."""
    return [
        sys.executable,
        "-B",
        str(config["helper_path"] / "r01_delivery_recovery_v2.py"),
        "--root",
        str(config["root_path"]),
        "--packet",
        str(config["packet_path"]),
        "--source-root",
        str(config["root_path"] / "runtime/source-a266-lf-subset-v1"),
        "--recovery-root",
        str(recovery),
    ]
