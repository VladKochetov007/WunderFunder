from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .common import LaunchError, atomic_write_json, process_identity, read_json, utc_now
from .fit_processes import base_environment
from .monitor import RECEIPT_SCHEMA


def start_dashboard(launch_root: Path, repo_root: Path) -> dict[str, Any]:
    """Start the optional read-only dashboard on IPv4 loopback port 18766."""
    launch_root = launch_root.expanduser()
    if launch_root.is_symlink():
        raise LaunchError("dashboard launch root cannot be a symlink")
    launch_root = launch_root.resolve(strict=True)
    receipt_path = launch_root / "launch_receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise LaunchError("dashboard root is not owned by the fixed-eight launcher")
    runs_root = launch_root / "runs"
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise LaunchError("dashboard run directory is missing or unsafe")
    log_path = launch_root / "dashboard.log"
    dashboard_receipt = launch_root / "dashboard_receipt.json"
    if (
        log_path.exists()
        or log_path.is_symlink()
        or dashboard_receipt.exists()
        or dashboard_receipt.is_symlink()
    ):
        raise LaunchError("dashboard output or receipt already exists")
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    log_file = os.fdopen(descriptor, "ab", buffering=0)
    script = repo_root.resolve(strict=True) / "scripts" / "research_dashboard.py"
    command = [
        str(Path(receipt["python"])),
        str(script),
        "--runs-root",
        str(runs_root),
        "--host",
        "127.0.0.1",
        "--port",
        "18766",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
            env=base_environment(repo_root),
        )
    finally:
        log_file.close()
    identity = process_identity(process.pid)
    if identity is None or identity["process_group"] != process.pid:
        raise LaunchError("dashboard process identity could not be recorded safely")
    result = {
        "schema": "wnn.galam-l4-quality8-dashboard.v1",
        "created_at": utc_now(),
        "host": "127.0.0.1",
        "port": 18766,
        "url": "http://127.0.0.1:18766/",
        "pid": process.pid,
        "start_ticks": identity["start_ticks"],
        "process_group": identity["process_group"],
        "log": str(log_path),
    }
    atomic_write_json(dashboard_receipt, result)
    return result
