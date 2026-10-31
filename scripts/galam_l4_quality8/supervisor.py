from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .admission import preflight
from .common import LaunchError, atomic_write_json, process_identity, read_json, utc_now
from .fit_processes import assert_reserved_root_fresh, base_environment, receipt_cells, start_fits
from .monitor import RECEIPT_SCHEMA, refresh_summary


def reserve_and_detach(
    *,
    python: Path,
    script: Path,
    repo_root: Path,
    panel: Path,
    profiles: list[str],
    data_root: Path,
    split_manifest: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Reserve one new output tree and start the fixed-eight supervisor detached."""
    if sys.platform != "linux":
        raise LaunchError("launch is available only on the Linux L4 receiving host")
    python = python.expanduser().absolute()
    if not python.is_file():
        raise LaunchError("--python must name the isolated interpreter file")
    script = script.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    output_root = output_root.expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise LaunchError(f"output root already exists: {output_root}")
    if output_root.parent.is_symlink() or not output_root.parent.is_dir():
        raise LaunchError("output parent must exist and cannot be a symlink")
    output_root.mkdir(mode=0o700)
    receipt_path = output_root / "launch_receipt.json"
    bootstrap = {
        "schema": RECEIPT_SCHEMA,
        "launch_id": uuid.uuid4().hex,
        "status": "reserved",
        "created_at": utc_now(),
        "output_root": str(output_root.resolve()),
        "panel": str(panel.expanduser().absolute()),
        "profiles": profiles,
        "data_root": str(data_root.expanduser().absolute()),
        "split_manifest": str(split_manifest.expanduser().absolute()),
        "repo_root": str(repo_root),
        "python": str(python),
        "cells": {},
    }
    atomic_write_json(receipt_path, bootstrap)
    log_path = output_root / "supervisor.log"
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    log_file = os.fdopen(descriptor, "ab", buffering=0)
    command = [
        str(python),
        str(script),
        "supervise",
        "--receipt",
        str(receipt_path),
    ]
    try:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
            env=base_environment(repo_root),
        )
    except Exception as error:
        failed = read_json(receipt_path)
        failed["status"] = "supervisor_start_failed"
        failed["error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(receipt_path, failed)
        raise
    finally:
        log_file.close()
    return {"receipt": str(receipt_path), "supervisor_pid": child.pid, "status": "starting"}


def run_supervisor(receipt_path: Path, poll_seconds: int = 30) -> int:
    """Validate profiles, start one detached process per L4, and monitor owned fits."""
    receipt_path = receipt_path.resolve(strict=True)
    receipt = read_json(receipt_path)
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != "reserved":
        raise LaunchError("supervisor requires its original reserved launch receipt")
    if Path(sys.executable).resolve() != Path(receipt["python"]).resolve():
        raise LaunchError("supervisor did not start under the pinned isolated interpreter")
    own = process_identity(os.getpid())
    if own is None:
        raise LaunchError("could not record the supervisor's Linux process identity")
    receipt.update(
        {
            "status": "preflight",
            "supervisor": {
                "pid": os.getpid(),
                "start_ticks": own["start_ticks"],
                "process_group": own["process_group"],
            },
        }
    )
    atomic_write_json(receipt_path, receipt)
    output_root = Path(receipt["output_root"])
    try:
        admitted = preflight(
            Path(receipt["panel"]),
            list(receipt["profiles"]),
            Path(receipt["data_root"]),
            Path(receipt["split_manifest"]),
            output_root,
            Path(receipt["repo_root"]),
            reserved_output=True,
        )
        if Path(sys.executable).resolve() != Path(receipt["python"]).resolve():
            raise LaunchError("isolated interpreter path changed during admission")
        assert_reserved_root_fresh(output_root)
        receipt.update(
            {
                "status": "admitted",
                "panel_sha256": admitted["panel_sha256"],
                "source_sha256": admitted["source_sha256"],
                "launcher_sha256": admitted["launcher_sha256"],
                "split_sha256": admitted["split_sha256"],
                "runtime": admitted["runtime"],
                "cells": receipt_cells(output_root, admitted),
                "admitted_at": utc_now(),
            }
        )
        atomic_write_json(receipt_path, receipt)
        children = start_fits(receipt_path, receipt, admitted)
    except Exception as error:
        current = read_json(receipt_path)
        current["status"] = "preflight_failed" if current.get("status") in ("reserved", "preflight") else "launch_failed"
        current["error"] = f"{type(error).__name__}: {error}"
        current["finished_at"] = utc_now()
        atomic_write_json(receipt_path, current)
        if current.get("cells"):
            refresh_summary(receipt_path)
        if not current.get("cells"):
            return 2
        children = {}

    while True:
        current = read_json(receipt_path)
        summary = refresh_summary(receipt_path)
        for cell_id, child in children.items():
            exit_code = child.poll()
            if exit_code is not None:
                current["cells"][cell_id]["return_code"] = int(exit_code)
        live = any(item.poll() is None for item in children.values()) or any(
            item.get("process_state") == "running" for item in summary["cells"].values()
        )
        if not live:
            current["status"] = (
                "completed" if summary["status"] == "completed" else "finished_with_failures"
            )
            current["finished_at"] = utc_now()
            atomic_write_json(receipt_path, current)
            refresh_summary(receipt_path)
            return 0 if summary["status"] == "completed" else 1
        atomic_write_json(receipt_path, current)
        time.sleep(poll_seconds)
