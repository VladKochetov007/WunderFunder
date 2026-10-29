from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    from scripts.tl3_finite17_dashboard_core import _append_event, _context, _event
except ModuleNotFoundError:
    from tl3_finite17_dashboard_core import _append_event, _context, _event


def _proc(pid: int, root: Path) -> dict[str, Any] | None:
    """Read one Linux process identity, command and environment."""
    try:
        stat = (root / str(pid) / "stat").read_text(encoding="utf-8")
        cmdline = (root / str(pid) / "cmdline").read_bytes().split(b"\0")
        environ = (root / str(pid) / "environ").read_bytes().split(b"\0")
    except OSError:
        return None
    fields = stat[stat.rfind(")") + 2 :].split()
    try:
        return {
            "state": fields[0],
            "pgrp": int(fields[2]),
            "session": int(fields[3]),
            "start_ticks": str(int(fields[19])),
            "cmdline": [part.decode() for part in cmdline if part],
            "env": dict(part.decode().split("=", 1) for part in environ if b"=" in part),
        }
    except (IndexError, ValueError, UnicodeDecodeError):
        return None


def _owned_process(
    task: dict[str, Any], node: dict[str, Any], root: Path, proc_root: Path
) -> tuple[int, str] | None:
    pid, uuid = task.get("pid"), task.get("gpu_uuid")
    if type(pid) is not int or task.get("state") != "running" or not isinstance(uuid, str):
        return None
    process = _proc(pid, proc_root)
    if (
        process is None
        or process["state"] in {"Z", "X"}
        or process["start_ticks"] != str(task.get("start_ticks"))
        or process["pgrp"] != pid
        or process["session"] != pid
        or process["env"].get("CUDA_VISIBLE_DEVICES") != uuid
    ):
        return None
    script = str((root / "scripts" / "train_rnn.py").resolve())
    command = process["cmdline"]
    index = next((i for i, arg in enumerate(command) if str(Path(arg).resolve()) == script), None)
    if index is None or command[index + 1 :] != node["args"]:
        return None
    return pid, uuid


def _telemetry() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read PID/GPU usage through nvidia-smi without initializing CUDA."""

    def query(fields: str) -> list[list[str]]:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return list(csv.reader(result.stdout.splitlines()))

    try:
        apps = query("compute-apps=pid,gpu_uuid,used_memory")
        gpus = query("gpu=uuid,utilization.gpu,memory.used")
    except (OSError, subprocess.SubprocessError):
        return [], []
    app_rows, gpu_rows = [], []
    for row in apps:
        try:
            app_rows.append({"pid": int(row[0]), "uuid": row[1].strip(), "memory": float(row[2])})
        except (IndexError, ValueError):
            continue
    for row in gpus:
        try:
            gpu_rows.append(
                {"uuid": row[0].strip(), "util": float(row[1]), "memory": float(row[2])}
            )
        except (IndexError, ValueError):
            continue
    return app_rows, gpu_rows


def _resource(
    view: Path,
    plan: dict[str, Any],
    node: dict[str, Any],
    task: dict[str, Any],
    epoch: float,
    process: tuple[int, str] | None,
    telemetry: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    proc_root: Path,
    known: dict[tuple[str, int | float, int], dict[str, Any]],
) -> int:
    if process is None:
        return 0
    pid, uuid = process
    apps, gpus = telemetry
    app = next((row for row in apps if row["pid"] == pid and row["uuid"] == uuid), None)
    gpu = next((row for row in gpus if row["uuid"] == uuid), None)
    if app is None or gpu is None:
        return 0
    metrics: dict[str, int | float] = {
        "resource/process_gpu_memory_used_mib": app["memory"],
        "resource/gpu_memory_used_mib": gpu["memory"],
        "resource/gpu_utilization_percent": gpu["util"],
    }
    try:
        pages = int((proc_root / str(pid) / "statm").read_text(encoding="ascii").split()[1])
        metrics["resource/process_rss_bytes"] = pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, IndexError, ValueError, AttributeError):
        pass
    prior = [step for kind, _epoch, step in known if kind == "resources"]
    event = _event(
        "resources",
        max(prior, default=-1) + 1,
        epoch,
        metrics,
        {
            **_context(plan, node),
            "pid": pid,
            "pid_start_ticks": str(task["start_ticks"]),
            "gpu_uuid": uuid,
            "process_identity_verified": True,
            "gpu_process_pid_uuid_verified": True,
        },
    )
    return int(_append_event(view / "monitor_events.jsonl", event, known))
