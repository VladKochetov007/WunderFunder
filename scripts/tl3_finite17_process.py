"""Owned-process and free-GPU checks for the fixed TL3 17-node graph."""

from __future__ import annotations

import csv
import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.tl3_finite17_contract import DriverError

EXCLUDED_GPU_INDICES = frozenset({4})


def _killpg(process_group: int, signum: int) -> None:
    """Call the Linux process-group signal primitive when available."""
    if not hasattr(os, "killpg"):
        raise DriverError("process-group signaling requires Linux")
    os.killpg(process_group, signum)


def proc_identity(pid: int) -> dict[str, int | str] | None:
    """Read Linux PID, process-group, session, and start-tick identity safely."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    try:
        return {
            "state": fields[0],
            "parent_pid": int(fields[1]),
            "process_group": int(fields[2]),
            "session": int(fields[3]),
            "start_ticks": int(fields[19]),
        }
    except (IndexError, ValueError):
        return None


def process_group_members(process_group: int, session: int) -> list[int]:
    """List only non-zombie members and reject any PID group outside its session."""
    members: list[int] = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        identity = proc_identity(int(path.name))
        if identity is None or identity["process_group"] != process_group:
            continue
        if identity["session"] != session:
            raise DriverError("process-group identity is mixed with an unrelated session")
        if identity["state"] not in {"Z", "X"}:
            members.append(int(path.name))
    return sorted(members)


def owned_group_state(record: dict[str, Any]) -> str:
    """Classify only the start-tick-bound session created or adopted by this driver."""
    pid = record.get("pid")
    pgid = record.get("process_group")
    session = record.get("session")
    if type(pid) is not int or type(pgid) is not int or type(session) is not int:
        return "invalid_receipt"
    if pid != pgid or pgid != session:
        return "invalid_receipt"
    leader = proc_identity(pid)
    if leader is not None and (
        leader["start_ticks"] != int(record.get("start_ticks", -1))
        or leader["process_group"] != pgid
        or leader["session"] != session
    ):
        return "pid_reused_or_unowned"
    try:
        members = process_group_members(pgid, session)
    except DriverError:
        return "mixed_or_unowned_group"
    return "running" if members else "exited"


def terminate_owned_group(
    record: dict[str, Any],
    *,
    grace_seconds: float = 60.0,
    killpg: Callable[[int, int], None] = _killpg,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """TERM then KILL the whole recorded group, only after session ownership checks."""
    if grace_seconds < 0:
        raise ValueError("process-group grace period cannot be negative")
    state = owned_group_state(record)
    if state == "exited":
        return True
    if state != "running":
        raise DriverError(f"refusing to signal an unowned process group: {state}")
    process_group = int(record["process_group"])
    session = int(record["session"])
    killpg(process_group, getattr(signal, "SIGTERM", 15))
    deadline = monotonic() + grace_seconds
    while monotonic() < deadline:
        if not process_group_members(process_group, session):
            return True
        sleep(min(0.25, max(0.0, deadline - monotonic())))
    if process_group_members(process_group, session):
        killpg(process_group, getattr(signal, "SIGKILL", 9))
        deadline = monotonic() + 5.0
        while monotonic() < deadline:
            if not process_group_members(process_group, session):
                return True
            sleep(0.05)
    return not process_group_members(process_group, session)


def nvidia_inventory(
    executable: str = "nvidia-smi",
) -> tuple[dict[int, str], dict[str, set[int]]]:
    """Read physical GPU UUIDs and current compute-app PIDs; errors fail closed."""
    gpu_result = subprocess.run(
        [executable, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    apps_result = subprocess.run(
        [
            executable,
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    devices: dict[int, str] = {}
    for row in csv.reader(gpu_result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 2 or not row[0].strip().isdecimal():
            raise DriverError("nvidia-smi returned malformed physical GPU identity")
        index, uuid = int(row[0].strip()), row[1].strip()
        if index in devices or not uuid.startswith("GPU-"):
            raise DriverError("nvidia-smi returned duplicate or invalid GPU identities")
        devices[index] = uuid
    if not devices:
        raise DriverError("nvidia-smi reported no physical GPUs")
    applications: dict[str, set[int]] = {uuid: set() for uuid in devices.values()}
    for row in csv.reader(apps_result.stdout.splitlines(), skipinitialspace=True):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) == 1 and row[0].strip().lower() == "no running processes found":
            continue
        if len(row) != 2 or not row[1].strip().isdecimal():
            raise DriverError("nvidia-smi returned malformed compute-app ownership")
        uuid, pid = row[0].strip(), int(row[1].strip())
        if uuid not in applications or pid < 2:
            raise DriverError("compute app refers to an unknown GPU or PID")
        applications[uuid].add(pid)
    return devices, applications


def protected_gpu_indices(
    receipt_path: Path, devices: dict[int, str] | None = None
) -> set[int]:
    """Reserve every nonterminal GALAM card without adopting or signaling its PIDs."""
    import json

    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise DriverError("the seven-GALAM ownership receipt is missing or unsafe")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    cells = receipt.get("cells")
    if not isinstance(cells, dict) or len(cells) != 8:
        raise DriverError("the protected GALAM receipt does not identify its exact eight cells")
    protected: set[int] = set()
    for record in cells.values():
        if not isinstance(record, dict):
            raise DriverError("malformed cell in the protected GALAM receipt")
        gpu = record.get("gpu")
        pid = record.get("pid")
        pgid = record.get("process_group")
        ticks = record.get("start_ticks")
        if (
            type(gpu) is not int
            or type(pid) is not int
            or type(pgid) is not int
            or not isinstance(ticks, (str, int))
        ):
            raise DriverError("protected GALAM cell lacks a PID-bound GPU identity")
        if devices is not None and (
            gpu not in devices or record.get("gpu_uuid") != devices[gpu]
        ):
            raise DriverError("protected GALAM GPU index/UUID crosswalk changed")
        identity = proc_identity(pid)
        if identity is not None and (
            identity["start_ticks"] == int(ticks) and identity["process_group"] == pgid
        ):
            protected.add(gpu)
    return protected


def free_gpu_devices(
    devices: dict[int, str],
    applications: dict[str, set[int]],
    protected: set[int],
    reserved: set[int],
) -> list[tuple[int, str]]:
    """Return only empty, nonexcluded cards, preserving every existing worker."""
    result = []
    for index, uuid in sorted(devices.items()):
        if index in EXCLUDED_GPU_INDICES or index in protected or index in reserved:
            continue
        if applications.get(uuid):
            continue
        result.append((index, uuid))
    return result


def start_identity(record: dict[str, Any], pid: int) -> dict[str, Any]:
    """Bind a freshly spawned child to its own PID-led process group and session."""
    identity = proc_identity(pid)
    if identity is None:
        raise DriverError("spawned task did not appear in /proc")
    if identity["process_group"] != pid or identity["session"] != pid:
        raise DriverError("spawned task did not create its own session and process group")
    return {
        **record,
        "pid": pid,
        "process_group": pid,
        "session": pid,
        "start_ticks": str(identity["start_ticks"]),
    }
