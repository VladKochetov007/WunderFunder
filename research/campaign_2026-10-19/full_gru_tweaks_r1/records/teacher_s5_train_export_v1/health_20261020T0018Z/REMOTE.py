"""One bounded, read-only startup snapshot for the exact S5 train export."""

from __future__ import annotations

import datetime as dt
import json
import os
import stat
import subprocess
from pathlib import Path

B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
UNIT = "computer-vision-homework-fg19-teacher-s5-train-export-v1.service"
INVOCATION = "9d2b2c9328f241bf896fd1ca75a3421b"
EXPECTED_MAIN_PID = "3802605"
GPU = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"
SOURCE = B / "source-fg19-cache-88becca8"
SOURCE_COMMIT = "88becca892b3d6b97022cf06076c84a7ab165974"
PREDICTIONS = B / "research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1"
LOG = B / "research_tmp/fg19_teacher_s5_train_export_v1/unit.log"
MAX_CGROUPS = 256
MAX_PREDICTION_NAMES = 64
MAX_LOG_BYTES = 262_144


def command(args: list[str], timeout: int = 10) -> dict[str, object]:
    """Capture one bounded read-only command without hiding failures."""
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        return {
            "argv": args,
            "timeout_seconds": timeout,
            "timed_out": True,
            "stdout": _decode(error.stdout),
            "stderr": _decode(error.stderr),
        }
    except OSError as error:
        return {"argv": args, "os_error": f"{type(error).__name__}: {error}"}
    return {
        "argv": args,
        "returncode": result.returncode,
        "stdout": result.stdout.decode(errors="replace"),
        "stderr": result.stderr.decode(errors="replace"),
    }


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def show_unit(unit: str, fields: list[str]) -> dict[str, str]:
    result = command(
        ["systemctl", "--user", "show", "--all", unit, *sum((["-p", key] for key in fields), [])]
    )
    if result.get("returncode") != 0:
        return {"_error": json.dumps(result, sort_keys=True)}
    return dict(line.split("=", 1) for line in str(result["stdout"]).splitlines() if "=" in line)


def cgroup_files(root: Path) -> tuple[list[Path], bool]:
    """Enumerate cgroup.procs files with a hard directory-count bound."""
    found: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        if not directory.is_dir():
            continue
        found.append(directory / "cgroup.procs")
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
        except OSError:
            continue
        if len(found) > MAX_CGROUPS:
            return found[:MAX_CGROUPS], True
    return found, False


def process_identity(pid: str) -> dict[str, object]:
    directory = Path("/proc") / pid
    try:
        owner = (directory / "cgroup").read_text().strip()
        uid = directory.stat().st_uid
        commandline = (
            (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        )
    except FileNotFoundError:
        return {"pid": pid, "exited_during_observation": True}
    except OSError as error:
        return {"pid": pid, "observation_error": f"{type(error).__name__}: {error}"}
    return {"pid": pid, "uid": uid, "cgroup": owner, "cmdline": commandline[:1000]}


def read_events(group: Path) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in ("memory.events", "memory.events.local"):
        try:
            values[name] = (group / name).read_text()
        except OSError as error:
            values[name] = f"unavailable: {type(error).__name__}: {error}"
    return values


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def path_stat(path: Path) -> dict[str, object]:
    """Return metadata only and do not follow symlinks."""
    try:
        value = path.lstat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "kind": "symlink"
        if stat.S_ISLNK(value.st_mode)
        else "directory"
        if stat.S_ISDIR(value.st_mode)
        else "regular"
        if stat.S_ISREG(value.st_mode)
        else "other",
        "mode": oct(value.st_mode),
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "device": value.st_dev,
        "inode": value.st_ino,
    }


def prediction_inventory(path: Path) -> dict[str, object]:
    """List at most 64 top-level names and lstat metadata; never open payloads."""
    directory = path_stat(path)
    if directory.get("kind") != "directory":
        return {
            "directory_stat": directory,
            "entries": [],
            "truncated": False,
            "payload_read_or_hashed": False,
        }

    entries_out: list[dict[str, object]] = []
    truncated = False
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if len(entries_out) >= MAX_PREDICTION_NAMES:
                    truncated = True
                    break
                try:
                    value = entry.stat(follow_symlinks=False)
                except OSError as error:
                    entries_out.append(
                        {"name": entry.name[:256], "stat_error": f"{type(error).__name__}: {error}"}
                    )
                    continue
                kind = (
                    "symlink"
                    if stat.S_ISLNK(value.st_mode)
                    else "directory"
                    if stat.S_ISDIR(value.st_mode)
                    else "regular"
                    if stat.S_ISREG(value.st_mode)
                    else "other"
                )
                entries_out.append(
                    {
                        "name": entry.name[:256],
                        "kind": kind,
                        "bytes": value.st_size,
                        "mtime_ns": value.st_mtime_ns,
                        "device": value.st_dev,
                        "inode": value.st_ino,
                        "partial_suffix": entry.name.endswith(".partial"),
                    }
                )
    except OSError as error:
        return {
            "directory_stat": directory,
            "entries": entries_out,
            "truncated": truncated,
            "scan_error": f"{type(error).__name__}: {error}",
            "payload_read_or_hashed": False,
        }
    return {
        "directory_stat": directory,
        "entries": entries_out,
        "entry_limit": MAX_PREDICTION_NAMES,
        "truncated": truncated,
        "payload_read_or_hashed": False,
    }


def log_tail(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - MAX_LOG_BYTES))
            data = stream.read(MAX_LOG_BYTES)
    except FileNotFoundError:
        return {"path": str(path), "exists": False, "tail_lines": []}
    except OSError as error:
        return {"path": str(path), "read_error": f"{type(error).__name__}: {error}"}
    lines = data.decode(errors="replace").splitlines()
    return {
        "path": str(path),
        "exists": True,
        "bytes": size,
        "tail_byte_cap": MAX_LOG_BYTES,
        "truncated_at_start": size > len(data),
        "line_count_in_tail": len(lines),
        "tail_lines": lines[-80:],
    }


UNIT_FIELDS = [
    "LoadState",
    "ActiveState",
    "SubState",
    "Result",
    "InvocationID",
    "MainPID",
    "ControlGroup",
    "ExecMainCode",
    "ExecMainStatus",
    "ExecMainStartTimestamp",
    "ExecStart",
    "CPUQuotaPerSecUSec",
    "MemoryCurrent",
    "MemoryPeak",
    "MemoryHigh",
    "MemoryMax",
    "MemorySwapMax",
]
PARENT_FIELDS = [
    "LoadState",
    "ActiveState",
    "SubState",
    "ControlGroup",
    "CPUQuotaPerSecUSec",
    "MemoryCurrent",
    "MemoryHigh",
    "MemoryMax",
    "MemorySwapMax",
]


def main() -> None:
    """Print a bounded startup snapshot without changing the unit or its files."""
    unit = show_unit(UNIT, UNIT_FIELDS)
    parent = show_unit("wnn-campaign.slice", PARENT_FIELDS)
    parent_group_text = parent.get("ControlGroup", "")
    if not parent_group_text or parent_group_text == "/":
        raise RuntimeError("campaign slice returned no explicit cgroup path")
    parent_group = Path("/sys/fs/cgroup") / parent_group_text.lstrip("/")
    paths, overflow = cgroup_files(parent_group)
    populated: dict[str, list[str]] = {}
    for proc_file in paths:
        try:
            pids = proc_file.read_text().split()
        except OSError:
            continue
        if pids:
            populated[str(proc_file.parent)] = pids
    all_pids = {pid for values in populated.values() for pid in values}

    own_group_text = unit.get("ControlGroup", "")
    own_group = (
        Path("/sys/fs/cgroup") / own_group_text.lstrip("/")
        if own_group_text and own_group_text != "/"
        else None
    )
    own_pids = read_text(own_group / "cgroup.procs") if own_group is not None else None

    gpu_status = command(
        [
            "nvidia-smi",
            "-i",
            GPU,
            "--query-gpu=uuid,name,memory.used,memory.free,memory.total",
            "--format=csv,noheader",
        ]
    )
    gpu_apps = command(
        [
            "nvidia-smi",
            "-i",
            GPU,
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader",
        ]
    )
    gpu_processes: list[dict[str, object]] = []
    for row in str(gpu_apps.get("stdout", "")).splitlines():
        if row.strip().startswith("No running processes found") or not row.strip():
            continue
        columns = [value.strip() for value in row.split(",")]
        if len(columns) != 3:
            gpu_processes.append({"raw": row, "parse_error": True})
            continue
        identity = process_identity(columns[0])
        identity.update(
            {
                "gpu_uuid": columns[1],
                "used_memory": columns[2],
                "campaign_descendant": columns[0] in all_pids,
            }
        )
        gpu_processes.append(identity)

    snapshot = {
        "schema": "full_gru_tweaks.teacher_s5_train_export_startup_health.v1",
        "captured_utc": dt.datetime.now(dt.UTC).isoformat(),
        "expected": {
            "unit": UNIT,
            "invocation_id": INVOCATION,
            "main_pid": EXPECTED_MAIN_PID,
            "gpu_uuid": GPU,
            "source_commit_pin": SOURCE_COMMIT,
            "source_path": str(SOURCE),
        },
        "unit": unit,
        "exact_invocation_match": unit.get("InvocationID") == INVOCATION,
        "expected_main_pid_match": unit.get("MainPID") == EXPECTED_MAIN_PID,
        "parent": parent,
        "parent_cgroup_scan": {
            "root": str(parent_group),
            "cgroup_count": len(paths),
            "overflow": overflow,
            "populated": populated,
        },
        "parent_memory_events": read_events(parent_group),
        "own_cgroup": {
            "path": own_group_text,
            "unavailable": "unit returned no explicit child cgroup path",
        }
        if own_group is None
        else {
            "path": own_group_text,
            "populated_pids": [] if own_pids is None else own_pids.split(),
            "memory_current": read_text(own_group / "memory.current"),
            "memory_peak": read_text(own_group / "memory.peak"),
            "events": read_events(own_group),
        },
        "campaign_processes": [process_identity(pid) for pid in sorted(all_pids, key=int)],
        "gpu": {"status": gpu_status, "compute_apps": gpu_apps, "processes": gpu_processes},
        "source_path_stat_only": path_stat(SOURCE),
        "prediction_directory_stat_only": prediction_inventory(PREDICTIONS),
        "log_tail": log_tail(LOG),
        "scope": (
            "one bounded read-only snapshot of only the exact export unit and campaign parent; "
            "no unit mutation, source-content verification, model/checkpoint load or hash, or "
            "prediction payload read or hash; prediction inventory is top-level stat-only, "
            "including .partial files"
        ),
    }
    print(json.dumps(snapshot, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
