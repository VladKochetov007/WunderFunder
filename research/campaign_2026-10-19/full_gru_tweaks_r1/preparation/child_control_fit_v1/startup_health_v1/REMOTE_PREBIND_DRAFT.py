#!/usr/bin/env python3
"""Capture one bounded read-only startup snapshot of the exact F0 CONTROL unit."""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

REMOTE_ROOT = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
PRIVATE_DIR = REMOTE_ROOT / "research_tmp/fg19_child_control_fit_v1"
RUN_DIR = REMOTE_ROOT / "data/artifacts/runs/fg19_child_control_f0_s1001_fit_v1"
UNIT = "computer-vision-homework-fg19-child-control-fit-v1.service"
GPU_UUID = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 4_000_000
MAX_DIR_ENTRIES = 128


def _run(args: list[str], timeout: int = 10) -> dict[str, object]:
    """Run one fixed read-only query and preserve bounded output."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return {
            "returncode": None,
            "stdout": stdout[:32768],
            "stderr": stderr[:32768],
            "timed_out": True,
        }
    except OSError as error:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": str(error)[:32768],
            "timed_out": False,
        }
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[:32768],
        "stderr": result.stderr[:32768],
        "timed_out": False,
    }


def _show() -> tuple[dict[str, object], dict[str, str]]:
    """Query properties for the CONTROL unit only; do not touch predecessor units."""
    keys = [
        "InvocationID",
        "LoadState",
        "ActiveState",
        "SubState",
        "Result",
        "MainPID",
        "ExecMainCode",
        "ExecMainStatus",
        "ControlGroup",
        "CPUUsageNSec",
        "MemoryCurrent",
        "MemoryPeak",
        "MemoryHigh",
        "MemoryMax",
        "MemorySwapMax",
        "CPUQuotaPerSecUSec",
        "ExecStart",
        "ExecMainStartTimestamp",
        "ExecMainExitTimestamp",
    ]
    args = ["systemctl", "--user", "show", "--all", UNIT]
    for key in keys:
        args.extend(("-p", key))
    query = _run(args)
    output = query.get("stdout")
    properties = (
        dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        if query.get("returncode") == 0 and isinstance(output, str)
        else {}
    )
    return query, properties


def _small_file(name: str) -> dict[str, object]:
    """Preserve one bounded receipt/log file; never follow symlinks."""
    path = PRIVATE_DIR / name
    if path.is_symlink():
        return {"present": True, "symlink": True}
    if not path.is_file():
        return {"present": path.exists(), "regular_file": False}
    stat = path.stat()
    if stat.st_size > MAX_FILE_BYTES:
        return {"present": True, "regular_file": True, "over_limit": True, "bytes": stat.st_size}
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        return {"present": True, "regular_file": True, "over_limit": True, "bytes": len(raw)}
    return {
        "present": True,
        "regular_file": True,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def _stat_inventory(path: Path) -> dict[str, object]:
    """List a run directory using stat metadata only; do not read file contents."""
    if path.is_symlink():
        return {"exists": True, "symlink": True, "entries": []}
    if not path.exists():
        return {"exists": False, "symlink": False, "entries": []}
    if not path.is_dir():
        return {"exists": True, "symlink": False, "not_directory": True, "entries": []}
    entries = list(path.iterdir())
    if len(entries) > MAX_DIR_ENTRIES:
        return {"exists": True, "symlink": False, "over_limit": True, "entry_count": len(entries)}
    result = []
    for entry in sorted(entries, key=lambda item: item.name):
        info = entry.lstat()
        result.append(
            {
                "name": entry.name,
                "mode": info.st_mode,
                "is_symlink": entry.is_symlink(),
                "bytes": info.st_size,
                "device": info.st_dev,
                "inode": info.st_ino,
                "mtime_ns": info.st_mtime_ns,
            }
        )
    return {"exists": True, "symlink": False, "entries": result}


def _own_cgroup_state(properties: dict[str, str]) -> dict[str, object] | None:
    """Read only the new CONTROL unit's bounded cgroup counters."""
    group = properties.get("ControlGroup", "")
    if not group or group == "/":
        return None
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    values: dict[str, object] = {"path": str(root)}
    for name in (
        "memory.current",
        "memory.peak",
        "memory.events",
        "memory.events.local",
        "cgroup.procs",
    ):
        path = root / name
        if not path.is_file():
            values[name] = None
            continue
        with path.open("rb") as stream:
            raw = stream.read(8193)
        values[name] = raw[:8192].decode("ascii", errors="replace")
        values[name + "_bytes_observed"] = len(raw)
        values[name + "_truncated"] = len(raw) > 8192
    return values


def _process_rss(pid: str) -> list[str] | None:
    """Read RSS/HWM lines for this unit's current main PID only."""
    if not pid.isdigit() or int(pid) <= 0:
        return None
    path = Path("/proc") / pid / "status"
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if len(raw) > 131072:
        return ["status_file_over_limit", str(len(raw))]
    return [
        line
        for line in raw.decode("ascii", errors="replace").splitlines()
        if line.startswith(("VmRSS:", "VmHWM:"))
    ]


def main() -> None:
    """Print a one-shot startup observation without acceptance or mutation."""
    query, properties = _show()
    receipts = {name: _small_file(name) for name in ("ADMISSION.json", "LAUNCH.json", "unit.log")}
    included_bytes = sum(
        item.get("bytes", 0)
        for item in receipts.values()
        if isinstance(item.get("bytes"), int) and isinstance(item.get("base64"), str)
    )
    if included_bytes > MAX_TOTAL_BYTES:
        raise ValueError("small-file receipt aggregate exceeds the 4 MB capture limit")
    gpu = {
        "memory": _run(
            [
                "nvidia-smi",
                "-i",
                GPU_UUID,
                "--query-gpu=uuid,memory.total,memory.used,memory.free",
                "--format=csv,noheader",
            ]
        ),
        "processes": _run(
            [
                "nvidia-smi",
                "-i",
                GPU_UUID,
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader",
            ]
        ),
    }
    capture = {
        "schema": "fg19-child-control-startup-capture.v1",
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # noqa: UP017 -- keep Python 3.10 compatibility
        "expected_unit": UNIT,
        "expected_run": "fg19_child_control_f0_s1001_fit_v1",
        "unit_query": query,
        "unit_properties": properties,
        "own_process_rss": _process_rss(properties.get("MainPID", "")),
        "own_cgroup_state": _own_cgroup_state(properties),
        "gpu": gpu,
        "receipts": receipts,
        "run_directory_stat_only": _stat_inventory(RUN_DIR),
        "scope": (
            "One read-only startup snapshot of the exact CONTROL unit, three bounded private-D receipts, "
            "its own process/cgroup, selected GPU and stat-only run directory. Invocation ID is captured "
            "from current systemd state and durable LAUNCH bytes after root's launch; no predecessor unit, "
            "model/checkpoint/optimizer/cache/dictionary/data payload is read; no action or acceptance occurs."
        ),
    }
    print(json.dumps(capture, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
