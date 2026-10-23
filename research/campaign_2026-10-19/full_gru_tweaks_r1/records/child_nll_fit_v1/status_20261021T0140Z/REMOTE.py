#!/usr/bin/env python3
"""Capture one bounded read-only startup snapshot of the exact F0 NLL unit."""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

REMOTE_ROOT = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
PRIVATE_DIR = REMOTE_ROOT / "research_tmp/fg19_child_nll_fit_v1"
RUN_DIR = REMOTE_ROOT / "data/artifacts/runs/fg19_child_nll_f0_s1001_fit_v1"
UNIT = "computer-vision-homework-fg19-child-nll-fit-v1.service"
EXPECTED_INVOCATION_ID = "c833c613fb92426981ab255a5faf4452"
EXPECTED_RECEIPTS = {
    "ADMISSION.json": (11_750, "54b97021a829d1352e9b48a7a6a64fa97a5d1d69c81883bd6518c514858a356c"),
    "LAUNCH.json": (8_787, "2b313f148559b12379f5c98ffd8e94805c5d1f8c33de18145ee4d3fa6efc7738"),
}
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


def _show(unit: str = UNIT) -> tuple[dict[str, object], dict[str, str]]:
    """Query fixed properties for the NLL unit or its campaign parent slice."""
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
    args = ["systemctl", "--user", "show", "--all", unit]
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
    """Read only the new NLL unit's bounded cgroup counters."""
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


def _parent_cgroup_state() -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    """Capture all populated campaign descendants and parent memory counters."""
    query, properties = _show("wnn-campaign.slice")
    group = properties.get("ControlGroup", "")
    if not group or group == "/":
        return query, properties, {"error": "campaign slice has no bounded cgroup path"}
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    paths = [root / "cgroup.procs", *root.rglob("cgroup.procs")]
    if len(paths) > 128:
        raise ValueError("campaign cgroup tree exceeds the 128-file bound")
    populated: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            raw = stream.read(32769)
        if len(raw) > 32768:
            raise ValueError("campaign cgroup process list exceeds the 32 KiB bound")
        value = raw.decode("ascii", errors="replace").strip()
        if value:
            populated[str(path)] = value
    parent_values: dict[str, object] = {"populated_descendants": populated}
    for name in ("memory.current", "memory.events", "memory.events.local"):
        path = root / name
        if not path.is_file():
            parent_values[name] = None
            continue
        with path.open("rb") as stream:
            raw = stream.read(8193)
        parent_values[name] = raw[:8192].decode("ascii", errors="replace").strip()
        parent_values[name + "_truncated"] = len(raw) > 8192
    return query, properties, parent_values


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


def _gpu_process_ownership(query: dict[str, object]) -> list[dict[str, object]]:
    """Attach UID and cgroup ownership metadata to every selected-GPU PID."""
    if query.get("returncode") != 0 or query.get("timed_out"):
        return []
    output = query.get("stdout")
    if not isinstance(output, str):
        return []
    owners: list[dict[str, object]] = []
    for row in output.splitlines():
        if row.strip().startswith("No running processes found"):
            continue
        columns = [column.strip() for column in row.split(",")]
        if len(columns) != 3 or not columns[0].isdigit():
            owners.append({"raw_row": row, "parse_error": True})
            continue
        pid = columns[0]
        proc = Path("/proc") / pid
        try:
            uid = proc.stat().st_uid
            with (proc / "cgroup").open("rb") as stream:
                raw = stream.read(32769)
            if len(raw) > 32768:
                owners.append({"pid": int(pid), "uid": uid, "cgroup_over_limit": True})
                continue
            cgroup = raw.decode("ascii", errors="replace").strip()
        except (FileNotFoundError, ProcessLookupError):
            owners.append(
                {
                    "pid": int(pid),
                    "gpu_uuid": columns[1],
                    "used_memory": columns[2],
                    "exited_during_observation": True,
                }
            )
            continue
        except OSError as error:
            owners.append({"pid": int(pid), "ownership_error": str(error)[:512]})
            continue
        owners.append(
            {
                "pid": int(pid),
                "gpu_uuid": columns[1],
                "used_memory": columns[2],
                "uid": uid,
                "cgroup": cgroup,
            }
        )
    return owners


def main() -> None:
    """Print a one-shot startup observation without acceptance or mutation."""
    query, properties = _show()
    if properties.get("InvocationID") != EXPECTED_INVOCATION_ID:
        raise RuntimeError("NLL unit invocation differs from the launched invocation")
    parent_query, parent_properties, parent_state = _parent_cgroup_state()
    receipts = {name: _small_file(name) for name in ("ADMISSION.json", "LAUNCH.json", "unit.log")}
    included_bytes = sum(
        item.get("bytes", 0)
        for item in receipts.values()
        if isinstance(item.get("bytes"), int) and isinstance(item.get("base64"), str)
    )
    if included_bytes > MAX_TOTAL_BYTES:
        raise ValueError("small-file receipt aggregate exceeds the 4 MB capture limit")
    for name, (expected_bytes, expected_sha) in EXPECTED_RECEIPTS.items():
        receipt = receipts[name]
        if (
            receipt.get("regular_file") is not True
            or receipt.get("bytes") != expected_bytes
            or receipt.get("sha256") != expected_sha
        ):
            raise RuntimeError(f"NLL {name} receipt identity differs")
    gpu_process_query = _run(
        [
            "nvidia-smi",
            "-i",
            GPU_UUID,
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader",
        ]
    )
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
        "processes": gpu_process_query,
        "process_ownership": _gpu_process_ownership(gpu_process_query),
    }
    capture = {
        "schema": "fg19-child-nll-startup-capture.v1",
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # noqa: UP017 -- keep Python 3.10 compatibility
        "expected_unit": UNIT,
        "expected_invocation_id": EXPECTED_INVOCATION_ID,
        "expected_run": "fg19_child_nll_f0_s1001_fit_v1",
        "unit_query": query,
        "unit_properties": properties,
        "parent_query": parent_query,
        "parent_properties": parent_properties,
        "parent_state": parent_state,
        "own_process_rss": _process_rss(properties.get("MainPID", "")),
        "own_cgroup_state": _own_cgroup_state(properties),
        "gpu": gpu,
        "receipts": receipts,
        "run_directory_stat_only": _stat_inventory(RUN_DIR),
        "scope": (
            "One read-only startup snapshot of the exact NLL unit invocation, three bounded private-D "
            "receipts, own process/cgroup, parent campaign cgroup population/current/events, selected GPU "
            "process UID/cgroup ownership and stat-only run directory. No predecessor unit, model/checkpoint/"
            "optimizer/cache/dictionary/data payload is read; no action or acceptance occurs."
        ),
    }
    print(json.dumps(capture, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
