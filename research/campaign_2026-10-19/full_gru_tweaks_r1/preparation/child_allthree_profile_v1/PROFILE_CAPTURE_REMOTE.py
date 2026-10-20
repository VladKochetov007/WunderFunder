#!/usr/bin/env python3
"""Make one bounded read-only capture of the exact all-three profile unit."""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

REMOTE_ROOT = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
PRIVATE_DIR = REMOTE_ROOT / "research_tmp/fg19_child_allthree_profile_v1"
RUN_DIR = REMOTE_ROOT / "data/artifacts/runs/fg19_child_allthree_f0_s1001_profile_v1"
UNIT = "computer-vision-homework-fg19-child-allthree-profile-v1.service"
INVOCATION = "1d62c3fb47ac4bd5be8e8ec8fe6759cd"
GPU_UUID = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"
MAX_RECEIPT_BYTES = 1_000_000
STAGED_CODE = {
    "child_allthree_profile_contract_v1.py": "ab68d868261106a7548b649243f4a645ece926b6138d683c731841cfb9f0b7c5",
    "child_allthree_profile_probe_v1.py": "446e679ddae0f05225209950d928790264c8cd57fb8836b79f671d06d03be9e4",
    "propensity_terminal_support.py": "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
}


def _run(args: list[str], timeout: int = 10) -> dict[str, object]:
    """Run a bounded read-only query and preserve its outcome."""
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
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[:32768],
        "stderr": result.stderr[:32768],
        "timed_out": False,
    }


def _properties(query: dict[str, object]) -> dict[str, str]:
    """Parse systemd properties only when the bounded query succeeded."""
    if query["returncode"] != 0 or query["timed_out"]:
        return {}
    output = query["stdout"]
    if not isinstance(output, str):
        return {}
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def _show(unit: str) -> tuple[dict[str, object], dict[str, str]]:
    """Capture a fixed property set for one named user unit."""
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
    return query, _properties(query)


def _small_file(name: str, path: Path) -> dict[str, object]:
    """Read one named small receipt or log and preserve exact bytes."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"expected regular capture input is absent: {name}")
    size = path.stat().st_size
    if size > MAX_RECEIPT_BYTES:
        raise ValueError(f"capture input exceeds the {MAX_RECEIPT_BYTES}-byte bound: {name}")
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def _staged_code_pins() -> dict[str, dict[str, object]]:
    """Hash only the three small Python files staged into private D."""
    values: dict[str, dict[str, object]] = {}
    for name, expected_sha256 in STAGED_CODE.items():
        path = PRIVATE_DIR / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"staged source file is not a regular file: {name}")
        if path.stat().st_size > 200_000:
            raise ValueError(f"staged source file exceeds the bounded source-code size: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        values[name] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
            "expected_sha256": expected_sha256,
            "matches_expected": digest == expected_sha256,
        }
    return values


def _markers(raw_log: bytes) -> dict[str, object]:
    """Extract exact machine-readable profile markers without hiding the raw log."""
    result: dict[str, object] = {"parse_errors": [], "counts": {}}
    try:
        lines = raw_log.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        result["parse_errors"] = [f"unit.log is not UTF-8: {error}"]
        return result
    prefixes = (
        "CHILD_PROFILE_INITIAL_STATE ",
        "PROFILE ",
        "PROFILE_UPDATE_EVIDENCE ",
        "GRAD_STATS ",
    )
    counts = {prefix.rstrip(): 0 for prefix in prefixes}
    parsed: dict[str, list[object]] = {prefix.rstrip(): [] for prefix in prefixes[:-1]}
    errors: list[str] = []
    for line in lines:
        for prefix in prefixes:
            if line.startswith(prefix):
                key = prefix.rstrip()
                counts[key] += 1
                if key in parsed:
                    try:
                        parsed[key].append(json.loads(line[len(prefix) :]))
                    except json.JSONDecodeError as error:
                        errors.append(f"{key}: {error}")
                break
    result.update({"counts": counts, "records": parsed, "parse_errors": errors})
    return result


def _stat_inventory(path: Path) -> dict[str, object]:
    """Record names and stat metadata only; never open output files."""
    if path.is_symlink():
        return {"exists": True, "symlink": True, "entries": []}
    if not path.exists():
        return {"exists": False, "symlink": False, "entries": []}
    if not path.is_dir():
        return {"exists": True, "symlink": False, "not_directory": True, "entries": []}
    entries = list(path.iterdir())
    if len(entries) > 128:
        raise ValueError(f"stat inventory exceeds 128 entries: {path}")
    inventory: list[dict[str, object]] = []
    for item in sorted(entries, key=lambda value: value.name):
        value = item.lstat()
        inventory.append(
            {
                "name": item.name,
                "mode": value.st_mode,
                "is_symlink": item.is_symlink(),
                "bytes": value.st_size,
                "device": value.st_dev,
                "inode": value.st_ino,
                "mtime_ns": value.st_mtime_ns,
            }
        )
    return {"exists": True, "symlink": False, "entries": inventory}


def _read_cgroup(path: Path, names: list[str]) -> dict[str, str | None]:
    """Read a fixed set of small cgroup files when present."""
    values: dict[str, str | None] = {}
    for name in names:
        item = path / name
        values[name] = item.read_text(encoding="ascii").strip() if item.is_file() else None
    return values


def _parent_cgroup() -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    """Capture campaign-slice population and memory event counters."""
    query, properties = _show("wnn-campaign.slice")
    group = properties.get("ControlGroup", "")
    if not group or group == "/":
        return query, properties, {"error": "campaign slice has no bounded cgroup path"}
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    paths = [root / "cgroup.procs", *root.rglob("cgroup.procs")]
    if len(paths) > 128:
        raise ValueError("campaign cgroup tree exceeds the 128-file bound")
    populated: dict[str, str] = {}
    for item in paths:
        if item.is_file():
            value = item.read_text(encoding="ascii").strip()
            if value:
                populated[str(item)] = value
    return (
        query,
        properties,
        {
            "populated": populated,
            **_read_cgroup(root, ["memory.current", "memory.events", "memory.events.local"]),
        },
    )


def _process_rss(pid: str) -> list[str] | None:
    """Read RSS high-water lines for the unit's current main PID, if any."""
    if not pid.isdigit() or int(pid) <= 0:
        return None
    path = Path("/proc") / pid / "status"
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if len(raw) > 131072:
        raise ValueError("unit process status exceeds the bounded read size")
    return [
        line
        for line in raw.decode("ascii", errors="replace").splitlines()
        if line.startswith(("VmRSS:", "VmHWM:"))
    ]


def main() -> None:
    """Print one bounded observation; perform no acceptance or mutation."""
    unit_query, unit = _show(UNIT)
    parent_query, parent, parent_state = _parent_cgroup()
    group = unit.get("ControlGroup", "")
    own_group = Path("/sys/fs/cgroup") / group.lstrip("/") if group and group != "/" else None
    own_state = (
        _read_cgroup(
            own_group,
            [
                "memory.current",
                "memory.peak",
                "memory.events",
                "memory.events.local",
                "cgroup.procs",
            ],
        )
        if own_group is not None and own_group.exists()
        else None
    )
    receipts = {
        name: _small_file(name, PRIVATE_DIR / name)
        for name in ("ADMISSION.json", "LAUNCH.json", "unit.log")
    }
    if sum(item["bytes"] for item in receipts.values()) > 4_000_000:
        raise ValueError("capture receipt aggregate exceeds the 4 MB transport bound")
    log_raw = base64.b64decode(receipts["unit.log"]["base64"], validate=True)
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
    terminal_success = (
        unit.get("InvocationID") == INVOCATION
        and unit.get("ActiveState") == "active"
        and unit.get("SubState") == "exited"
        and unit.get("MainPID") == "0"
        and unit.get("ExecMainCode") == "1"
        and unit.get("ExecMainStatus") == "0"
        and unit.get("Result") == "success"
    )
    record = {
        "schema": "fg19-child-allthree-profile-capture.v1",
        "captured_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "expected_unit": UNIT,
        "expected_invocation_id": INVOCATION,
        "terminal_success": terminal_success,
        "unit_query": unit_query,
        "unit_properties": unit,
        "parent_query": parent_query,
        "parent_properties": parent,
        "parent_state": parent_state,
        "own_cgroup_state": own_state,
        "own_process_rss": _process_rss(unit.get("MainPID", "")),
        "gpu": gpu,
        "receipts": receipts,
        "staged_code_pins": _staged_code_pins(),
        "profile_log_markers": _markers(log_raw),
        "private_directory_stat_only": _stat_inventory(PRIVATE_DIR),
        "run_directory_stat_only": _stat_inventory(RUN_DIR),
        "scope": (
            "One bounded read-only capture of the exact unit, small receipts/log, profile markers, "
            "campaign and unit cgroups, selected GPU and stat-only run directory. No checkpoint, "
            "cache, dictionary, feature or data payload is opened or hashed; this record is not "
            "an acceptance decision."
        ),
    }
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
