from __future__ import annotations

import base64
import hashlib
import json
import os
import selectors
import stat
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

UNIT = "computer-vision-homework-fg19-first-cell-export-v1.service"
INVOCATION = "UNBOUND_ACTUAL_INVOCATION"
ADMISSION_BYTES = 0
ADMISSION_SHA256 = "UNBOUND_ACTUAL_ADMISSION"
LAUNCH_BYTES = 0
LAUNCH_SHA256 = "UNBOUND_ACTUAL_LAUNCH"
D = Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/"
    "fg19_first_cell_f0_s1001_evaluation_v1"
)
GPU_UUID = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"


def read_regular(path: Path, cap: int, *, tail: bool = False) -> tuple[bytes, dict[str, int]]:
    """Read bounded metadata or a log tail; never open prediction or model arrays."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(f"unsafe metadata file: {path}")
    if not tail and before.st_size > cap:
        raise ValueError(f"metadata exceeds cap: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"file changed before open: {path}")
        if tail:
            os.lseek(fd, max(0, opened.st_size - cap), os.SEEK_SET)
        raw = bytearray()
        while len(raw) <= cap:
            block = os.read(fd, min(65536, cap + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if len(raw) > cap:
        raise ValueError("bounded metadata read exceeded cap")
    if not tail and (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("metadata changed while read")
    return bytes(raw), {
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def encoded(raw: bytes) -> dict[str, object]:
    """Describe exact captured bytes without JSON re-serialization."""
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def show(unit: str, fields: tuple[str, ...]) -> dict[str, str]:
    """Inspect one exact unit with explicit empty properties retained."""
    command = ["systemctl", "--user", "show", "--all", unit]
    for field in fields:
        command += ["-p", field]
    result = subprocess.run(command, capture_output=True, text=True, timeout=12, check=False)
    if result.returncode:
        raise RuntimeError(f"exact unit query failed: {unit}")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def cgroup_snapshot(control_group: str) -> dict[str, object] | None:
    """Read scoped counters and all populated descendants; root cgroup is forbidden."""
    if control_group in ("", "/"):
        return None
    root = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    if not root.is_dir():
        return None
    if root.resolve(strict=True) != root:
        raise ValueError("cgroup redirected")
    proc_files = [root / "cgroup.procs", *root.rglob("cgroup.procs")]
    if len(proc_files) > 128:
        raise ValueError("cgroup descendant count exceeded bound")
    populated = {}
    for path in proc_files:
        value = path.read_text(encoding="ascii")
        if value.strip():
            populated[str(path)] = value
    counters = {}
    for name in ("memory.current", "memory.peak", "memory.events", "memory.events.local"):
        path = root / name
        counters[name] = path.read_text(encoding="ascii") if path.exists() else None
    return {"path": str(root), "populated": populated, "counters": counters}


def output_inventory() -> list[dict[str, object]]:
    """Stat fixed export paths only, without reading or hashing array payloads."""
    paths = [D / "exports", D / "exports/EXPORT_MANIFEST_V1.json"]
    for arm in ("control", "ema", "sparse", "nll", "allthree"):
        paths += [
            D / "exports" / arm,
            D / "exports" / arm / "predictions.npy",
            D / "exports" / arm / "PREDICTION_SIDECAR.json",
        ]
    rows = []
    for path in paths:
        if not path.exists():
            rows.append({"path": str(path), "exists": False})
            continue
        observed = path.lstat()
        if path.is_symlink():
            raise ValueError("output path is a symlink")
        rows.append(
            {
                "path": str(path),
                "exists": True,
                "bytes": observed.st_size,
                "device": observed.st_dev,
                "inode": observed.st_ino,
                "mtime_ns": observed.st_mtime_ns,
                "is_directory": stat.S_ISDIR(observed.st_mode),
            }
        )
    return rows


def require(condition: object, message: str) -> None:
    """Reject malformed bounded GPU metadata."""
    if not condition:
        raise RuntimeError(message)


def _bounded_gpu_query(args: list[str]) -> tuple[str, str]:
    """Run one bounded nvidia-smi query without shell interpretation."""
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(
        process.stdout is not None and process.stderr is not None, "GPU query pipes unavailable"
    )
    output = {"stdout": bytearray(), "stderr": bytearray()}
    streams = {process.stdout: "stdout", process.stderr: "stderr"}
    deadline = time.monotonic() + 15
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                require(remaining > 0, "selected-GPU query timed out")
                events = selector.select(remaining)
                require(events, "selected-GPU query timed out")
                for key, _ in events:
                    label = streams[key.fileobj]
                    chunk = os.read(key.fileobj.fileno(), min(4096, 65537 - len(output[label])))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output[label].extend(chunk)
                    require(
                        len(output[label]) <= 65536, "selected-GPU query output exceeded 64 KiB"
                    )
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        require(
            returncode == 0,
            "selected-GPU query returned nonzero: "
            + output["stderr"].decode("utf-8", errors="replace")[:512],
        )
    except (OSError, subprocess.TimeoutExpired, RuntimeError):
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
        process.stderr.close()
    return output["stdout"].decode("utf-8", errors="strict"), output["stderr"].decode(
        "utf-8", errors="replace"
    )


def _bounded_proc_read(path: Path, maximum: int) -> bytes:
    """Read a proc metadata file with no-follow and an explicit byte ceiling."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, f"proc metadata exceeded {maximum} bytes: {path.name}")
    return raw


def _proc_starttime(proc: Path, pid: str) -> str:
    """Return Linux /proc stat starttime, distinguishing PID reuse."""
    text = _bounded_proc_read(proc / "stat", 8192).decode("ascii", errors="strict")
    prefix, separator, remainder = text.rpartition(")")
    require(separator == ")" and prefix.startswith(pid + " ("), "proc stat identity is malformed")
    fields = remainder.strip().split()
    require(len(fields) > 19 and fields[19].isdigit(), "proc stat starttime is unavailable")
    return fields[19]


def observe_gpu(unit_cgroup: str) -> dict[str, object]:
    """Snapshot selected-GPU owners without modifying any process."""
    prefix = ["nvidia-smi", "-i", GPU_UUID]
    identity, _ = _bounded_gpu_query(
        prefix + ["--query-gpu=uuid,name,memory.free,memory.total", "--format=csv,noheader"]
    )
    rows, _ = _bounded_gpu_query(
        prefix + ["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"]
    )
    lines = [line.strip() for line in rows.splitlines() if line.strip()]
    if len(lines) == 1 and lines[0].startswith("No running processes found"):
        lines = []
    require(len(lines) <= 256, "GPU process inventory exceeds cap")
    owners = []
    for line in lines:
        columns = [part.strip() for part in line.split(",")]
        require(
            len(columns) == 3 and columns[0].isdigit() and columns[1] == GPU_UUID,
            "GPU process row differs from selected UUID",
        )
        proc = Path("/proc") / columns[0]
        before = proc.stat(follow_symlinks=False)
        start = _proc_starttime(proc, columns[0])
        cgroup = _bounded_proc_read(proc / "cgroup", 32768).decode("ascii").strip()
        after = proc.stat(follow_symlinks=False)
        require(
            (before.st_ino, before.st_uid, start)
            == (after.st_ino, after.st_uid, _proc_starttime(proc, columns[0])),
            "GPU PID identity changed during observation",
        )
        owners.append(
            {
                "pid": int(columns[0]),
                "gpu_uuid": columns[1],
                "used_memory": columns[2],
                "uid": after.st_uid,
                "cgroup": cgroup,
                "exact_export_owner": bool(unit_cgroup)
                and unit_cgroup not in ("", "/")
                and cgroup == "0::" + unit_cgroup
                and after.st_uid == os.getuid(),
                "process_action": "none",
            }
        )
    return {"identity_sample": identity, "process_owners": owners}


def main() -> None:
    """Observe one bound export invocation, capturing metadata only on terminal success."""
    if len(INVOCATION) != 32 or any(c not in "0123456789abcdef" for c in INVOCATION):
        raise ValueError("actual invocation remains unbound")
    if D.is_symlink() or D.resolve(strict=True) != D or not D.is_dir():
        raise ValueError("private export evidence root is redirected or unavailable")
    for name, size, expected_sha in (
        ("ADMISSION.json", ADMISSION_BYTES, ADMISSION_SHA256),
        ("LAUNCH.json", LAUNCH_BYTES, LAUNCH_SHA256),
    ):
        raw, _ = read_regular(D / name, 1_000_000)
        if (
            type(size) is not int
            or size <= 0
            or (len(raw), hashlib.sha256(raw).hexdigest()) != (size, expected_sha)
        ):
            raise ValueError(f"original export receipt differs: {name}")
        value = json.loads(raw)
        if value.get("unit") != UNIT:
            raise ValueError("original receipt names a different unit")
        if name == "LAUNCH.json" and value.get("invocation_id") != INVOCATION:
            raise ValueError("original launch names a different invocation")
    props = show(
        UNIT,
        (
            "Id",
            "LoadState",
            "ActiveState",
            "SubState",
            "MainPID",
            "ExecMainCode",
            "ExecMainStatus",
            "InvocationID",
            "Result",
            "ControlGroup",
            "MemoryCurrent",
            "MemoryPeak",
            "CPUUsageNSec",
            "ExecMainStartTimestamp",
            "ExecMainExitTimestamp",
        ),
    )
    if props.get("Id") != UNIT or props.get("InvocationID") != INVOCATION:
        raise ValueError("actual unit invocation differs")
    log, log_stat = read_regular(D / "unit.log", 262144, tail=True)
    parent = show("wnn-campaign.slice", ("ControlGroup", "MemoryCurrent"))
    result = {
        "schema": "fg19.first-cell-export-bounded-status.v1",
        "captured_utc": datetime.now(UTC).isoformat(),
        "unit": UNIT,
        "invocation_id": INVOCATION,
        "systemd": props,
        "own_cgroup": cgroup_snapshot(props.get("ControlGroup", "")),
        "parent_cgroup": cgroup_snapshot(parent.get("ControlGroup", "")),
        "selected_gpu": observe_gpu(props.get("ControlGroup", "")),
        "log_tail": {**encoded(log), "file_stat_separate_sample": log_stat},
        "output_stat_only": output_inventory(),
        "terminal_metadata": {},
        "array_payloads_read_or_hashed": False,
    }
    for name in ("ADMISSION.json", "LAUNCH.json"):
        raw, _ = read_regular(D / name, 1_000_000)
        result[name] = encoded(raw)
    is_success = (
        props.get("ActiveState"),
        props.get("SubState"),
        props.get("MainPID"),
        props.get("ExecMainCode"),
        props.get("ExecMainStatus"),
        props.get("Result"),
    ) == ("active", "exited", "0", "1", "0", "success")
    if is_success:
        raw, _ = read_regular(D / "exports/EXPORT_MANIFEST_V1.json", 1_000_000)
        result["terminal_metadata"]["EXPORT_MANIFEST_V1.json"] = encoded(raw)
        for arm in ("control", "ema", "sparse", "nll", "allthree"):
            raw, _ = read_regular(D / "exports" / arm / "PREDICTION_SIDECAR.json", 100000)
            result["terminal_metadata"][f"{arm}/PREDICTION_SIDECAR.json"] = encoded(raw)
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
