"""Bounded read-only capture of the fixed first-cell score unit and scalar result."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
SCRATCH_ROOT = PROJECT_BASE / "research_tmp"
SCORE_D = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_score_v1"
EXPORT_D = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_evaluation_v3"
RESULT_PATH = EXPORT_D / "RESULT.json"
SCORE_UNIT = "computer-vision-homework-fg19-first-cell-score-v1.service"
EXPORT_UNIT = "computer-vision-homework-fg19-first-cell-export-v3.service"

# Root binds these five literals only from this exact future score admission/launch.
SCORE_ADMISSION_BYTES = 69650
SCORE_ADMISSION_SHA256 = '1252b2f1d39ef255fac83c722cfc1a380705963a0b2f9dede35dcabed45ba631'
SCORE_LAUNCH_BYTES = 4193
SCORE_LAUNCH_SHA256 = '8386643a1b5d80cb3cc69d332d6e822c8864c1c29281edeca8a9b0daf2c63818'
SCORE_INVOCATION_ID = 'fbc58e9d2a7e47629c6a8feb6302aae8'

MAX_CAPTURE_FILE_BYTES = 1_000_000
MAX_CAPTURE_AGGREGATE_BYTES = 4_000_000
MAX_LOG_TAIL_BYTES = 262_144
CGROUP_ROOT = Path("/sys/fs/cgroup")
GPU_UUID = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"
MAX_GPU_QUERY_BYTES = 65_536
MAX_GPU_PROCESS_ROWS = 64
MAX_CGROUP_DIRECTORIES = 128
MAX_CGROUP_ENTRIES_PER_DIRECTORY = 256
MAX_CGROUP_PROCS_FILE_BYTES = 32_768
MAX_CGROUP_PROCS_AGGREGATE_BYTES = 262_144
MAX_CGROUP_METADATA_FILE_BYTES = 4_096
MAX_PROC_CGROUP_BYTES = 4_096
MAX_COMMAND_CHUNK_BYTES = 4_096

SCORER_CODE_PINS = {
    "first_cell_score_contract.py": (
        20_165,
        "b6e3f7a9f9663f519037c54a2077f70d83fbf0b2ba54dc3e158eb5e3b173479d",
    ),
    "first_cell_score_core.py": (
        27_226,
        "30c456e7db875e2c075a6200c857c375087b18d9c952c354ae9bd3dceb0a0143",
    ),
    "first_cell_score_predictions.py": (
        11_056,
        "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d",
    ),
    "score_first_cell.py": (
        10_630,
        "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6",
    ),
}

UNIT_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "InvocationID",
    "Result",
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
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
    "ExecMainStartTimestampMonotonic",
    "ExecMainExitTimestampMonotonic",
)


def _require_bound_pins() -> None:
    """Fail closed before any path, process, or systemd query while unbound."""
    for label, byte_count, digest in (
        ("admission", SCORE_ADMISSION_BYTES, SCORE_ADMISSION_SHA256),
        ("launch", SCORE_LAUNCH_BYTES, SCORE_LAUNCH_SHA256),
    ):
        if type(byte_count) is not int or byte_count <= 0:
            raise RuntimeError(f"root has not bound exact score {label} bytes")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"root has not bound exact score {label} SHA-256")
    if re.fullmatch(r"[0-9a-f]{32}", SCORE_INVOCATION_ID) is None:
        raise RuntimeError("root has not bound the actual score invocation ID")


def _read_regular(path: Path, byte_cap: int) -> bytes:
    """Read a bounded regular file without following its final symlink."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > byte_cap:
        raise ValueError(f"unsafe or oversized capture input: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError(f"capture input changed before read: {path}")
        parts: list[bytes] = []
        remaining = byte_cap + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            parts.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(parts)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or len(raw) > byte_cap:
        raise ValueError(f"capture input changed size during read: {path}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        raise ValueError(f"capture input changed during read: {path}")
    return raw


def _read_pinned_receipt(name: str, byte_count: int, digest: str) -> bytes:
    raw = _read_regular(SCORE_D / name, MAX_CAPTURE_FILE_BYTES)
    if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"score receipt bytes/SHA-256 mismatch: {name}")
    return raw


def _json_object(raw: bytes, label: str) -> dict[str, object]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _validate_receipt_metadata(
    admission_raw: bytes, launch_raw: bytes, expected_invocation_id: str
) -> dict[str, object]:
    """Join score receipt bytes to the fixed unit, invocation, and result path."""
    admission = _json_object(admission_raw, "score admission")
    launch = _json_object(launch_raw, "score launch")
    admission_sha = hashlib.sha256(admission_raw).hexdigest()
    if (
        admission.get("schema") != "fg19.first-cell-score-admission.v1"
        or admission.get("status") != "FRESH_GATES_PASSED_SUBMISSION_PENDING"
        or admission.get("unit") != SCORE_UNIT
        or admission.get("result_path") != str(RESULT_PATH)
        or admission.get("export_unit") != EXPORT_UNIT
    ):
        raise ValueError("score admission identity/schema/result path mismatch")
    export_invocation = admission.get("export_invocation_id")
    if (
        not isinstance(export_invocation, str)
        or re.fullmatch(r"[0-9a-f]{32}", export_invocation) is None
    ):
        raise ValueError("score admission export invocation ID is malformed")
    if admission.get("result_create_only_and_absent_before_mutation") is not True:
        raise ValueError("score admission does not bind a create-only absent result path")
    for key, staged_path in (
        ("export_acceptance", "accepted_export/EXPORT_ACCEPTANCE.json"),
        ("export_exact_stop", "accepted_export/EXPORT_EXACT_STOP.json"),
    ):
        pin = admission.get(key)
        if (
            not isinstance(pin, dict)
            or type(pin.get("bytes")) is not int
            or pin["bytes"] <= 0
            or not isinstance(pin.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) is None
            or pin.get("staged_path") != staged_path
        ):
            raise ValueError(f"score admission {key} receipt binding is malformed")
    if (
        launch.get("schema") != "fg19.first-cell-score-launch.v1"
        or launch.get("unit") != SCORE_UNIT
        or launch.get("invocation_id") != expected_invocation_id
        or launch.get("export_unit") != EXPORT_UNIT
        or launch.get("export_invocation_id") != export_invocation
        or launch.get("admission_sha256") != admission_sha
    ):
        raise ValueError("score launch identity/admission binding mismatch")
    submission = launch.get("submission")
    if (
        not isinstance(submission, dict)
        or submission.get("outcome") != "submitted"
        or type(submission.get("returncode")) is not int
        or submission.get("returncode") != 0
    ):
        raise ValueError("score launch does not record one confirmed submission")
    staged = admission.get("staged_files")
    if not isinstance(staged, dict):
        raise ValueError("score admission staged-file map is malformed")
    code_metadata: dict[str, dict[str, object]] = {}
    for name, (expected_bytes, expected_sha) in SCORER_CODE_PINS.items():
        receipt = staged.get(name)
        if not isinstance(receipt, dict) or (receipt.get("bytes"), receipt.get("sha256")) != (
            expected_bytes,
            expected_sha,
        ):
            raise ValueError(f"score admission staged-code pin differs: {name}")
        code_metadata[name] = {"bytes": expected_bytes, "sha256": expected_sha}
    return {
        "admission_sha256": admission_sha,
        "export_invocation_id": export_invocation,
        "staged_code_pins": code_metadata,
    }


def _verify_code_pins(staged_code_pins: object) -> dict[str, dict[str, object]]:
    """Verify each of the four admitted scorer modules without importing it."""
    if not isinstance(staged_code_pins, dict):
        raise ValueError("score admission staged-code pin map is malformed")
    verified: dict[str, dict[str, object]] = {}
    for name, (expected_bytes, expected_sha) in SCORER_CODE_PINS.items():
        expected = staged_code_pins.get(name)
        if not isinstance(expected, dict) or (
            expected.get("bytes"),
            expected.get("sha256"),
        ) != (expected_bytes, expected_sha):
            raise ValueError(f"score admission staged-code pin differs: {name}")
        raw = _read_regular(SCORE_D / name, MAX_CAPTURE_FILE_BYTES)
        actual_sha = hashlib.sha256(raw).hexdigest()
        if len(raw) != expected_bytes or actual_sha != expected_sha:
            raise ValueError(f"private scorer code bytes/SHA-256 mismatch: {name}")
        verified[name] = {"bytes": len(raw), "sha256": actual_sha}
    return verified


def _show(unit: str) -> dict[str, str]:
    command = ["systemctl", "--user", "show", "--all", unit]
    for key in UNIT_PROPERTIES:
        command.extend(("-p", key))
    returncode, stdout, _stderr = _run_bounded_command(
        command, timeout_seconds=10, stdout_byte_cap=65_536, stderr_byte_cap=4_096
    )
    if returncode != 0:
        raise RuntimeError(f"systemctl show failed or exceeded bound for exact unit {unit}")
    return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)


def _run_bounded_command(
    command: list[str], timeout_seconds: float, stdout_byte_cap: int, stderr_byte_cap: int
) -> tuple[int, str, str]:
    """Capture one child command with byte limits enforced while reading pipes."""
    if timeout_seconds <= 0 or stdout_byte_cap < 0 or stderr_byte_cap < 0:
        raise ValueError("invalid bounded-command limits")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        bufsize=0,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError("bounded command pipes were not created")
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": stdout_byte_cap, "stderr": stderr_byte_cap}
    streams = {process.stdout: "stdout", process.stderr: "stderr"}
    deadline = time.monotonic() + timeout_seconds
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("bounded command exceeded its deadline")
                events = selector.select(remaining)
                if not events:
                    raise TimeoutError("bounded command exceeded its deadline")
                for key, _mask in events:
                    label = streams[key.fileobj]
                    remaining_bytes = limits[label] + 1 - len(outputs[label])
                    try:
                        chunk = os.read(
                            key.fileobj.fileno(), min(MAX_COMMAND_CHUNK_BYTES, remaining_bytes)
                        )
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    outputs[label].extend(chunk)
                    if len(outputs[label]) > limits[label]:
                        raise ValueError(f"bounded command {label} exceeded its byte cap")
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except (OSError, subprocess.TimeoutExpired, TimeoutError, ValueError) as error:
        raise RuntimeError("bounded command failed its time or output limit") from error
    finally:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
    try:
        stdout = bytes(outputs["stdout"]).decode("utf-8", errors="strict")
        stderr = bytes(outputs["stderr"]).decode("utf-8", errors="replace")
    except UnicodeDecodeError as error:
        raise RuntimeError("bounded command stdout was not valid UTF-8") from error
    return returncode, stdout, stderr


def _checked_cgroup(control_group: str) -> Path:
    if not control_group.startswith("/") or ".." in Path(control_group).parts:
        raise ValueError("invalid systemd ControlGroup path")
    group = CGROUP_ROOT / control_group.lstrip("/")
    if group == CGROUP_ROOT or not group.is_dir():
        raise ValueError("systemd ControlGroup is absent or resolves to cgroup root")
    group.relative_to(CGROUP_ROOT)
    return group


def _cgroup_populated(group: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    stack = [group]
    visited = 0
    aggregate_bytes = 0
    while stack:
        current = stack.pop()
        visited += 1
        if visited > MAX_CGROUP_DIRECTORIES:
            raise ValueError("parent cgroup directory scan exceeded bound")
        proc_path = current / "cgroup.procs"
        try:
            raw = _bounded_metadata_read(proc_path, MAX_CGROUP_PROCS_FILE_BYTES)
        except FileNotFoundError:
            pass
        else:
            aggregate_bytes += len(raw)
            if aggregate_bytes > MAX_CGROUP_PROCS_AGGREGATE_BYTES:
                raise ValueError("parent cgroup process inventory exceeded aggregate byte cap")
            result[str(proc_path)] = raw.decode("ascii", errors="strict").strip()
        with os.scandir(current) as entries:
            for entry_count, entry in enumerate(entries, start=1):
                if entry_count > MAX_CGROUP_ENTRIES_PER_DIRECTORY:
                    raise ValueError("one parent cgroup directory exceeded entry cap")
                if entry.is_symlink():
                    raise ValueError("parent cgroup scan encountered a symlink")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(current / entry.name)
                    if visited + len(stack) > MAX_CGROUP_DIRECTORIES:
                        raise ValueError("parent cgroup directory scan exceeded bound")
    return result


def _cgroup_text(group: Path, names: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in names:
        path = group / name
        try:
            raw = _bounded_metadata_read(path, MAX_CGROUP_METADATA_FILE_BYTES)
        except FileNotFoundError:
            continue
        values[name] = raw.decode("ascii", errors="strict")
    return values


def _bounded_metadata_read(path: Path, byte_cap: int) -> bytes:
    """Read one metadata file with no-follow and a hard pre-accumulation cap."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"metadata path is not a regular file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"metadata path changed before read: {path.name}")
        chunks: list[bytes] = []
        total = 0
        while total <= byte_cap:
            chunk = os.read(descriptor, min(MAX_COMMAND_CHUNK_BYTES, byte_cap + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    finally:
        os.close(descriptor)
    if total > byte_cap:
        raise ValueError(f"metadata file exceeded {byte_cap} bytes: {path.name}")
    return b"".join(chunks)


def _gpu_identity() -> dict[str, object]:
    """Capture bounded selected-GPU identity and process ownership metadata only."""
    prefix = ["nvidia-smi", "-i", GPU_UUID]
    device_returncode, device_stdout, device_stderr = _run_bounded_command(
        prefix + ["--query-gpu=uuid,name,memory.free,memory.total", "--format=csv,noheader"],
        timeout_seconds=10,
        stdout_byte_cap=4_096,
        stderr_byte_cap=4_096,
    )
    if device_returncode != 0:
        raise RuntimeError("selected GPU identity query failed or exceeded its bound")
    rows = [line.strip() for line in device_stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("selected GPU query did not return exactly one device row")
    columns = [item.strip() for item in rows[0].split(",")]
    if len(columns) != 4 or columns[0] != GPU_UUID:
        raise ValueError("selected GPU identity differs from the pinned UUID")
    processes_returncode, processes_stdout, processes_stderr = _run_bounded_command(
        prefix + ["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
        timeout_seconds=10,
        stdout_byte_cap=MAX_GPU_QUERY_BYTES,
        stderr_byte_cap=4_096,
    )
    if processes_returncode != 0:
        raise RuntimeError("selected GPU process query failed or exceeded its bound")
    identities: list[dict[str, object]] = []
    for row in processes_stdout.splitlines():
        row = row.strip()
        if not row or row.startswith("No running processes found"):
            continue
        if len(identities) >= MAX_GPU_PROCESS_ROWS:
            raise ValueError("selected GPU process rows exceeded the fixed bound")
        parts = [item.strip() for item in row.split(",")]
        if len(parts) != 3 or not parts[0].isdigit() or parts[1] != GPU_UUID:
            raise ValueError("selected GPU process identity row is malformed")
        pid = int(parts[0])
        if pid <= 0:
            raise ValueError("selected GPU process PID is not positive")
        proc = Path("/proc") / str(pid)
        try:
            cgroup = _bounded_metadata_read(proc / "cgroup", MAX_PROC_CGROUP_BYTES).decode(
                "ascii", errors="strict"
            )
            uid = proc.stat().st_uid
        except OSError as error:
            raise RuntimeError(f"selected GPU PID identity became unavailable: {pid}") from error
        else:
            identities.append(
                {
                    "pid": pid,
                    "gpu_uuid": parts[1],
                    "used_memory": parts[2],
                    "uid": uid,
                    "cgroup": cgroup.strip(),
                }
            )
    return {
        "selected_gpu_uuid": GPU_UUID,
        "device_query": {
            "returncode": device_returncode,
            "stdout": device_stdout,
            "stderr": device_stderr,
        },
        "process_query": {
            "returncode": processes_returncode,
            "stdout": processes_stdout,
            "stderr": processes_stderr,
        },
        "processes": identities,
    }


def _file_record(raw: bytes) -> dict[str, object]:
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def _capture_log_tail(path: Path = SCORE_D / "unit.log") -> dict[str, object]:
    """Capture a bounded stable tail of the exact score unit log."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("score unit log is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError("score unit log changed before tail capture")
        captured_bytes = min(before.st_size, MAX_LOG_TAIL_BYTES)
        offset = before.st_size - captured_bytes
        os.lseek(descriptor, offset, os.SEEK_SET)
        tail = os.read(descriptor, captured_bytes + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(tail) != captured_bytes
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity
    ):
        raise ValueError("score unit log changed during tail capture")
    return {
        "bytes": before.st_size,
        "captured_bytes": len(tail),
        "tail_offset_bytes": offset,
        "tail_sha256": hashlib.sha256(tail).hexdigest(),
        "tail_base64": base64.b64encode(tail).decode("ascii"),
    }


def _require_exact_unit_identity(unit: dict[str, str]) -> None:
    """Reject a systemd response whose Id is not the requested score unit."""
    if unit.get("Id") != SCORE_UNIT:
        raise ValueError("systemd returned a different score unit Id")


def _strict_terminal_success(unit: dict[str, str]) -> bool:
    """Return true only for the exact successful systemd terminal tuple."""
    return unit.get("Result") == "success" and (
        unit.get("ActiveState"),
        unit.get("SubState"),
        unit.get("MainPID"),
        unit.get("ExecMainCode"),
        unit.get("ExecMainStatus"),
    ) == ("active", "exited", "0", "1", "0")


def _capture_result(terminal_success: bool, result_path: Path = RESULT_PATH) -> dict[str, object]:
    """Read the scalar result file only after strict successful unit completion."""
    if not terminal_success:
        return {"read": False, "reason": "unit_not_strict_success"}
    try:
        raw = _read_regular(result_path, MAX_CAPTURE_FILE_BYTES)
    except FileNotFoundError:
        return {"read": False, "reason": "result_missing_after_success"}
    except (OSError, ValueError) as error:
        return {
            "read": False,
            "reason": "result_not_safely_captured",
            "error_type": type(error).__name__,
        }
    return {"read": True, **_file_record(raw)}


def main() -> None:
    """Capture exact score metadata and, on success only, the compact scalar result."""
    _require_bound_pins()
    directory_stat = SCORE_D.lstat()
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError("private score directory is not a real directory")
    admission_raw = _read_pinned_receipt(
        "ADMISSION.json", SCORE_ADMISSION_BYTES, SCORE_ADMISSION_SHA256
    )
    launch_raw = _read_pinned_receipt("LAUNCH.json", SCORE_LAUNCH_BYTES, SCORE_LAUNCH_SHA256)
    receipt_metadata = _validate_receipt_metadata(admission_raw, launch_raw, SCORE_INVOCATION_ID)
    code_pins = _verify_code_pins(receipt_metadata["staged_code_pins"])
    unit = _show(SCORE_UNIT)
    _require_exact_unit_identity(unit)
    if unit.get("InvocationID") != SCORE_INVOCATION_ID:
        raise ValueError("live score unit invocation differs from pinned launch")
    parent = _show("wnn-campaign.slice")
    parent_control_group = parent.get("ControlGroup", "")
    if not parent_control_group.endswith("/wnn-campaign.slice"):
        raise ValueError("campaign parent slice ControlGroup mismatch")
    parent_group = _checked_cgroup(parent_control_group)
    populated = _cgroup_populated(parent_group)
    parent_events = _cgroup_text(
        parent_group, ("memory.current", "memory.events", "memory.events.local")
    )
    own = None
    unit_control_group = unit.get("ControlGroup", "")
    if unit_control_group not in ("", "/"):
        if not unit_control_group.startswith(parent_control_group.rstrip("/") + "/"):
            raise ValueError("score unit cgroup is not inside the campaign slice")
        own = _cgroup_text(
            _checked_cgroup(unit_control_group),
            ("memory.current", "memory.peak", "memory.events", "memory.events.local"),
        )
    gpu = _gpu_identity()
    files = {
        "ADMISSION.json": _file_record(admission_raw),
        "LAUNCH.json": _file_record(launch_raw),
        "unit.log": _capture_log_tail(),
    }
    terminal_success = _strict_terminal_success(unit)
    result_capture = _capture_result(terminal_success)
    if result_capture.get("read") is True:
        files["RESULT.json"] = result_capture
    captured_bytes = sum(
        int(value.get("captured_bytes" if name == "unit.log" else "bytes", 0))
        for name, value in files.items()
    )
    if captured_bytes > MAX_CAPTURE_AGGREGATE_BYTES:
        raise ValueError("score result capture aggregate exceeds 4 MB")
    response = {
        "schema": "fg19.first-cell-score-bounded-status.v1",
        "captured_utc": datetime.now(UTC).isoformat(),
        "unit_name": SCORE_UNIT,
        "invocation_id": SCORE_INVOCATION_ID,
        "terminal_success": terminal_success,
        "unit": unit,
        "parent": parent,
        "parent_events": parent_events,
        "own_cgroup_files": own,
        "parent_populated_files": populated,
        "selected_gpu": gpu,
        "files": files,
        "result_capture": {key: value for key, value in result_capture.items() if key != "base64"},
        "code_pins": code_pins,
        "scope": "Read-only bounded score-unit metadata, pinned receipts, four scorer-code pins, unit.log tail, and only on strict systemd success the exact export-root RESULT.json bytes; no predictions, targets, moments, model, cache, source, or checkpoint payloads are opened.",
    }
    print(json.dumps(response, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
