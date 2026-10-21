"""Conditionally stop only the accepted first-cell score unit after exit."""

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

SCORE_UNIT = "computer-vision-homework-fg19-first-cell-score-v3.service"
SCORE_INVOCATION = "UNBOUND_ROOT_SCORE_INVOCATION"
ACCEPTANCE_BYTES = 0
ACCEPTANCE_SHA256 = "UNBOUND_ROOT_SCORE_ACCEPTANCE_SHA256"
ACCEPTANCE_BASE64 = "UNBOUND_ROOT_SCORE_ACCEPTANCE_BASE64"
ACCEPTANCE_SCHEMA = "fg19.first-cell-score-root-acceptance.v1"
ACCEPTANCE_STATUS = "ACCEPTED_FIXED_FIRST_CELL_DEVELOPMENT_SCORE_REPORT"
RESULT_SCHEMA = "fg19.first-cell-five-arm-streaming-score.v1"
RESULT_STATUS = "FIXED_FIRST_CELL_DEVELOPMENT_EVIDENCE_SCORED"
CGROUP_ROOT = Path("/sys/fs/cgroup")
MAX_ACCEPTANCE_BYTES = 1_000_000
MAX_COMMAND_CHUNK_BYTES = 4_096
MAX_CGROUP_DIRECTORIES = 128
MAX_CGROUP_ENTRIES_PER_DIRECTORY = 256
MAX_CGROUP_PROCS_FILE_BYTES = 32_768
MAX_CGROUP_PROCS_AGGREGATE_BYTES = 262_144
UNIT_PROPERTIES = ("Id", "LoadState", "ActiveState", "SubState", "MainPID") + (
    "InvocationID",
    "ExecMainCode",
    "ExecMainStatus",
    "Result",
    "ControlGroup",
)


def _is_hex(value: object, size: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == size
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_receipt(
    raw: bytes, byte_count: int, acceptance_sha: str, invocation: str
) -> tuple[str, dict[str, object]]:
    if (
        type(byte_count) is not int
        or not 0 < byte_count <= MAX_ACCEPTANCE_BYTES
        or not _is_hex(acceptance_sha, 64)
        or not _is_hex(invocation, 32)
    ):
        raise ValueError("score stop acceptance or invocation is unbound")
    digest = hashlib.sha256(raw).hexdigest()
    if (len(raw), digest) != (byte_count, acceptance_sha):
        raise ValueError("ROOT score acceptance bytes/SHA-256 differ")
    acceptance = json.loads(raw)
    if not isinstance(acceptance, dict):
        raise ValueError("ROOT score acceptance must be a JSON object")
    native = acceptance.get("native_terminal")
    result = acceptance.get("result")
    if not isinstance(native, dict) or not isinstance(result, dict):
        raise ValueError("ROOT score acceptance lacks native/result objects")
    if any(
        acceptance.get(key) != value
        for key, value in {
            "schema": ACCEPTANCE_SCHEMA,
            "status": ACCEPTANCE_STATUS,
            "unit": SCORE_UNIT,
            "invocation_id": invocation,
            "stop_status": "PENDING_EXACT_STOP",
        }.items()
    ):
        raise ValueError("ROOT score acceptance identity/status differs")
    if (
        native.get("active_state") != "active"
        or native.get("sub_state") != "exited"
        or type(native.get("main_pid")) is not int
        or native["main_pid"] != 0
        or type(native.get("exec_main_code")) is not int
        or native["exec_main_code"] != 1
        or type(native.get("exec_main_status")) is not int
        or native["exec_main_status"] != 0
    ):
        raise ValueError("ROOT acceptance lacks strict successful native exit facts")
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != RESULT_STATUS
        or type(result.get("bytes")) is not int
        or result["bytes"] <= 0
        or not _is_hex(result.get("sha256"), 64)
    ):
        raise ValueError("ROOT acceptance lacks the fixed score-result pin")
    if not isinstance(result.get("path"), str) or not result["path"]:
        raise ValueError("ROOT acceptance result path is missing")
    return digest, result


def _run_bounded(
    command: list[str], timeout: float, stdout_cap: int, stderr_cap: int
) -> tuple[int, str, str]:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError("systemd command pipes were unavailable")
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": stdout_cap, "stderr": stderr_cap}
    streams = {process.stdout: "stdout", process.stderr: "stderr"}
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("systemd command deadline exceeded")
                for key, _ in selector.select(remaining):
                    name = streams[key.fileobj]
                    capacity = limits[name] + 1 - len(outputs[name])
                    chunk = os.read(key.fileobj.fileno(), min(MAX_COMMAND_CHUNK_BYTES, capacity))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    outputs[name].extend(chunk)
                    if len(outputs[name]) > limits[name]:
                        raise ValueError("systemd command output exceeded its bound")
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=1)
        for stream in streams:
            stream.close()
    return (
        returncode,
        bytes(outputs["stdout"]).decode("utf-8", errors="strict"),
        bytes(outputs["stderr"]).decode("utf-8", errors="replace"),
    )


def _show(unit: str) -> dict[str, str]:
    if unit not in (SCORE_UNIT, "wnn-campaign.slice"):
        raise ValueError("unit outside the exact score stop scope")
    command = ["systemctl", "--user", "show", "--all", unit]
    for key in UNIT_PROPERTIES:
        command.extend(("-p", key))
    returncode, stdout, _ = _run_bounded(command, 12, 65_536, 4_096)
    if returncode:
        raise RuntimeError("exact systemd show failed")
    return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)


def _metadata_read(path: Path, byte_cap: int) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("cgroup metadata is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        if (opened.st_dev, opened.st_ino) != identity:
            raise ValueError("cgroup metadata changed before read")
        data = bytearray()
        while len(data) <= byte_cap:
            chunk = os.read(descriptor, min(MAX_COMMAND_CHUNK_BYTES, byte_cap + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
    finally:
        os.close(descriptor)
    if len(data) > byte_cap:
        raise ValueError("cgroup process metadata exceeded its byte cap")
    return bytes(data)


def _checked_cgroup(group: str, cgroup_root: Path) -> Path:
    if not group.startswith("/") or ".." in Path(group).parts:
        raise ValueError("campaign cgroup path is invalid")
    if not group.endswith("/wnn-campaign.slice"):
        raise ValueError("systemd cgroup is not the campaign parent slice")
    root = cgroup_root / group.lstrip("/")
    if root == cgroup_root or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("campaign cgroup is absent, redirected, or resolves to host root")
    return root


def _scan_populated(root: Path) -> dict[str, str]:
    pending, visited, aggregate = [root], 0, 0
    result: dict[str, str] = {}
    while pending:
        current = pending.pop()
        visited += 1
        if visited > MAX_CGROUP_DIRECTORIES:
            raise ValueError("campaign cgroup directory cap exceeded")
        raw = _metadata_read(current / "cgroup.procs", MAX_CGROUP_PROCS_FILE_BYTES)
        aggregate += len(raw)
        if aggregate > MAX_CGROUP_PROCS_AGGREGATE_BYTES:
            raise ValueError("campaign cgroup aggregate process cap exceeded")
        if raw.strip():
            result[str(current)] = raw.decode("ascii", errors="strict").strip()
        with os.scandir(current) as entries:
            for count, entry in enumerate(entries, start=1):
                if count > MAX_CGROUP_ENTRIES_PER_DIRECTORY or entry.is_symlink():
                    raise ValueError("campaign cgroup entry cap or symlink violation")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(current / entry.name)
                    if visited + len(pending) > MAX_CGROUP_DIRECTORIES:
                        raise ValueError("campaign cgroup directory cap exceeded")
    return result


def _populated() -> dict[str, str]:
    parent = _show("wnn-campaign.slice")
    if parent.get("Id") != "wnn-campaign.slice":
        raise ValueError("campaign parent slice identity differs")
    group = parent.get("ControlGroup", "")
    return _scan_populated(_checked_cgroup(group, CGROUP_ROOT))


def _require_before(properties: dict[str, str], invocation: str) -> None:
    expected = {
        "Id": SCORE_UNIT,
        "InvocationID": invocation,
        "ActiveState": "active",
        "SubState": "exited",
        "MainPID": "0",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "Result": "success",
    }
    if any(properties.get(key) != value for key, value in expected.items()):
        raise ValueError("score unit is not the accepted successful exited invocation")


def _require_after(properties: dict[str, str]) -> None:
    expected = {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "MainPID": "0",
        "InvocationID": "",
    }
    if any(properties.get(key) != value for key, value in expected.items()):
        raise RuntimeError("exact score stop outcome requires inspection; never retry")


def main() -> None:
    """Stop the score unit once after its exact ROOT result acceptance is durable."""
    raw = base64.b64decode(ACCEPTANCE_BASE64, validate=True)
    acceptance_sha, result = _validate_receipt(
        raw, ACCEPTANCE_BYTES, ACCEPTANCE_SHA256, SCORE_INVOCATION
    )
    before = _show(SCORE_UNIT)
    _require_before(before, SCORE_INVOCATION)
    populated_before = _populated()
    if populated_before:
        raise ValueError("campaign has populated descendants before score stop")
    stop_code, stop_stdout, stop_stderr = _run_bounded(
        ["systemctl", "--user", "stop", SCORE_UNIT], 30, 4_096, 4_096
    )
    after = _show(SCORE_UNIT)
    populated_after = _populated()
    _require_after(after)
    if stop_code != 0 or populated_after:
        raise RuntimeError(
            "exact score stop failed or campaign cgroup remained populated; never retry"
        )
    print(
        json.dumps(
            {
                "schema": "fg19-first-cell-score-exact-owned-stop.v1",
                "captured_utc": datetime.now(UTC).isoformat(),
                "unit": SCORE_UNIT,
                "invocation_id": SCORE_INVOCATION,
                "acceptance_sha256": acceptance_sha,
                "result_sha256": result["sha256"],
                "before": before,
                "after": after,
                "populated_before": populated_before,
                "populated_after": populated_after,
                "stop_returncode": stop_code,
                "stop_stdout": stop_stdout,
                "stop_stderr": stop_stderr,
                "scope": "One ROOT-accepted fixed development score unit only; stop status makes no score-quality or public-transfer claim.",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
