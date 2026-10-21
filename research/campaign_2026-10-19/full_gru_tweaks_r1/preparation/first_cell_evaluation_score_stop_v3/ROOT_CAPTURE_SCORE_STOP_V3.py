"""Capture one accepted score-unit exact stop; preserve raw bytes before parsing."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY = Path(
    "/Users/nb/Documents/wunder/WunderFunder-maintenance/research/campaign_2026-10-19/full_gru_tweaks_r1"
)
D = REPOSITORY / "records/first_cell_evaluation_v1/score_v3/exact_stop"
BODY_PIN = (0, "UNBOUND_ROOT_SCORE_BODY_SHA256")
REVIEW_PIN = (0, "UNBOUND_ROOT_SCORE_BODY_REVIEW_SHA256")
ACCEPTANCE_PIN = (0, "UNBOUND_ROOT_SCORE_ACCEPTANCE_SHA256")
SCORE_UNIT = "computer-vision-homework-fg19-first-cell-score-v2.service"
SCORE_INVOCATION = "UNBOUND_ROOT_SCORE_INVOCATION"
ACCEPTANCE_PATH = D.parent / "SCORE_ACCEPTANCE.json"
ACCEPTANCE_SCHEMA = "fg19.first-cell-score-root-acceptance.v1"
ACCEPTANCE_STATUS = "ACCEPTED_FIXED_FIRST_CELL_DEVELOPMENT_SCORE_REPORT"
RESULT_SCHEMA = "fg19.first-cell-five-arm-streaming-score.v1"
RESULT_STATUS = "FIXED_FIRST_CELL_DEVELOPMENT_EVIDENCE_SCORED"
MAX_ACCEPTANCE_BYTES = 1_000_000
MAX_STOP_STDOUT_BYTES = 128_000
EXPECTED_STOP_KEYS = frozenset(
    {
        "schema",
        "captured_utc",
        "unit",
        "invocation_id",
        "acceptance_sha256",
        "result_sha256",
        "before",
        "after",
        "populated_before",
        "populated_after",
        "stop_returncode",
        "stop_stdout",
        "stop_stderr",
        "scope",
    }
)
EXPECTED_SCOPE = (
    "One ROOT-accepted fixed development score unit only; stop status makes no score-quality "
    "or public-transfer claim."
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_hex(value: object, size: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == size
        and all(char in "0123456789abcdef" for char in value)
    )


def _read_pin(path: Path, expected: tuple[int, str], *, byte_cap: int) -> bytes:
    if (
        type(expected[0]) is not int
        or expected[0] <= 0
        or expected[0] > byte_cap
        or not _is_hex(expected[1], 64)
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != expected[0]
    ):
        raise ValueError(f"wrong regular pinned metadata input: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(descriptor)
        before = path.lstat()
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"pinned metadata changed before read: {path}")
        raw = bytearray()
        while len(raw) <= byte_cap:
            chunk = os.read(descriptor, min(4096, byte_cap + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
    finally:
        os.close(descriptor)
    if (len(raw), digest(raw)) != expected:
        raise ValueError(f"pinned metadata bytes/SHA-256 differ: {path}")
    return bytes(raw)


def _bound_identity() -> None:
    for label, value in (
        ("body", BODY_PIN),
        ("review", REVIEW_PIN),
        ("acceptance", ACCEPTANCE_PIN),
    ):
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or type(value[0]) is not int
            or value[0] <= 0
            or not _is_hex(value[1], 64)
        ):
            raise RuntimeError(f"actual ROOT {label} byte/SHA pin is still unbound")
    if not _is_hex(SCORE_INVOCATION, 32):
        raise RuntimeError("actual ROOT score invocation is still unbound")
    if SCORE_UNIT != "computer-vision-homework-fg19-first-cell-score-v2.service":
        raise RuntimeError("score unit differs from the reviewed score-v2 scope")


def _validate_acceptance(raw: bytes) -> tuple[str, dict[str, object]]:
    acceptance_sha = digest(raw)
    if (len(raw), acceptance_sha) != ACCEPTANCE_PIN:
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
            "invocation_id": SCORE_INVOCATION,
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
        or not isinstance(result.get("path"), str)
        or not result["path"]
    ):
        raise ValueError("ROOT acceptance lacks the exact successful score-result pin")
    return acceptance_sha, result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(name: str, raw: bytes) -> None:
    path = D / name
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(D)


def _validate_stop(raw: bytes, acceptance_sha: str, result_sha: str) -> None:
    if len(raw) > MAX_STOP_STDOUT_BYTES:
        raise ValueError("score stop response exceeds its byte bound; raw bytes preserved")
    stopped = json.loads(raw)
    if not isinstance(stopped, dict) or set(stopped) != EXPECTED_STOP_KEYS:
        raise ValueError("score stop response does not have the exact 14-key schema")
    try:
        captured = datetime.fromisoformat(stopped["captured_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("score stop capture timestamp is malformed") from error
    if captured.tzinfo is None or captured.utcoffset() is None:
        raise ValueError("score stop capture timestamp lacks a timezone")
    if (
        stopped.get("schema") != "fg19-first-cell-score-exact-owned-stop.v1"
        or stopped.get("unit") != SCORE_UNIT
        or stopped.get("invocation_id") != SCORE_INVOCATION
        or stopped.get("acceptance_sha256") != acceptance_sha
        or stopped.get("result_sha256") != result_sha
    ):
        raise ValueError("score stop receipt identity/result join differs")
    before = stopped.get("before")
    after = stopped.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("score stop before/after facts must be objects")
    before_values = {
        "Id": SCORE_UNIT,
        "InvocationID": SCORE_INVOCATION,
        "ActiveState": "active",
        "SubState": "exited",
        "MainPID": "0",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "Result": "success",
    }
    if any(before.get(key) != value for key, value in before_values.items()):
        raise ValueError("score stop before-state is not the exact successful 0/1/0 exit")
    after_values = {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "MainPID": "0",
        "InvocationID": "",
    }
    if any(after.get(key) != value for key, value in after_values.items()):
        raise ValueError("score stop after-state is not the exact not-found/inactive/dead state")
    if (
        type(stopped.get("stop_returncode")) is not int
        or stopped["stop_returncode"] != 0
        or stopped.get("populated_before") != {}
        or stopped.get("populated_after") != {}
        or stopped.get("stop_stdout") != ""
        or stopped.get("stop_stderr") != ""
        or stopped.get("scope") != EXPECTED_SCOPE
    ):
        raise ValueError("score stop return code, process snapshots, streams, or scope differ")


def main() -> None:
    """Submit exactly one accepted score stop and preserve raw bytes before parse."""
    _bound_identity()
    if D.is_symlink() or not D.is_dir() or D.resolve(strict=True) != D:
        raise ValueError("exact private score-stop directory is absent or redirected")
    if ACCEPTANCE_PATH.is_symlink() or not ACCEPTANCE_PATH.is_file():
        raise ValueError("actual ROOT score acceptance is absent or redirected")
    for name in ("STDOUT.raw.json", "STDERR.raw.txt", "TRANSPORT.json", "EXACT_STOP.json"):
        if (D / name).exists() or (D / name).is_symlink():
            raise ValueError("prior score-stop outcome exists; inspect it and never retry")

    body = _read_pin(D / "REMOTE.py", BODY_PIN, byte_cap=1_000_000)
    _read_pin(D / "SOURCE_REVIEW.json", REVIEW_PIN, byte_cap=128_000)
    acceptance_raw = _read_pin(ACCEPTANCE_PATH, ACCEPTANCE_PIN, byte_cap=MAX_ACCEPTANCE_BYTES)
    acceptance_sha, result = _validate_acceptance(acceptance_raw)

    started = datetime.now(UTC).isoformat()
    timed_out = False
    try:
        command = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            "beleriand",
            "python3 -E -B -",
        ]
        completed = subprocess.run(
            command,
            input=body,
            capture_output=True,
            timeout=60,
            check=False,
        )
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        returncode = None
        timed_out = True
    finished = datetime.now(UTC).isoformat()

    _write_exclusive("STDOUT.raw.json", stdout)
    _write_exclusive("STDERR.raw.txt", stderr)
    transport = {
        "schema": "fg19.first-cell-score-exact-stop-root-transport.v1",
        "started_utc": started,
        "finished_utc": finished,
        "body_bytes": len(body),
        "body_sha256": digest(body),
        "source_review_bytes": REVIEW_PIN[0],
        "source_review_sha256": REVIEW_PIN[1],
        "acceptance_bytes": ACCEPTANCE_PIN[0],
        "acceptance_sha256": acceptance_sha,
        "ssh_returncode": returncode,
        "timed_out": timed_out,
        "stdout_bytes": len(stdout),
        "stdout_sha256": digest(stdout),
        "stderr_bytes": len(stderr),
        "stderr_sha256": digest(stderr),
        "persisted_before_parse": True,
        "retry_performed": False,
    }
    _write_exclusive("TRANSPORT.json", (json.dumps(transport, indent=2) + "\n").encode())

    if returncode != 0 or timed_out or stderr or len(stdout) > MAX_STOP_STDOUT_BYTES:
        raise RuntimeError("score stop raw outcome is durable; inspect and never retry")
    _validate_stop(stdout, acceptance_sha, result["sha256"])
    _write_exclusive("EXACT_STOP.json", stdout)
    print(
        json.dumps(
            {
                "status": "EXACT_OWNED_SUCCESSFUL_SCORE_STOP_CAPTURED",
                "bytes": len(stdout),
                "sha256": digest(stdout),
                "transport": transport,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
