from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class LaunchError(RuntimeError):
    """Raised when the frozen eight-fit launch contract is not satisfied."""


def sha256_file(path: Path) -> str:
    """Hash one file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a non-symlink regular file."""
    if path.is_symlink() or not path.is_file():
        raise LaunchError(f"expected a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LaunchError(f"expected a JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: object) -> None:
    """Durably replace a JSON receipt without following an existing symlink."""
    if path.is_symlink():
        raise LaunchError(f"receipt path cannot be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    """Flush directory metadata on POSIX after durable receipt replacement."""
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def parse_cpu_list(value: str) -> tuple[int, ...]:
    """Parse Linux cpulist syntax such as ``0-3,8,10-11`` without duplicates."""
    cpus: list[int] = []
    for part in value.split(","):
        bounds = part.split("-")
        if len(bounds) not in (1, 2) or not all(bound.isdecimal() for bound in bounds):
            raise LaunchError(f"invalid CPU list: {value!r}")
        start = int(bounds[0])
        end = int(bounds[-1])
        if end < start:
            raise LaunchError(f"invalid CPU range: {part!r}")
        cpus.extend(range(start, end + 1))
    if not cpus or len(set(cpus)) != len(cpus):
        raise LaunchError(f"CPU list is empty or contains duplicates: {value!r}")
    return tuple(sorted(cpus))


def process_identity(pid: int) -> dict[str, int | str] | None:
    """Read a Linux process's start ticks and process group without signalling it."""
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
            "process_group": int(fields[2]),
            "start_ticks": int(fields[19]),
        }
    except (IndexError, ValueError):
        return None


def owned_process_state(record: dict[str, Any]) -> str:
    """Classify only a PID whose start ticks and group match its launch receipt."""
    pid = record.get("pid")
    if type(pid) is not int or pid < 2:
        return "invalid_receipt"
    current = process_identity(pid)
    if current is None:
        return "exited"
    if (
        current["start_ticks"] != record.get("start_ticks")
        or current["process_group"] != record.get("process_group")
    ):
        return "pid_reused_or_unowned"
    if current["state"] in ("Z", "X"):
        return "exited"
    return "running"


def utc_now() -> str:
    """Return one UTC timestamp suitable for durable JSON receipts."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_jsonl_tail(path: Path, max_bytes: int = 1 << 20) -> dict[str, Any] | None:
    """Read the last complete JSON object from a bounded tail of a JSONL file."""
    if path.is_symlink() or not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        lines = stream.read(max_bytes).splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None
