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


def require(condition: bool, message: str) -> None:
    """Fail closed before the first support-output mutation."""
    if not condition:
        raise RuntimeError(message)


def digest(raw: bytes) -> str:
    """Return a SHA-256 digest for exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def json_object(raw: bytes, name: str) -> dict[str, object]:
    """Parse a bounded JSON object while retaining its original bytes."""
    value = json.loads(raw)
    require(isinstance(value, dict), f"JSON object expected: {name}")
    return value


def stable_file(path: Path, cap: int) -> tuple[bytes, dict[str, int]]:
    """Read one bounded regular file through a no-follow descriptor and verify stability."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(), f"unsafe file: {path}")
    require(before.st_size <= cap, f"file exceeds cap: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        opened = os.fstat(descriptor)
        require(
            (opened.st_dev, opened.st_ino, opened.st_size)
            == (before.st_dev, before.st_ino, before.st_size),
            f"file changed before read: {path}",
        )
        raw = bytearray()
        while len(raw) <= cap:
            block = os.read(descriptor, min(65_536, cap + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(len(raw) == before.st_size and len(raw) <= cap, f"file changed or exceeded cap: {path}")
    require(
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        f"file changed while read: {path}",
    )
    return bytes(raw), {
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def file_hash(path: Path, cap: int) -> tuple[str, dict[str, int]]:
    """Stream a source file hash while checking its stable inode and timestamp."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(), f"unsafe source file: {path}")
    require(before.st_size <= cap, f"source file exceeds cap: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    hasher = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        require(
            (opened.st_dev, opened.st_ino, opened.st_size)
            == (before.st_dev, before.st_ino, before.st_size),
            f"source changed before hash: {path}",
        )
        while block := os.read(descriptor, 1 << 20):
            hasher.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        f"source changed while hashing: {path}",
    )
    return hasher.hexdigest(), {
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


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


def _memory_bytes(value: str) -> int:
    """Parse nvidia-smi's integer MiB field without accepting unknown values."""
    fields = value.strip().split()
    require(
        len(fields) == 2 and fields[0].isdigit() and fields[1] == "MiB",
        "selected-GPU memory field is malformed",
    )
    return int(fields[0]) * 1024 * 1024


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


def show(unit: str, fields: tuple[str, ...], timeout: int = 15) -> dict[str, str]:
    """Read named systemd properties for one exact unit."""
    command = ["systemctl", "--user", "show", "--all", unit]
    for field in fields:
        command.extend(("-p", field))
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    require(result.returncode == 0, f"systemctl show failed for {unit}")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def write_exclusive(path: Path, raw: bytes) -> None:
    """Create and fsync one private file without replacing existing output."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    """Create one compact JSON receipt with exclusive creation."""
    write_exclusive(
        path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    )
