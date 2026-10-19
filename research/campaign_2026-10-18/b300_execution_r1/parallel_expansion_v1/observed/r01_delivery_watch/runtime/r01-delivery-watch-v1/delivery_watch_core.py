"""Bounded receipts, manifest, and process helpers for the R01 delivery watcher."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

PANEL_SHA256 = "471f465cc1d13fc6e5153a4c998390bb14e06823fed4c6721f929ce95fee47fb"
HELPER_MANIFEST_SHA256 = "d5ac193fc3241b7e54d985e71077167f1170fb0c2300967631226937ab22ae8f"
HELPER_SCHEMA = "tabdata-r01-delivery-recovery-manifest.v2.4"
EXPECTED_DEADLINE_UNIX = 1792648272
MEMORY_LIMIT_BYTES = 16 * 1024**3
POLL_SECONDS = 10
VALIDATE_TIMEOUT_SECONDS = 1800
EXPORT_TIMEOUT_SECONDS = 600
GRADE_TIMEOUT_SECONDS = 3600
PUBLISH_INTERVAL_SECONDS = 600


def sha256_file(path: Path) -> str:
    """Hash one regular nonsymlink file in bounded memory."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular nonsymlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read a nonsymlink UTF-8 JSON object."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular nonsymlink JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace watcher-owned status without touching fit artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".delivery-watch-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Append one durable event record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_helper_bundle(helper_dir: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify the exact v2.3 manifest and each listed helper file."""
    manifest_path = helper_dir / "MANIFEST.json"
    if sha256_file(manifest_path) != expected_sha256 or expected_sha256 != HELPER_MANIFEST_SHA256:
        raise ValueError("R01 recovery helper manifest differs from the configured v2.3 pin")
    manifest = read_object(manifest_path)
    files = manifest.get("files")
    if manifest.get("schema") != HELPER_SCHEMA or not isinstance(files, dict):
        raise ValueError("R01 recovery helper manifest schema/files are malformed")
    for relative, expected in files.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or not isinstance(expected, str):
            raise ValueError(f"unsafe helper manifest entry: {relative!r}")
        path = (helper_dir / rel).resolve(strict=True)
        if (
            not path.is_relative_to(helper_dir.resolve(strict=True))
            or sha256_file(path) != expected
        ):
            raise ValueError(f"R01 recovery helper file differs from manifest: {relative}")
    return manifest


def verify_watcher_files(script_dir: Path, files: dict[str, str]) -> None:
    """Check watcher files pinned by config; defer only the optional publisher."""
    required = {
        "r01_delivery_watch.py",
        "delivery_watch_core.py",
        "delivery_watch_state.py",
        "delivery_watch_phases.py",
        "delivery_watch_grades.py",
        "delivery_watch_receipts.py",
        "publish_recovery.py",
    }
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("watcher config must pin every executable watcher and publisher module")
    for relative, expected in files.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or not isinstance(expected, str):
            raise ValueError(f"unsafe watcher file entry: {relative!r}")
        path = script_dir / rel
        if not path.exists() and relative == "publish_recovery.py":
            continue
        if sha256_file(path) != expected:
            raise ValueError(f"watcher file differs from the configured hash: {relative}")


def verify_runtime_envelope(gpu_uuid: str) -> None:
    """Require the dedicated watcher unit's exact device, CPU, and RAM envelope."""
    if os.name != "posix" or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("R01 delivery watcher requires Linux CPU-affinity support")
    if set(os.sched_getaffinity(0)) != {172, 174}:
        raise RuntimeError("R01 delivery watcher must be pinned to CPUs 172 and 174")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid or "," in gpu_uuid:
        raise RuntimeError("R01 exports require the single configured GPU1 UUID")
    cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    v2 = next((line.split("::", 1)[1] for line in cgroup if line.startswith("0::")), None)
    if v2 is None:
        raise RuntimeError("R01 delivery watcher requires a cgroup v2 memory limit")
    cgroup_path = (Path("/sys/fs/cgroup") / v2.lstrip("/")).resolve(strict=True)
    if not cgroup_path.is_relative_to(Path("/sys/fs/cgroup").resolve(strict=True)):
        raise RuntimeError("watcher cgroup path escapes the cgroup v2 mount")
    memory_max = (cgroup_path / "memory.max").read_text(encoding="ascii").strip()
    if memory_max != str(MEMORY_LIMIT_BYTES):
        raise RuntimeError("R01 delivery watcher cgroup must have an exact 16 GiB memory cap")


def actor_still_running(resource: dict[str, Any]) -> bool:
    """Detect the exact frozen train actor after its finalizer wrote a receipt."""
    pid = resource.get("pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError("train actor resource receipt has no valid PID")
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        return False
    stat = (proc / "stat").read_text(encoding="ascii")
    state = stat.rsplit(")", 1)[1].split()[0]
    if state == "Z":
        return False
    cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
    return b"gpu_entrypoint.py" in cmdline and b"train_cell.py" in cmdline


def remaining(deadline_unix: int) -> int:
    """Return positive integer seconds until the frozen delivery deadline."""
    left = deadline_unix - time.time()
    if left <= 0:
        raise TimeoutError("the frozen R01 delivery deadline has expired")
    return max(1, int(left))


def fresh_attempt_allowed(
    verified_receipt_exists: bool, attempt_path_exists: bool, prior_status: str
) -> bool:
    """Admit a phase only when it has no receipt, path, or durable prior claim."""
    return (
        verified_receipt_exists is False
        and attempt_path_exists is False
        and prior_status == "pending"
    )
