from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import signal
import time
import select
import tarfile
from pathlib import Path, PurePosixPath

BUNDLE_SLUG = "e482-t4-sixfit-20261005"
EXPECTED_ARCHIVE_SHA256 = "c9d2b88ef47540cb9ecb13f659552ba5fe1efb351999e76882387d65a2dcc061"
EXPECTED_EXTERNAL_MOUNTS = {
    "datasets/valid.parquet": {
        "dataset": "vladkochetov07/wunder-funder",
        "filename": "valid.parquet",
    }
}
EXPECTED_MANIFEST_SHA256 = "db8c0dd1df5b2d9fab1a99c631aabe5af3aa8eb7798a35980570598784b1fb9a"


def release_mounts(root: Path, mounts: dict) -> None:
    """Unlink only external input symlinks before Kaggle captures working outputs."""
    for logical in mounts:
        target = root / logical
        if target.is_symlink():
            target.unlink()


def validate_external_mounts(
    root: Path,
    mounts: dict,
    archive_members: set[str],
    expected_mounts: dict,
) -> None:
    """Reject undeclared or unsafe mount targets before creating any symlink."""
    if not isinstance(mounts, dict) or mounts != expected_mounts:
        raise ValueError("External mappings differ from the controller-bound kernel")
    if set(mounts) != set(expected_mounts):
        raise ValueError("External mapping keys differ from the controller-bound kernel")
    for logical in mounts:
        if not isinstance(logical, str) or not logical:
            raise ValueError("External input path is not a nonempty string")
        path = PurePosixPath(logical)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in logical
            or path.as_posix() != logical
            or logical not in expected_mounts
        ):
            raise ValueError("Unsafe external input mapping")
        if logical in archive_members:
            raise ValueError("External input mapping overlaps an archive member")
        target = root / path
        if target.exists() or target.is_symlink():
            raise ValueError("External input target already exists")


def find_external_candidate(input_root: Path, dataset: str, filename: str) -> Path:
    """Resolve a single regular file inside the authorized mounted dataset root."""
    parts = dataset.split("/") if isinstance(dataset, str) else []
    if (
        len(parts) != 2
        or not all(parts)
        or not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise ValueError("Invalid external dataset file descriptor")
    slug = parts[1]
    candidates = list(input_root.glob(f"**/{slug}/{filename}"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one authorized input: {dataset}/{filename}")
    input_root_real = input_root.resolve(strict=True)
    candidate = candidates[0]
    dataset_roots = [parent for parent in candidate.parents if parent.name == slug]
    if not dataset_roots:
        raise ValueError("Mounted file has no authorized dataset root")
    dataset_root = dataset_roots[0].resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if (
        not dataset_root.is_relative_to(input_root_real)
        or not resolved.is_file()
        or not resolved.is_relative_to(dataset_root)
    ):
        raise ValueError("Mounted input resolves outside the authorized dataset root")
    return resolved


def validate_archive_members(members: list, allowed: set[str], external: set[str]) -> None:
    """Reject unsafe, linked, duplicate, undeclared and omitted archive files."""
    expected = allowed - external
    seen = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in member.name
            or path.as_posix() != member.name
        ):
            raise ValueError("Archive path escapes experiment directory")
        if not member.isfile() or member.name not in expected or member.name in seen:
            raise ValueError("Unsupported, undeclared or duplicate archive member")
        seen.add(member.name)
    if seen != expected:
        raise ValueError("Archive member set is incomplete")


def _kernel_work() -> None:
    """Extract a verified private bundle and mount hashed raw data without copies."""
    matches = list(Path("/kaggle/input").glob(f"**/{BUNDLE_SLUG}/bundle-manifest.json"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one named private bundle")
    manifest_bytes = matches[0].read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Private bundle manifest identity mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("private") is not True:
        raise ValueError("Bundle is not marked private")
    if manifest.get("external_mounts") != EXPECTED_EXTERNAL_MOUNTS:
        raise ValueError("External mappings differ from the controller-bound kernel")
    archive = matches[0].parent / "bundle.wnn"
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    if (
        digest.hexdigest() != manifest["archive_sha256"]
        or digest.hexdigest() != EXPECTED_ARCHIVE_SHA256
    ):
        raise ValueError("Private bundle archive identity mismatch")
    root = Path("/kaggle/working/experiment")
    if shutil.disk_usage(root.parent).free < manifest["expanded_bytes"] + 2 * 1024**3:
        raise ValueError("Insufficient working disk to extract bundle with reserve")
    root.mkdir(exist_ok=False)
    allowed = {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"])
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        archive_entries = tar.getmembers()
        validate_archive_members(archive_entries, allowed, set(manifest["external_mounts"]))
        for member in archive_entries:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("Archive path escapes experiment directory")
            if not member.isfile() or member.name not in allowed or member.name in seen:
                raise ValueError("Unsupported, undeclared or duplicate archive member")
            seen.add(member.name)
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("Archive member has no file content")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, 8 * 1024**2)
    if seen != allowed - set(manifest["external_mounts"]):
        raise ValueError("Extracted archive member inventory is incomplete")
    archive_members = seen
    mounts = manifest["external_mounts"]
    validate_external_mounts(root, mounts, archive_members, EXPECTED_EXTERNAL_MOUNTS)
    resolved_mounts = {}
    for logical, mapping in mounts.items():
        dataset = (
            mapping["dataset"] if isinstance(mapping, dict) else "vladkochetov07/wunder-funder"
        )
        filename = mapping["filename"] if isinstance(mapping, dict) else mapping
        resolved_mounts[logical] = find_external_candidate(Path("/kaggle/input"), dataset, filename)
    for logical, source in resolved_mounts.items():
        target = root / PurePosixPath(logical)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise ValueError("External input target already exists")
        target.symlink_to(source)
    command = [
        sys.executable,
        str(root / "research/campaign_2026-10-05/activation_t4_replication/e482_remote.py"),
        "--output",
        "/kaggle/working/e482-results",
    ]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        subprocess.run(command, check=True)
    finally:
        release_mounts(root, manifest["external_mounts"])


def _proc_identity(pid: int) -> tuple[str, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        return None
    return fields[0], int(fields[19])


def _children(pid: int) -> list[int]:
    try:
        raw = Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []
    return [int(value) for value in raw.split()] if raw else []


def _descendants(root_pid: int, monitor_pid: int) -> list[int]:
    ordered: list[int] = []
    seen = {root_pid, monitor_pid}

    def visit(pid: int) -> None:
        for child in _children(pid):
            if child in seen:
                continue
            seen.add(child)
            visit(child)
            ordered.append(child)

    visit(root_pid)
    return ordered


def _write_timeout(path: Path, record: dict[str, object]) -> None:
    payload = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(payload)
    os.replace(temp, path)


def _watch(
    root_pid: int,
    root_start_ticks: int,
    timeout_seconds: int,
    report: str,
    ready_fd: int,
) -> None:
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    os.write(ready_fd, b"R")
    os.close(ready_fd)
    monitor_pid = os.getpid()
    started = time.monotonic()
    deadline = started + timeout_seconds
    while not stopping:
        identity = _proc_identity(root_pid)
        if identity is None or identity[1] != root_start_ticks or identity[0] == "Z":
            return
        now = time.monotonic()
        if now >= deadline:
            break
        time.sleep(min(1.0, deadline - now))
    if stopping:
        return

    victims = _descendants(root_pid, monitor_pid)
    captured = []
    for pid in victims:
        identity = _proc_identity(pid)
        if identity is not None:
            captured.append({"pid": pid, "start_ticks": identity[1]})
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    grace_deadline = time.monotonic() + 10.0
    while time.monotonic() < grace_deadline:
        if not any(
            identity is not None and identity[0] != "Z"
            for item in captured
            if (identity := _proc_identity(int(item["pid"]))) is not None
        ):
            break
        time.sleep(0.2)

    killed = []
    for item in captured:
        pid = int(item["pid"])
        identity = _proc_identity(pid)
        if identity is not None and identity[0] != "Z" and identity[1] == item["start_ticks"]:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                pass

    record = {
        "status": "whole_job_deadline_exceeded",
        "root_pid": root_pid,
        "root_start_ticks": root_start_ticks,
        "timeout_seconds": timeout_seconds,
        "watchdog_elapsed_seconds": time.monotonic() - started,
        "term_sent_pids": [item["pid"] for item in captured],
        "kill_sent_pids": killed,
        "kill_order": "descendants first; main kernel process after timeout record is flushed",
    }
    try:
        _write_timeout(Path(report), record)
    finally:
        identity = _proc_identity(root_pid)
        if identity is not None and identity[1] == root_start_ticks:
            try:
                os.kill(root_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def start_watchdog(
    timeout_seconds: int = 32_400,
    report_path: str = "/kaggle/working/whole_job_watchdog.json",
) -> int:
    """Fork a monitor before any worker setup and return its PID to the parent."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    root_pid = os.getpid()
    identity = _proc_identity(root_pid)
    if identity is None:
        raise RuntimeError("Cannot bind watchdog to the current process identity")
    ready_read_fd, ready_write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read_fd)
        try:
            _watch(root_pid, identity[1], timeout_seconds, report_path, ready_write_fd)
        except BaseException as exc:
            record = {
                "status": "watchdog_internal_error_fail_closed",
                "root_pid": root_pid,
                "root_start_ticks": identity[1],
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
            try:
                _write_timeout(Path(report_path), record)
            except BaseException:
                pass
            current = _proc_identity(root_pid)
            if current is not None and current[1] == identity[1]:
                try:
                    os.kill(root_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            os._exit(0)
    os.close(ready_write_fd)
    ready, _, _ = select.select([ready_read_fd], [], [], 5.0)
    if not ready or os.read(ready_read_fd, 1) != b"R":
        os.close(ready_read_fd)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        raise RuntimeError("Whole-job watchdog failed to start; refusing kernel work")
    os.close(ready_read_fd)
    return pid


def stop_watchdog(pid: int) -> None:
    """Stop and reap the monitor after the kernel's ordinary work finishes."""
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return
    if waited == pid:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    while True:
        try:
            os.waitpid(pid, 0)
            return
        except InterruptedError:
            continue


def main() -> None:
    watchdog_pid = start_watchdog(
        timeout_seconds=3_500,
        report_path="/kaggle/working/whole_job_watchdog.json",
    )
    try:
        _kernel_work()
    finally:
        stop_watchdog(watchdog_pid)


if __name__ == "__main__":
    main()
