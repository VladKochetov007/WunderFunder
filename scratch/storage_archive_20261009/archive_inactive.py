from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path


def identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError(f"Expected one regular file: {path}")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return value.hexdigest()


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def assert_unused(path: Path) -> None:
    target = str(path)
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            for descriptor in (process / "fd").iterdir():
                try:
                    if os.readlink(descriptor) == target:
                        raise RuntimeError(f"Open in process {process.name}: {path}")
                except FileNotFoundError:
                    continue
            if target in (process / "maps").read_text():
                raise RuntimeError(f"Mapped in process {process.name}: {path}")
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue


def record(journal: Path, event: dict[str, object]) -> None:
    event["utc"] = datetime.now(UTC).isoformat()
    with journal.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(event, sort_keys=True), flush=True)


def move_one(source: Path, destination: Path, journal: Path, expected: list[int]) -> None:
    archive = journal.parent.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.parent.resolve().is_relative_to(archive):
        raise ValueError("Destination parent leaves archive")
    if destination.is_symlink():
        raise ValueError(f"Destination is a symlink: {destination}")
    if destination.exists():
        identity(destination)
    records = (
        [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    )
    previous = next((row for row in reversed(records) if row["source"] == str(source)), None)
    if source.is_symlink() or not source.exists():
        if source.is_symlink() and source.resolve(strict=True) != destination:
            raise ValueError(f"Unexpected symlink: {source}")
        if previous is None or digest(destination) != previous["sha256"]:
            raise ValueError(f"Archived checksum changed: {destination}")
        finish_handoff(source, destination, journal, previous, expected)
        return
    original = identity(source)
    if original != tuple(expected):
        raise ValueError(f"Source changed since admission: {source}")
    assert_unused(source)
    if source.stat().st_dev == destination.parent.stat().st_dev:
        raise ValueError("Archive must be on a separate filesystem")
    required = 20_000_000_000 + (0 if destination.exists() else original[2])
    if shutil.disk_usage(destination.parent).free < required:
        raise RuntimeError("Archive reserve below 20 GB")
    temporary = destination.with_name(destination.name + ".archive-partial")
    copied_hash = hashlib.sha256()
    if not destination.exists():
        if temporary.exists() or temporary.is_symlink():
            identity(temporary)
            temporary.unlink()
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while block := reader.read(8 * 1024 * 1024):
                writer.write(block)
                copied_hash.update(block)
            writer.flush()
            os.fsync(writer.fileno())
            os.posix_fadvise(reader.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        shutil.copystat(source, temporary)
        if temporary.stat().st_size != original[2] or digest(temporary) != copied_hash.hexdigest():
            raise ValueError(f"Copy verification failed: {source}")
        os.replace(temporary, destination)
        sync_directory(destination.parent)
        checksum = copied_hash.hexdigest()
    else:
        checksum = digest(source)
        if digest(destination) != checksum:
            raise ValueError(f"Existing destination differs: {destination}")
    if identity(source) != original:
        raise ValueError(f"Source changed during copy: {source}")
    assert_unused(source)
    receipt = {
        "source": str(source),
        "destination": str(destination),
        "bytes": original[2],
        "sha256": checksum,
        "status": "verified",
        "handoff": str(
            Path(tempfile.mkdtemp(prefix=".archive-handoff-", dir=source.parent)) / source.name
        ),
    }
    record(journal, receipt)
    # Rename into a private directory before creating the link exclusively.
    # A concurrent replacement is preserved and detected, never overwritten.
    source.rename(receipt["handoff"])
    sync_directory(source.parent)
    finish_handoff(source, destination, journal, receipt, expected)


def finish_handoff(
    source: Path, destination: Path, journal: Path, receipt: dict[str, object], expected: list[int]
) -> None:
    saved = Path(str(receipt["handoff"]))
    if saved.exists() or saved.is_symlink():
        if identity(saved) != tuple(expected):
            raise ValueError(f"Changed source preserved at {saved}")
        assert_unused(saved)
    elif not source.is_symlink():
        raise ValueError(f"Interrupted source unavailable: {saved}")
    if not source.is_symlink():
        source.symlink_to(destination)
    if source.resolve(strict=True) != destination:
        raise ValueError(f"Concurrent source change preserved; original at {saved}")
    sync_directory(source.parent)
    if saved.exists():
        if identity(saved) != tuple(expected):
            raise ValueError(f"Changed source preserved at {saved}")
        assert_unused(saved)
        saved.unlink()
        saved.parent.rmdir()
    record(journal, {**receipt, "status": "moved"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan["repository"]).resolve()
    archive = Path(plan["archive"]).resolve()
    archive.mkdir(parents=True, exist_ok=True, mode=0o700)
    journal = archive / "moves.jsonl"
    shutil.copy2(args.plan, archive / "plan.json")
    for entry in plan["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative path: {relative}")
        with Path("/tmp/wnn-codex-support.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            move_one(root / relative, archive / relative, journal, entry["identity"])
        # Let queued cloud checks acquire the shared support slot between files.
        time.sleep(1)


if __name__ == "__main__":
    main()
