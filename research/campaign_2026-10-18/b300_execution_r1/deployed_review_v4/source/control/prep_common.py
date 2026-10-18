"""Bounded waiting and subprocess helpers shared by L4 input preparation."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from journal import Journal
from policy import sha256_file
from runtime import run_single

WAIT_SECONDS = 24 * 60 * 60
PREP_SECONDS = 12 * 60 * 60
DATASETS = ("train.parquet", "valid.parquet", "valid_mask.parquet")
DATASET_BYTES = {
    "train.parquet": 29_240_219_011,
    "valid.parquet": 5_615_509_128,
    "valid_mask.parquet": 91_972_846,
}
PARTIAL_SUFFIXES = (".partial", ".part", ".tmp", ".incomplete")


def _heartbeat_wait(journal: Journal, data_dir: Path, deadline: float) -> None:
    """Wait in finite heartbeat intervals for all final parquet files to settle."""
    previous: tuple[tuple[str, int, int], ...] | None = None
    stable_since = 0.0
    while time.monotonic() < deadline:
        partials = [path for path in data_dir.glob("*") if path.name.endswith(PARTIAL_SUFFIXES)]
        files = [data_dir / name for name in DATASETS]
        if all(
            path.is_file() and path.stat().st_size == DATASET_BYTES[path.name] for path in files
        ):
            current = tuple(
                (path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in files
            )
            if current == previous:
                if time.monotonic() - stable_since >= 30:
                    journal.record("datasets_stable", files=current)
                    return
            else:
                previous = current
                stable_since = time.monotonic()
        else:
            previous = None
            stable_since = time.monotonic()
        journal.heartbeat(
            phase="wait_for_final_datasets",
            present=[path.name for path in files if path.is_file()],
            partials=[path.name for path in partials],
        )
        time.sleep(30)
    raise TimeoutError("final train/valid/valid_mask parquets did not settle within 24 hours")


def wait_for_datasets(root: Path, journal: Journal) -> None:
    """Wait for non-partial, stable dataset files without opening or hashing them."""
    data_dir = root / "data" / "datasets"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"expected staged datasets directory: {data_dir}")
    _heartbeat_wait(journal, data_dir, time.monotonic() + WAIT_SECONDS)


def _copy_pinned_script(source: Path, destination: Path) -> str:
    source_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != source_hash:
            raise RuntimeError(
                f"staged preparation script differs from frozen source: {destination}"
            )
    else:
        shutil.copyfile(source, destination)
        if sha256_file(destination) != source_hash:
            raise RuntimeError(f"copied preparation script hash changed: {destination}")
    return source_hash


def _run_checked(
    *,
    name: str,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    cpus: tuple[int, ...],
    log: Path,
    journal: Journal,
    root: Path,
    deadline: float,
) -> dict[str, Any]:
    environment = {**os.environ, **env, "WNN_CWD": str(cwd)}
    result = run_single(
        name=name,
        argv=argv,
        environment=environment,
        cpus=cpus,
        log_path=log,
        timeout_seconds=PREP_SECONDS,
        journal=journal,
        root=root,
        global_deadline=deadline,
    )
    if result["returncode"] != 0 or result["status"] != "completed":
        raise RuntimeError(f"preparation step {name} failed; inspect {log}: {result}")
    return result


def _atomic_copy(source: Path, destination: Path) -> None:
    """Create a new split manifest by same-filesystem temporary write and fsync."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != source.read_bytes():
            raise FileExistsError(
                f"existing split differs from regenerated membership: {destination}"
            )
        return
    temporary = destination.with_name(f".{destination.name}.controller-tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with source.open("rb") as reader, temporary.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
