"""Safe marker-owned local mirror directories and atomic native-file writes."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from multistage_github_common import (
    RESULTS_BRANCH,
    SNAPSHOT_SCHEMA,
    canonical_json,
    safe_run_id,
)


def atomic_write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    """Atomically replace one regular output file without following symlinks."""
    if path.is_symlink():
        raise ValueError(f"refusing to write through a symlink: {path.name}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_local_mirror_root(path: Path) -> None:
    """Create or verify a marker-owned, non-symlink directory for mirror outputs."""
    if path.exists() or path.is_symlink():
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("local mirror root must be a real directory")
        marker = path / ".multistage_github_mirror.json"
        if marker.is_symlink() or not marker.is_file():
            raise ValueError("existing local mirror root lacks its ownership marker")
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record != {
            "schema": SNAPSHOT_SCHEMA,
            "repository": "VladKochetov007/WunderFunder",
            "branch": RESULTS_BRANCH,
        }:
            raise ValueError("existing local mirror root belongs to another source")
        return
    path.mkdir(parents=True, exist_ok=False)
    marker = {
        "schema": SNAPSHOT_SCHEMA,
        "repository": "VladKochetov007/WunderFunder",
        "branch": RESULTS_BRANCH,
    }
    atomic_write(path / ".multistage_github_mirror.json", canonical_json(marker) + b"\n")


def safe_run_directory(root: Path, run_id: object) -> Path:
    """Resolve a safe direct child path and reject symlink or non-directory collisions."""
    if not safe_run_id(run_id):
        raise ValueError("unsafe run ID")
    path = root / str(run_id)
    if path.exists() or path.is_symlink():
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("run output path is not a real directory")
    else:
        path.mkdir(exist_ok=False)
    return path
