from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from wnn.splits import file_sha256
from wnn.training.environment import TrainingEnvironment


def _source_path(root: Path, path: Path) -> Path:
    """Accept only regular, non-symlink source files inside the canonical root."""
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source path is outside source root: {candidate}") from error
    current = root
    for part in relative.parts:
        if part in (".", ".."):
            raise ValueError("source path must not traverse directories")
        current = current / part
        if current.is_symlink():
            raise ValueError(f"source symlink is not admitted: {current}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"source is not a regular file under the root: {candidate}")
    return resolved


def _package_sources(root: Path) -> list[Path]:
    """Collect every WNN Python source without following symbolic links."""
    package = root / "src" / "wnn"
    if package.is_symlink() or not package.is_dir() or not package.resolve().is_relative_to(root):
        raise ValueError("source package must be a real directory within source root")
    found: list[Path] = []

    def _raise_walk_error(error: OSError) -> None:
        raise error

    for parent, directories, files in os.walk(
        package, followlinks=False, onerror=_raise_walk_error
    ):
        base = Path(parent)
        if any((base / name).is_symlink() for name in directories):
            raise ValueError(f"source package contains a symbolic-link directory: {base}")
        for name in files:
            if name.endswith(".py"):
                found.append(_source_path(root, base / name))
    if not found:
        raise ValueError("source package contains no Python files")
    return sorted(found)


def training_source_identity(environment: TrainingEnvironment) -> dict[str, object]:
    """Bind the entrypoint, all WNN Python sources and available dependency files.

    Source-only packets may omit dependency files; missing files remain explicit.
    The expanded v2 identity invalidates exact resumes against v1 source digests.
    """
    root = environment.source_root.resolve(strict=True)
    entrypoint = _source_path(root, environment.entrypoint)
    dependencies = [root / "pyproject.toml", root / "uv.lock"]
    if any(path.is_symlink() for path in dependencies):
        raise ValueError("dependency source symlinks are not admitted")
    missing_dependencies = [path.name for path in dependencies if not path.exists()]
    present_dependencies = [_source_path(root, path) for path in dependencies if path.exists()]
    paths = set(_package_sources(root)) | {entrypoint} | set(present_dependencies)
    files = {path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(paths)}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "wnn.training-source.v2",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
        "missing_dependency_files": missing_dependencies,
        "runtime_dependencies": {"attrs": version("attrs")},
    }
