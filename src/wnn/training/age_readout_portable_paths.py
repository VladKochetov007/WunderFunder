"""Validate physical bundle and work paths for the Kaggle CPU adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PortablePaths:
    """Map frozen logical recipe paths onto this Kaggle worker's mounted paths."""

    bundle_root: Path
    raw_sources: dict[str, Path]
    cache_root: Path
    output_root: Path
    resume_root: Path
    profile_root: Path
    workspace_root: Path
    transport_sha256: str
    cleanup_after_success: bool

    @classmethod
    def read(cls, path: Path) -> PortablePaths:
        """Validate the frozen disk roots, raw mounts and path-map location."""
        value = _json(path)
        if value.get("schema") != "wnn.age-readout-portable-path-map.v1":
            raise ValueError("portable path-map schema is unsupported")
        if type(value.get("cleanup_after_success", True)) is not bool:
            raise ValueError("portable cleanup flag must be a JSON boolean")
        paths = cls(
            bundle_root=_absolute(value["bundle_root"]),
            raw_sources={key: _absolute(raw) for key, raw in value["raw_sources"].items()},
            cache_root=_absolute(value["cache_root"]),
            output_root=_absolute(value["output_root"]),
            resume_root=_absolute(value["resume_root"]),
            profile_root=_absolute(value["profile_root"]),
            workspace_root=_absolute(value["workspace_root"]),
            transport_sha256=str(value["transport_sha256"]),
            cleanup_after_success=value.get("cleanup_after_success", True),
        )
        if set(paths.raw_sources) != {"train", "valid"}:
            raise ValueError("portable path map must name train and valid raw sources")
        kaggle_mode = bool(value.get("require_kaggle_paths", True))
        if kaggle_mode:
            input_root = Path("/kaggle/input")
            bundle_root = Path("/kaggle/working/disk_scratch/age_actor_bundle")
            work_root = Path("/kaggle/working/disk_scratch/age_actor_work")
            if paths.workspace_root != work_root or paths.bundle_root != bundle_root:
                raise ValueError("portable bundle and work paths differ from root's pinned roots")
            if any(not source.is_relative_to(input_root) for source in paths.raw_sources.values()):
                raise ValueError("portable Parquet inputs must remain under /kaggle/input")
            if any("wunder-funder" not in source.parts for source in paths.raw_sources.values()):
                raise ValueError(
                    "portable Parquet sources must come from the pinned Kaggle dataset"
                )
            if not path.absolute().is_relative_to(bundle_root):
                raise ValueError("portable path map must live inside the read-only bundle root")
            _reject_symlink_components(path.absolute(), kaggle_mode=True)
        _reject_symlink_components(paths.bundle_root, kaggle_mode=kaggle_mode)
        for source in paths.raw_sources.values():
            _reject_symlink_components(source, kaggle_mode=kaggle_mode)
            if kaggle_mode and not source.is_file():
                raise ValueError("portable raw source map must point to regular Parquet files")
        for destination in (
            paths.workspace_root,
            paths.cache_root,
            paths.output_root,
            paths.resume_root,
            paths.profile_root,
        ):
            if not destination.is_relative_to(paths.workspace_root):
                raise ValueError(
                    "portable outputs must stay under the configured physical work root"
                )
            _reject_symlink_components(destination, kaggle_mode=kaggle_mode, allow_missing=True)
        if len(paths.transport_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in paths.transport_sha256
        ):
            raise ValueError("portable transport identity must be a lowercase SHA256 digest")
        return paths

    def cache_dir(self, fold: int) -> Path:
        """Return the derived cache directory for one fold."""
        return self.cache_root / f"fold{fold}"


def _absolute(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("portable path maps must use absolute traversal-free paths")
    return path


def regular_bundle_file(path: Path, root: Path) -> Path:
    """Resolve a regular bundle file without accepting traversal or symlink paths."""
    absolute_root = root.absolute()
    absolute_path = path.absolute()
    if not absolute_path.is_relative_to(absolute_root):
        raise ValueError(f"portable file escapes its pinned root: {path}")
    cursor = absolute_root
    for part in absolute_path.relative_to(absolute_root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"portable file path contains a symlink: {cursor}")
    if not absolute_path.is_file():
        raise FileNotFoundError(f"portable input is not a regular file: {absolute_path}")
    return absolute_path


def _reject_symlink_components(
    path: Path, *, kaggle_mode: bool, allow_missing: bool = False
) -> None:
    if not kaggle_mode:
        return
    roots = (Path("/kaggle/working"), Path("/kaggle/input"))
    root = next((candidate for candidate in roots if path.is_relative_to(candidate)), None)
    if root is None:
        raise ValueError(f"portable path is outside Kaggle mounts: {path}")
    cursor = root
    for part in path.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"portable path contains a symlink component: {cursor}")
        if not cursor.exists() and not allow_missing:
            raise FileNotFoundError(cursor)
    if path.exists() and not (path.is_file() or path.is_dir()):
        raise ValueError(f"portable path is not a regular file or directory: {path}")
    if not allow_missing and not path.exists():
        raise FileNotFoundError(path)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value
