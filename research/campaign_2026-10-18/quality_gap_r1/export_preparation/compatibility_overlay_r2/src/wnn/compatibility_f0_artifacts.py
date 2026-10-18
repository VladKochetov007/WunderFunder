"""Exact restored-file bindings for the isolated Main F0 compatibility export."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_F0_RESTORE_BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1")
_F0_ARTIFACTS = {
    "control_checkpoint": {
        "path": "artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt",
        "restore_path": _F0_RESTORE_BASE
        / "restored-controls-v1/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt",
        "bytes": 3983015,
        "sha256": "e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda",
    },
    "control_provenance": {
        "path": "artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/provenance.json",
        "restore_path": _F0_RESTORE_BASE
        / "restored-controls-v1/artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/provenance.json",
        "bytes": 166152,
        "sha256": "bdf643043884b1344564b636a2972ad52b8202ee30cf22598b1afd90405dd961",
    },
    "propensity_checkpoint": {
        "path": "artifacts/runs/c20260929_prop_f0/model.pt",
        "restore_path": _F0_RESTORE_BASE
        / "restored-propensity-v1/artifacts/runs/c20260929_prop_f0/model.pt",
        "bytes": 165623,
        "sha256": "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7",
    },
    "propensity_provenance": {
        "path": "artifacts/runs/c20260929_prop_f0/provenance.json",
        "restore_path": _F0_RESTORE_BASE
        / "restored-propensity-v1/artifacts/runs/c20260929_prop_f0/provenance.json",
        "bytes": 21626,
        "sha256": "f6d6cf68db53c728556c1750e869e65d5d4813d622399aec768809a47985e812",
    },
}


def _file_sha256(path: Path) -> str:
    """Hash a small restored artifact without loading tensors into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_f0_checkpoint_bindings() -> dict[str, dict[str, object]]:
    """Return immutable expected logical paths and exact B restore bindings."""
    return {
        name: {
            "path": str(binding["path"]),
            "restore_path": str(binding["restore_path"]),
            "bytes": int(binding["bytes"]),
            "sha256": str(binding["sha256"]),
        }
        for name, binding in _F0_ARTIFACTS.items()
    }


def verify_f0_checkpoint_bindings(project_root: Path) -> dict[str, dict[str, object]]:
    """Verify staged symlinks resolve to the pinned B restore files and hashes."""
    verified: dict[str, dict[str, object]] = {}
    for name, expected in expected_f0_checkpoint_bindings().items():
        restored_path = Path(str(expected["restore_path"]))
        staged_path = project_root / str(expected["path"])
        if staged_path.resolve(strict=True) != restored_path.resolve(strict=True):
            raise ValueError(f"staged F0 artifact does not point to its restored source: {name}")
        for path in (restored_path, staged_path):
            stat = path.stat()
            if stat.st_size != expected["bytes"] or _file_sha256(path) != expected["sha256"]:
                raise ValueError(f"restored F0 artifact has an unexpected identity: {name}")
        verified[name] = expected
    return verified


def verify_f0_overlay_support_files(
    overlay_root: Path,
    entries: list[dict[str, object]],
    expected_paths: set[str],
) -> None:
    """Verify safe paths and exact hashes for every staged overlay support file.

    Args:
        overlay_root: Root directory containing the staged support files.
        entries: Manifest entries with relative paths, sizes, and SHA-256 values.
        expected_paths: Exact relative-path set required for this overlay.

    Raises:
        ValueError: If paths are unsafe, files differ, or the inventory is incomplete.
    """
    if not isinstance(entries, list):
        raise ValueError("compatibility overlay support inventory is invalid")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("compatibility overlay support entry is invalid")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            raise ValueError("compatibility overlay support path is invalid")
        relative = Path(raw_path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != raw_path
            or raw_path in seen
        ):
            raise ValueError("compatibility overlay manifest has an unsafe support path")
        seen.add(raw_path)
        path = overlay_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"compatibility overlay support file is missing or linked: {relative}")
        stat = path.stat()
        if stat.st_size != entry.get("bytes") or _file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"compatibility overlay support file changed: {relative}")
    if seen != expected_paths:
        raise ValueError("compatibility overlay support inventory is incomplete or unexpected")


def verify_f0_canonical_cache_link(
    source: Path,
    expected_source: Path,
    expected_resolved: Path,
) -> tuple[str, str]:
    """Require the staged cache symlink to preserve its observed B source chain.

    Args:
        source: The staged project cache path.
        expected_source: The observed logical cache symlink on B.
        expected_resolved: The resolved cache target recorded on B.

    Returns:
        The logical observed source and the resolved target as absolute strings.

    Raises:
        ValueError: If either symlink or the resolved target differs from its pin.
    """
    if not source.is_symlink() or os.readlink(source) != str(expected_source):
        raise ValueError("canonical F0 cache must link to the observed B data cache path")
    if not expected_source.is_symlink():
        raise ValueError("observed canonical F0 cache path is no longer a symlink")
    resolved = source.resolve(strict=True)
    if resolved != expected_resolved:
        raise ValueError("canonical F0 cache resolves to an unexpected source")
    if expected_source.resolve(strict=True) != resolved:
        raise ValueError("observed canonical F0 cache link changed its resolved source")
    return str(expected_source), str(resolved)
