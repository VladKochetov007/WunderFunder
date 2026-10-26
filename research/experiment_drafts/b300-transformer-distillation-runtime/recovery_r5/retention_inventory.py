"""Inventory retained R5 metadata and exact prediction payload membership."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any  # Validated manifests contain heterogeneous JSON fields.

from epoch_contract import file_sha256, safe_file

_STALE_RECEIPT_TEMP = re.compile(r"\.retention_(?:intent|complete)\.json\.[A-Za-z0-9_-]+\.partial")


def _artifact(root: Path, relative: str) -> dict[str, Any]:
    """Describe one regular in-root artifact with its exact size and digest."""
    path = safe_file(root, relative)
    return {"path": relative, "sha256": file_sha256(path), "size": path.stat().st_size}


def _tree_artifacts(root: Path, *, fit: bool) -> list[dict[str, Any]]:
    """Inventory retained fit/sweep metadata without copying prediction payloads."""
    result: list[dict[str, Any]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            if (current_path / name).is_symlink():
                raise ValueError("retention roots must not contain symlink directories")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("retention roots may contain only regular files")
            relative = path.relative_to(root).as_posix()
            parts = Path(relative).parts
            if fit and parts[0] == "epoch_checkpoints":
                continue
            if fit and len(parts) == 1 and _STALE_RECEIPT_TEMP.fullmatch(name):
                continue
            if not fit and "predictions" in parts and name != "manifest.json":
                continue
            result.append(_artifact(root, relative))
    return sorted(result, key=lambda item: item["path"])


def _validate_prediction_inventory(
    sweep_dir: Path, completed: int, manifest: dict[str, Any]
) -> None:
    """Reject unmanifested or missing payload files in a scored epoch export."""
    directory = sweep_dir / "epochs" / f"epoch-{completed:06d}" / "predictions"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("retained-epoch prediction directory is missing or unsafe")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("retained-epoch prediction manifest lacks its file inventory")
    names = {"manifest.json"}
    for sequence in sequences:
        if not isinstance(sequence, dict) or not isinstance(sequence.get("file"), str):
            raise ValueError("retained-epoch prediction inventory is malformed")
        name = sequence["file"]
        if Path(name).name != name or name in names:
            raise ValueError("retained-epoch prediction filenames must be unique and local")
        names.add(name)
    observed: set[str] = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("retained-epoch prediction inventory contains an unsafe path")
        observed.add(path.name)
    if observed != names:
        raise ValueError("retained-epoch prediction files differ from their manifest")
