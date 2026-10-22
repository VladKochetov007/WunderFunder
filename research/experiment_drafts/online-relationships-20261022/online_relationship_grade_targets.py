"""Load and align outer-fold labels only after every endpoint passes validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any  # JSON metadata is validated at runtime.

import numpy as np
from online_relationship_grade_contract import ROWS, _require_sha, _resolve, read_json
from online_relationship_grade_core import sha256_file

ROOT = Path(__file__).resolve().parents[3]


def load_targets(
    recipe: dict[str, Any], fold_records: dict[int, dict[str, Any]]
) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Open outer-fold labels only after every endpoint has passed its artifact gate."""
    target_record = recipe["target_cache"]
    target_path = _resolve(ROOT, target_record["path"])
    _require_sha(target_path, target_record["sha256"], "official target cache")
    stat = target_path.stat()
    for fold_id, fold in fold_records.items():
        split = read_json(_resolve(ROOT, fold["split_manifest"]))
        source = split["sources"].get("cache/valid_targets.npz", {})
        if source.get("bytes") != stat.st_size or source.get("mtime_ns") != stat.st_mtime_ns:
            raise ValueError(
                f"F{fold_id} target cache file identity differs from the split manifest"
            )

    with np.load(target_path, allow_pickle=False) as archive:
        required = {"row_group", "y", "mask"}
        if not required.issubset(archive.files):
            raise ValueError("official target archive lacks row_group, y or mask")
        groups = np.asarray(archive["row_group"])
        targets = np.asarray(archive["y"])
        mask = np.asarray(archive["mask"])
    if groups.ndim != 1 or groups.dtype.kind not in "iu" or len(np.unique(groups)) != len(groups):
        raise ValueError("official target row groups are malformed or duplicated")
    if targets.shape != (len(groups), ROWS, 2) or mask.shape != (len(groups), ROWS):
        raise ValueError("official target arrays do not match the full valid-cache schema")
    if targets.dtype.str != "<f4":
        raise ValueError("official target labels must use the cache's little-endian FP32 values")
    if mask.dtype != np.bool_:
        raise ValueError("official target score mask must be boolean")
    if not np.isfinite(targets[mask]).all():
        raise ValueError("official target cache contains nonfinite scored labels")
    position = {int(group): index for index, group in enumerate(groups)}
    folds: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for fold_id, fold in fold_records.items():
        eval_groups = [int(group) for group in fold["evaluation_groups"]]
        if any(group not in position for group in eval_groups):
            raise ValueError(f"F{fold_id} evaluation groups are missing from the target cache")
        indices = np.asarray([position[group] for group in eval_groups], dtype=np.int64)
        split = read_json(_resolve(ROOT, fold["split_manifest"]))
        component_by_group = {
            int(record["row_group"]): int(record["component"])
            for record in split["sequences"]["valid"]
        }
        components = np.asarray(
            [component_by_group[group] for group in eval_groups], dtype=np.int64
        )
        folds[fold_id] = targets[indices], mask[indices], components
    if sha256_file(target_path) != target_record["sha256"]:
        raise ValueError("official target cache changed during scoring")
    return groups, folds
