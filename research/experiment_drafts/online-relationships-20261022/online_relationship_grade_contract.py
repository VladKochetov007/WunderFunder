"""Frozen recipe and shared identity checks for the relationship grader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any  # JSON metadata is validated at runtime.

from online_relationship_grade_core import sha256_file

ROWS = 20_000
WIDTH = 117
RAW_END = 112
PARENT_START = 115
PARENT_END = 117
ARMS = ("A", "B", "C")
LAGS = (1, 16)
EXPECTED_CELLS = {(fold, lag, arm) for fold in (0, 1) for lag in LAGS for arm in ARMS}
ROW_WEIGHT_CONTRACT = {
    "mask_weight": False,
    "prop_power": 2.0,
    "prop_eps": 0.03,
    "prop_cache": False,
    "class_weight": None,
    "valid_true_mask": 0.03,
    "sync_free": False,
    "since_weight": 0.0,
    "since_floor": 0.1,
    "episode_weight": 0.0,
    "episode_floor": 0.1,
}


@dataclass(frozen=True)
class ParentCacheArtifacts:
    """Verified mmap and source identities for one fold's parent endpoint."""

    prediction: Any
    identity: dict[str, Any]


@dataclass(frozen=True)
class CandidateArtifacts:
    """Verified prediction mmap, sidecar and provenance identity for one cell."""

    prediction: Any
    identity: dict[str, Any]


def read_json(path: Path, *, max_bytes: int = 5_000_000) -> dict[str, Any]:
    """Read bounded UTF-8 JSON metadata and require an object root."""
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"metadata size is invalid: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata root must be an object: {path}")
    return value


def row_refs_sha256(refs: list[dict[str, Any]]) -> str:
    """Hash ordered, normalized source-qualified split and row-group references."""
    normalized = [{"split": str(ref["split"]), "row_group": int(ref["row_group"])} for ref in refs]
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def validate_recipe(
    recipe_path: Path, root: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    """Pin recipe bytes and require an admitted complete F0/F1 × lag × arm panel."""
    recipe = read_json(recipe_path)
    recipe_sha = sha256_file(recipe_path)
    if recipe_sha != expected_sha256:
        raise ValueError("recipe bytes differ from the supplied frozen recipe SHA256")
    if recipe.get("schema") != "wnn.online-relationships.v1":
        raise ValueError("unsupported online relationship recipe")
    if recipe.get("experiment_id") != "E-ONLINE-RELATIONSHIPS-20261022-R1":
        raise ValueError("unexpected online relationship experiment identity")
    if recipe.get("status") not in {"admitted", "completed"}:
        raise ValueError("recipe is not root-admitted for quality grading")
    cells = recipe.get("cells", [])
    if recipe.get("admitted_new_fits") != 12 or len(cells) != 12:
        raise ValueError("root has not admitted the complete 12-fit panel")
    grid = {(int(cell["fold"]), int(cell["lag"]), str(cell["arm"])) for cell in cells}
    if grid != EXPECTED_CELLS or len(grid) != len(cells):
        raise ValueError("recipe cells are not the exact F0/F1 × lag1/16 × A/B/C panel")
    expected_seeds = {0: 1001, 1: 1002}
    if any(int(cell["seed"]) != expected_seeds[int(cell["fold"])] for cell in cells):
        raise ValueError("recipe cell seed differs from its fold assignment")
    if len({str(cell["run"]) for cell in cells}) != 12:
        raise ValueError("recipe contains duplicate run identities")
    if len({str(cell["output"]) for cell in cells}) != 12:
        raise ValueError("recipe contains duplicate output directories")
    if recipe.get("new_gru_fits") != 0 or recipe.get("new_teacher_fits") != 0:
        raise ValueError("recipe includes unplanned teacher or GRU fits")

    scratch = Path(recipe["resource_contract"]["scratch"]).resolve()
    for cell in cells:
        if not _resolve(root, cell["output"]).is_relative_to(scratch):
            raise ValueError("candidate output directory escapes physical scratch")
    folds = {int(fold["fold"]): fold for fold in recipe.get("folds", [])}
    if set(folds) != {0, 1}:
        raise ValueError("recipe must freeze exactly F0 and F1")
    for fold_id, record in folds.items():
        if int(record["seed"]) != expected_seeds[fold_id]:
            raise ValueError(f"F{fold_id} recipe seed differs from its panel")
        split_path = _resolve(root, record["split_manifest"])
        subset_path = _resolve(root, record["subset"]["path"])
        _require_sha(split_path, record["split_manifest_sha256"], "campaign split")
        _require_sha(subset_path, record["subset"]["sha256"], "fit subset")
        if not _resolve(root, record["parent_cache"]).is_relative_to(scratch):
            raise ValueError(f"F{fold_id} parent cache escapes physical scratch")
    target = recipe.get("target_cache", {})
    if not target.get("path") or not target.get("sha256"):
        raise ValueError("recipe has no pinned official target cache")
    return recipe, recipe_sha


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _expect_path_value(root: Path, value: dict[str, Any], key: str, expected: Path) -> None:
    if _resolve(root, value.get(key, "")) != expected.resolve():
        raise ValueError(f"artifact path field {key} differs from its recipe source")


def _expect_path_hash(
    value: dict[str, Any], path_key: str, hash_key: str, expected_path: Path, expected_sha: str
) -> None:
    root = Path(__file__).resolve().parents[3]
    _expect_path_value(root, value, path_key, expected_path)
    if value.get(hash_key) != expected_sha:
        raise ValueError(f"artifact hash field {hash_key} differs from its recipe source")


def _verify_source_hashes(root: Path, source_hashes: dict[str, str]) -> None:
    """Verify every source module pinned by a cache or run provenance record."""
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("artifact has no source-code hash map")
    for name, expected in source_hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"source path escapes the repository: {name}")
        _require_sha(path, expected, f"source {name}")


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} is missing or its SHA256 differs from the frozen recipe")


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()
