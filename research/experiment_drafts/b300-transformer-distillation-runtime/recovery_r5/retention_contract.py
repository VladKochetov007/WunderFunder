"""Parse and structurally validate immutable R5 retention plans."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any  # Validated manifests contain heterogeneous JSON fields.

from epoch_contract import canonical_sha256, read_object
from retention_plan import PLAN_SCHEMA, _implementation_hashes

MAX_PLAN_BYTES = 32_000_000
INTENT_SCHEMA = "wnn.recovery-r5-retention-intent.v1"
COMPLETE_SCHEMA = "wnn.recovery-r5-retention-complete.v1"
HASH_RE = re.compile(r"[0-9a-f]{64}")


def _load_plan(path: Path, requested_sha256: str) -> tuple[dict[str, Any], str]:
    """Load one exact plan file and verify caller, embedded hash, and source hashes."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("retention plan must be a regular non-symlink file")
    path = path.resolve(strict=True)
    plan, raw = read_object(path, MAX_PLAN_BYTES)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        raw_sha256 != requested_sha256
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "ready"
        or plan.get("plan_sha256") != canonical_sha256(unsigned)
    ):
        raise ValueError("retention plan hash, schema, or status is invalid")
    if set(plan) != {
        "schema",
        "status",
        "selection_path",
        "selection_file_sha256",
        "selection",
        "selected_completed_epochs",
        "selection_rule",
        "implementation_sha256",
        "folds",
        "plan_sha256",
    }:
        raise ValueError("retention plan fields differ from the R5 contract")
    if (
        not isinstance(plan.get("folds"), list)
        or len(plan["folds"]) != 2
        or any(type(item.get("fold")) is not int for item in plan["folds"])
        or [item.get("fold") for item in plan["folds"]] != [0, 2]
    ):
        raise ValueError("retention plan must contain exactly ordered F0 and F2 fits")
    if plan.get("implementation_sha256") != _implementation_hashes():
        raise ValueError("retention validator/source code changed after plan creation")
    selected = plan.get("selected_completed_epochs")
    if type(selected) is not int or not 1 <= selected <= 30:
        raise ValueError("retention selected epoch is outside the fixed 30-epoch horizon")
    selection = plan.get("selection")
    if not isinstance(selection, dict) or selection.get("selected_completed_epochs") != selected:
        raise ValueError("retention selection is detached from its selected epoch")
    for fold in plan["folds"]:
        _validate_fold_plan(fold, selected)
    return plan, raw_sha256


def _validate_fold_plan(fold: dict[str, Any], selected_completed: int) -> None:
    """Require one selected checkpoint and exactly the indexed 29-checkpoint complement."""
    required = {
        "fold",
        "fit_dir",
        "sweep_dir",
        "identity",
        "config_path",
        "config_file_sha256",
        "config_recipe_sha256",
        "outer_split",
        "outer_split_sha256",
        "inner_split",
        "inner_split_sha256",
        "sweep_plan_sha256",
        "sweep_sha256",
        "index_sha256",
        "index_file_sha256",
        "index_entries",
        "final_model_sha256",
        "selected_completed_epochs",
        "selected_checkpoint",
        "remove",
        "curve",
        "fit_artifacts",
        "sweep_artifacts",
    }
    if not isinstance(fold, dict) or set(fold) != required:
        raise ValueError("retention fold fields differ from the R5 contract")
    entries, selected, removed, curve = (
        fold["index_entries"],
        fold["selected_checkpoint"],
        fold["remove"],
        fold["curve"],
    )
    if (
        type(fold["fold"]) is not int
        or fold["fold"] not in (0, 2)
        or type(fold["selected_completed_epochs"]) is not int
        or fold["selected_completed_epochs"] != selected_completed
        or not isinstance(entries, list)
        or len(entries) != 30
        or not isinstance(selected, dict)
        or set(selected) != {"path", "sha256", "size"}
        or not isinstance(removed, list)
        or len(removed) != 29
        or not isinstance(curve, list)
        or len(curve) != 30
    ):
        raise ValueError("retention fold lacks its full index, curve, or exact deletion set")

    previous_step = 0
    for position, (entry, record) in enumerate(zip(entries, curve, strict=True), start=1):
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {"epoch", "completed_epochs", "global_step", "checkpoint_path", "checkpoint_sha256"}
            or type(entry.get("epoch")) is not int
            or entry.get("epoch") != position - 1
            or type(entry.get("completed_epochs")) is not int
            or entry.get("completed_epochs") != position
            or type(entry.get("global_step")) is not int
            or entry.get("global_step") <= previous_step
            or not isinstance(entry.get("checkpoint_sha256"), str)
            or HASH_RE.fullmatch(entry["checkpoint_sha256"]) is None
            or entry.get("checkpoint_path")
            != f"epoch_checkpoints/epoch-{position:06d}-{entry['checkpoint_sha256']}.pt"
        ):
            raise ValueError("retention index entries are malformed or unordered")
        previous_step = entry["global_step"]
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "completed_epochs",
                "checkpoint_path",
                "checkpoint_sha256",
                "score_sha256",
                "prediction_manifest_sha256",
                "receipt_sha256",
                "score",
            }
            or record.get("completed_epochs") != position
            or record.get("checkpoint_path") != entry["checkpoint_path"]
            or record.get("checkpoint_sha256") != entry["checkpoint_sha256"]
            or not isinstance(record.get("score"), dict)
        ):
            raise ValueError("retention curve differs from the complete epoch index")
        for key in ("score_sha256", "prediction_manifest_sha256", "receipt_sha256"):
            if not isinstance(record.get(key), str) or HASH_RE.fullmatch(record[key]) is None:
                raise ValueError("retention curve contains an invalid content hash")
        wp = record["score"].get("wp")
        if (
            not isinstance(wp, (int, float))
            or isinstance(wp, bool)
            or not math.isfinite(wp)
            or not -1.000000001 <= wp <= 1.000000001
        ):
            raise ValueError("retention curve contains invalid pooled WP")

    chosen = entries[selected_completed - 1]
    if (
        selected.get("path") != chosen["checkpoint_path"]
        or selected.get("sha256") != chosen["checkpoint_sha256"]
        or type(selected.get("size")) is not int
        or selected["size"] < 0
    ):
        raise ValueError("retention selected checkpoint is detached from the epoch index")
    by_path = {item.get("path"): item for item in removed if isinstance(item, dict)}
    expected = {
        entry["checkpoint_path"]: entry["checkpoint_sha256"]
        for entry in entries
        if entry["completed_epochs"] != selected_completed
    }
    if (
        len(by_path) != len(removed)
        or set(by_path) != set(expected)
        or list(by_path) != sorted(by_path)
        or any(set(item) != {"path", "sha256", "size"} for item in removed)
        or any(by_path[path].get("sha256") != digest for path, digest in expected.items())
        or any(HASH_RE.fullmatch(str(item.get("sha256"))) is None for item in removed)
        or any(type(item.get("size")) is not int or item["size"] < 0 for item in removed)
    ):
        raise ValueError("retention deletion list differs from the indexed epoch complement")

    for field in (
        "config_file_sha256",
        "config_recipe_sha256",
        "outer_split_sha256",
        "inner_split_sha256",
        "sweep_plan_sha256",
        "sweep_sha256",
        "index_sha256",
        "index_file_sha256",
        "final_model_sha256",
    ):
        if not isinstance(fold.get(field), str) or HASH_RE.fullmatch(fold[field]) is None:
            raise ValueError(f"retention fold has invalid hash field {field}")
    for field in ("fit_dir", "sweep_dir", "config_path", "outer_split", "inner_split"):
        if not isinstance(fold.get(field), str) or not Path(fold[field]).is_absolute():
            raise ValueError(f"retention fold has invalid absolute path {field}")
    if not isinstance(fold.get("identity"), dict):
        raise ValueError("retention fold identity is malformed")


def _plan_path(value: str) -> Path:
    """Require an absolute readable regular metadata file without symlink traversal."""
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("retention references must be absolute regular files")
    return path.resolve(strict=True)
