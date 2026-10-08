"""Validation for final-training teacher caches covering every supplied group."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from wnn.distillation_cache import (
    CACHE_FORMAT,
    CACHE_VERSION,
    FAMILY_ORDER,
    FAMILY_WIDTH,
    MAIN_WIDTH,
    SEQUENCE_ROWS,
    row_groups_sha256,
)
from wnn.splits import file_sha256


def _target_groups(data_root: Path, split: str) -> list[int]:
    """Read stable row-group identities, without loading sequence arrays."""
    with np.load(data_root / "cache" / f"{split}_targets.npz", allow_pickle=False) as index:
        groups = np.asarray(index["row_group"], dtype=np.int64)
    if groups.ndim != 1 or len(np.unique(groups)) != len(groups):
        raise ValueError(f"{split} target index must contain unique one-dimensional groups")
    return list(map(int, groups))


def validate_all_data_training_groups(
    train_groups: list[int],
    valid_groups: list[int],
    expected_train_groups: list[int],
    expected_valid_groups: list[int],
) -> None:
    """Require each supplied row group exactly once in the final-training pool."""
    for split, actual, expected in (
        ("train", train_groups, expected_train_groups),
        ("valid", valid_groups, expected_valid_groups),
    ):
        actual_groups = list(map(int, actual))
        expected_groups = list(map(int, expected))
        if (
            len(actual_groups) != len(set(actual_groups))
            or len(expected_groups) != len(set(expected_groups))
            or set(actual_groups) != set(expected_groups)
            or len(actual_groups) != len(expected_groups)
        ):
            raise ValueError(
                f"all-data student fit must include each supplied {split} group exactly once"
            )


def _valid_sha256(value: object, label: str) -> str:
    """Require a lowercase SHA-256 identity from dynamic metadata."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a hexadecimal SHA-256 digest") from error
    if value.lower() != value:
        raise ValueError(f"{label} must use lowercase hexadecimal")
    return value


def _validate_source_records(
    metadata: dict[str, Any],
    orders: dict[str, list[int]],
    split_sha256: str,
    allowed_fit_groups: dict[str, set[int]],
) -> None:
    """Bind each weighted teacher to its full-order predictions and checkpoint."""
    coefficients = metadata.get("source_coefficients")
    records = metadata.get("sources")
    if not isinstance(coefficients, list) or not isinstance(records, list):
        raise ValueError("all-data cache must record source coefficients and checkpoint records")
    if len(coefficients) < 4 or len(records) != len(coefficients):
        raise ValueError("all-data cache needs at least four uniquely recorded teacher sources")
    if {item.get("family") for item in coefficients if isinstance(item, dict)} != set(FAMILY_ORDER):
        raise ValueError("all-data cache must cover the four declared teacher families")
    source_runs = [item.get("run") for item in coefficients if isinstance(item, dict)]
    if len(source_runs) != len(coefficients) or len(set(source_runs)) != len(source_runs):
        raise ValueError("all-data teacher runs must be uniquely recorded")
    record_by_run = {item.get("run"): item for item in records if isinstance(item, dict)}
    if len(record_by_run) != len(records) or set(record_by_run) != set(source_runs):
        raise ValueError("all-data cache source records do not match weighted teacher runs")
    values: list[float] = []
    for source in coefficients:
        if not isinstance(source, dict):
            raise ValueError("teacher coefficient entries must be objects")
        run = source.get("run")
        family = source.get("family")
        if not isinstance(run, str) or family not in FAMILY_ORDER:
            raise ValueError("teacher source run or family is invalid")
        try:
            coefficient = float(source["coefficient"])
            epoch = int(source["epoch"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"teacher source settings are invalid: {run}") from error
        if not math.isfinite(coefficient) or coefficient <= 0 or epoch < 0:
            raise ValueError(f"teacher source coefficient or epoch is invalid: {run}")
        values.append(coefficient)
        record = record_by_run[run]
        try:
            record_coefficient = float(record["coefficient"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"teacher coefficient record is invalid: {run}") from error
        if (
            record.get("family") != family
            or record.get("epoch") != epoch
            or not math.isfinite(record_coefficient)
            or abs(record_coefficient - coefficient) > 1e-7
        ):
            raise ValueError(f"teacher coefficient record disagrees: {run}")
        _valid_sha256(record.get("checkpoint_sha256"), f"{run} checkpoint hash")
        closure = record.get("ancestor_closure")
        if not isinstance(closure, list) or not closure or not isinstance(closure[0], dict):
            raise ValueError(f"teacher {run} lacks its verified checkpoint ancestry")
        direct_groups = record.get("fit_row_groups")
        if not isinstance(direct_groups, dict) or direct_groups != closure[0].get("fit_row_groups"):
            raise ValueError(f"teacher {run} fit membership differs from its ancestry record")
        direct_counts = record.get("fit_group_counts")
        direct_hashes = record.get("fit_row_group_hashes")
        if not isinstance(direct_counts, dict) or not isinstance(direct_hashes, dict):
            raise ValueError(f"teacher {run} direct fit-membership summary is missing")
        if closure[0].get("run") != run or closure[0].get("checkpoint_sha256") != record.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"teacher {run} direct checkpoint differs from its ancestry record")
        for ancestor in closure:
            if not isinstance(ancestor, dict):
                raise ValueError(f"teacher {run} ancestry entries must be objects")
            ancestor_run = ancestor.get("run")
            _valid_sha256(ancestor.get("checkpoint_sha256"), f"{run} ancestor checkpoint hash")
            if ancestor.get("split_manifest_sha256") != split_sha256:
                raise ValueError(f"teacher {run} ancestry uses another clean split")
            fit_rows = ancestor.get("fit_row_groups")
            counts = ancestor.get("fit_group_counts")
            hashes = ancestor.get("fit_row_group_hashes")
            if (
                not isinstance(fit_rows, dict)
                or not isinstance(counts, dict)
                or not isinstance(hashes, dict)
            ):
                raise ValueError(f"teacher {run} ancestry lacks complete fit membership")
            for fit_split in ("train", "valid"):
                groups = list(map(int, fit_rows.get(fit_split, [])))
                if len(groups) != len(set(groups)) or not set(groups).issubset(
                    allowed_fit_groups[fit_split]
                ):
                    raise ValueError(
                        f"teacher {run} ancestor fits outside clean {fit_split} membership"
                    )
                if counts.get(fit_split) != len(groups) or hashes.get(
                    fit_split
                ) != row_groups_sha256(groups):
                    raise ValueError(f"teacher {run} ancestor fit-membership summary differs")
                if ancestor is closure[0] and (
                    direct_counts.get(fit_split) != len(groups)
                    or direct_hashes.get(fit_split) != row_groups_sha256(groups)
                ):
                    raise ValueError(f"teacher {run} direct fit-membership summary differs")
            if not isinstance(ancestor_run, str) or not ancestor_run:
                raise ValueError(f"teacher {run} ancestry run identity is invalid")
        predictions = record.get("predictions")
        if not isinstance(predictions, dict):
            raise ValueError(f"teacher prediction records are missing: {run}")
        for split in ("train", "valid"):
            prediction = predictions.get(split)
            if not isinstance(prediction, dict):
                raise ValueError(f"teacher {run} has no {split} prediction record")
            if prediction.get("row_groups_sha256") != row_groups_sha256(orders[split]):
                raise ValueError(f"teacher {run} {split} prediction order is incomplete")
            if prediction.get("checkpoint_sha256") != record.get("checkpoint_sha256"):
                raise ValueError(f"teacher {run} {split} prediction is bound to another checkpoint")
            _valid_sha256(prediction.get("prediction_sha256"), f"{run}/{split} prediction hash")
            _valid_sha256(prediction.get("sidecar_sha256"), f"{run}/{split} sidecar hash")
    total = math.fsum(values)
    if abs(total - 0.8125) > 1e-7 or abs(total - float(metadata.get("coefficient_sum", 0))) > 1e-7:
        raise ValueError("all-data teacher-source coefficients must retain the declared 0.8125 sum")


def validate_all_data_distillation_cache(
    cache_root: Path,
    data_root: Path,
    variant: str,
    *,
    require_families: bool,
) -> dict[str, Any]:
    """Fail closed unless targets exactly cover all supplied train and valid groups.

    This cache is for final training. It deliberately has no held-out evaluation
    membership and must never be used to claim held-out performance.
    """
    if variant == "history":
        from wnn.history_swap_cache import validate_history_distillation_cache

        return validate_history_distillation_cache(
            cache_root, data_root, require_families=require_families
        )
    if variant not in ("source", "normalized", "homogeneous"):
        raise ValueError(f"unknown distillation target variant: {variant}")
    cache_root = cache_root.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != CACHE_FORMAT or metadata.get("version") != CACHE_VERSION:
        raise ValueError("unsupported heterogeneous distillation cache format")
    if metadata.get("membership_mode") != "all_supplied_train_valid":
        raise ValueError("cache is not explicitly bound to all-supplied-data membership")
    if metadata.get("teacher_training_scope") != "clean_fold_fit_only":
        raise ValueError("cache does not declare clean-fold-only teacher training ancestry")
    if metadata.get("teacher_prediction_coverage") != "complete_train_valid_target_index":
        raise ValueError(
            "teacher predictions are not declared over the complete supplied target index"
        )
    if metadata.get("heldout_claim") is not False:
        raise ValueError("all-data cache cannot carry a held-out score claim")
    if tuple(metadata.get("family_order", ())) != FAMILY_ORDER:
        raise ValueError("distillation family-head order differs from the declared order")
    if metadata.get("evaluation_valid_groups") != []:
        raise ValueError("all-data student cache must have no evaluation groups")

    orders = {split: _target_groups(data_root, split) for split in ("train", "valid")}
    if metadata.get("fit_groups") != orders:
        raise ValueError("all-data cache row groups differ from complete target-index membership")
    for split in ("train", "valid"):
        if metadata.get(f"{split}_row_groups_sha256") != row_groups_sha256(orders[split]):
            raise ValueError(f"all-data cache {split} row-group digest differs")

    teacher_source = metadata.get("teacher_source_split")
    if not isinstance(teacher_source, dict):
        raise ValueError("all-data cache lacks its separate teacher-training split")
    split_path = Path(str(teacher_source.get("path", "")))
    if split_path.is_absolute() or ".." in split_path.parts:
        raise ValueError("teacher source split path must be repository-relative")
    resolved_split = (data_root / split_path).resolve(strict=True)
    if not resolved_split.is_relative_to(data_root):
        raise ValueError("teacher source split escapes the data root")
    expected_split_sha = _valid_sha256(teacher_source.get("sha256"), "teacher source split hash")
    if file_sha256(resolved_split) != expected_split_sha:
        raise ValueError("teacher source split changed after cache construction")
    teacher_manifest = json.loads(resolved_split.read_text())
    if teacher_manifest.get("version") != 1:
        raise ValueError("teacher source split has an unsupported manifest version")
    allowed_fit_groups = {
        split: set(map(int, teacher_manifest.get("fit", {}).get(split, [])))
        for split in ("train", "valid")
    }
    if not allowed_fit_groups["train"] or not allowed_fit_groups["valid"]:
        raise ValueError("teacher source split has empty clean fitting membership")
    manifest_excluded = list(map(int, teacher_manifest.get("evaluation", {}).get("valid", [])))
    excluded = teacher_source.get("evaluation_valid_groups")
    if not isinstance(excluded, list) or not excluded:
        raise ValueError("teacher source split must record its excluded evaluation groups")
    if list(map(int, excluded)) != manifest_excluded:
        raise ValueError("teacher source exclusion list differs from its clean split manifest")
    if teacher_source.get("teachers_fit_evaluation_groups") is not False:
        raise ValueError(
            "teacher source ancestry does not attest exclusion of its evaluation groups"
        )
    _validate_source_records(metadata, orders, expected_split_sha, allowed_fit_groups)

    files = metadata.get("files")
    if not isinstance(files, dict):
        raise ValueError("all-data cache file manifest is missing")
    verified: dict[str, str] = {}
    for split in ("train", "valid"):
        keys = [f"main_{variant}_{split}"]
        if require_families:
            keys.append(f"families_{split}")
        for key in keys:
            entry = files.get(key)
            if not isinstance(entry, dict):
                raise ValueError(f"all-data cache manifest is missing {key}")
            relative_path = Path(str(entry.get("path", "")))
            if relative_path.is_absolute() or len(relative_path.parts) != 1:
                raise ValueError(f"cache filename must be a single relative path: {key}")
            path = (cache_root / relative_path).resolve(strict=True)
            if not path.is_relative_to(cache_root):
                raise ValueError(f"cache file escapes its root: {key}")
            expected_groups = orders[split]
            row_groups = list(map(int, entry.get("row_groups", [])))
            if row_groups != expected_groups:
                raise ValueError(f"cache row groups are incomplete or reordered for {key}")
            if entry.get("row_groups_sha256") != row_groups_sha256(expected_groups):
                raise ValueError(f"cache row-group hash differs for {key}")
            width = MAIN_WIDTH if key.startswith("main_") else FAMILY_WIDTH
            shape = (len(expected_groups), SEQUENCE_ROWS, width)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(
                    f"cache shape or dtype differs for {key}: {array.shape}, {array.dtype}"
                )
            if entry.get("shape") != list(shape) or entry.get("dtype") != "float32":
                raise ValueError(f"cache manifest shape or dtype differs for {key}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 << 20):
                    digest.update(chunk)
            actual_sha = digest.hexdigest()
            if actual_sha != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
                raise ValueError(f"cache bytes disagree with metadata for {key}")
            for start in range(0, len(array), 64):
                if not np.isfinite(array[start : start + 64]).all():
                    raise ValueError(f"cache contains nonfinite values for {key}")
            verified[key] = actual_sha
    return {
        "metadata": metadata,
        "metadata_sha256": file_sha256(metadata_path),
        "verified_files": verified,
    }
