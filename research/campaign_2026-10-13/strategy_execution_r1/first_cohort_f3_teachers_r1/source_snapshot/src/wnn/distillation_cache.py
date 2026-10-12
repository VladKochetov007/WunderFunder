"""Validation helpers for campaign-bound heterogeneous distillation caches."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from wnn.splits import CampaignSplit

CACHE_FORMAT = "wnn.heterogeneous-distillation-cache"
CACHE_VERSION = 1
SEQUENCE_ROWS = 20_000
MAIN_WIDTH = 2
FAMILY_WIDTH = 8
FAMILY_ORDER = ("strong", "full_finetuned", "raw_wmse", "samefold_baseline")
MAIN_VARIANTS = ("source", "normalized", "homogeneous")
CLEANFOLD_T3_CACHE_PROFILE = "cleanfold_homogeneous_strong_t3_v1"


def row_groups_sha256(row_groups: list[int] | tuple[int, ...]) -> str:
    """Hash ordered int64 row-group identities in a platform-independent encoding."""
    ordered = np.asarray(row_groups, dtype="<i8")
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def file_sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    """Hash a large cache file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cleanfold_t3_profile(metadata: dict[str, Any], variant: str,
                                   require_families: bool) -> None:
    """Validate the narrow three-teacher cleanfold cache without relaxing legacy caches."""
    if variant != "homogeneous" or require_families:
        raise ValueError("cleanfold T3 cache supports only the two-column homogeneous target")
    if metadata.get("family_order") != ["strong"] or metadata.get("homogeneous_family") != "strong":
        raise ValueError("cleanfold T3 cache must contain only its declared strong family")
    coefficients = metadata.get("source_coefficients")
    sources = metadata.get("sources")
    if not isinstance(coefficients, list) or len(coefficients) != 3:
        raise ValueError("cleanfold T3 cache requires exactly three strong teachers")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("cleanfold T3 checkpoint records must match its three teachers")
    runs = [item.get("run") for item in coefficients if isinstance(item, dict)]
    if len(runs) != 3 or len(set(runs)) != 3 or any(not name for name in runs):
        raise ValueError("cleanfold T3 teacher run names must be unique and present")
    coefficient_values = []
    for item in coefficients:
        try:
            value = float(item["coefficient"])
            epoch = item["epoch"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("cleanfold T3 source coefficient record is invalid") from error
        if item.get("family") != "strong" or epoch != 2 or not np.isfinite(value) or value <= 0:
            raise ValueError("cleanfold T3 sources must be positive-weight strong epoch-2 teachers")
        coefficient_values.append(value)
    coefficient_sum = math.fsum(coefficient_values)
    if (abs(coefficient_sum - 0.3) > 1e-7
            or abs(coefficient_sum - float(metadata.get("coefficient_sum", 0.0))) > 1e-7
            or abs(float(metadata.get("homogeneous_scale", 0.0)) - 0.8125) > 1e-7):
        raise ValueError("cleanfold T3 requires three original 0.1 strong coefficients and a 0.8125 homogeneous scale")
    by_run = {item.get("run"): item for item in sources if isinstance(item, dict)}
    if len(by_run) != 3 or set(by_run) != set(runs):
        raise ValueError("cleanfold T3 checkpoint records do not match source coefficients")
    for item in coefficients:
        record = by_run[item["run"]]
        if (record.get("family") != "strong" or record.get("epoch") != 2
                or abs(float(record.get("coefficient", 0.0)) - float(item["coefficient"])) > 1e-7):
            raise ValueError(f"cleanfold T3 source record disagrees: {item['run']}")
    target = metadata.get("target_contract", {})
    if (target.get("formula") != "float32(0.8125 * float32_mean(three_epoch2_predictions))"
            or target.get("teacher_epoch") != 2
            or target.get("teacher_mean") != "unweighted arithmetic mean in float32"
            or target.get("clipping") != "none during cache construction"
            or abs(float(target.get("coefficient_multiplier", 0.0)) - 0.8125) > 1e-7):
        raise ValueError("cleanfold T3 target arithmetic contract differs")
    student = metadata.get("student_contract", {})
    expected_student = {
        "epochs": 6,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 5,
        "hidden": 256,
        "layers": 2,
        "batch": 16,
        "chunk": 2000,
        "distill_alpha": 0.9,
        "since_weight": 0.0,
        "inherit_init_preprocessing": False,
    }
    if any(student.get(key) != value for key, value in expected_student.items()):
        raise ValueError("cleanfold T3 cache is bound to a different student recipe")


def validate_distillation_cache(
    cache_root: Path,
    data_root: Path,
    split: CampaignSplit,
    variant: str,
    *,
    require_families: bool,
) -> dict[str, Any]:
    """Verify cache identity, fitting membership, row alignment, values and bytes.

    Args:
        cache_root: Directory written by ``scripts/build_distill_cache.py``.
        data_root: Repository/data root containing the frozen target indexes.
        split: Validated clean campaign split used by the student fit.
        variant: Main-target variant: source, normalized or homogeneous.
        require_families: Whether the training-only eight-column targets are needed.

    Returns:
        Parsed cache metadata and the hashes of the arrays verified for this run.

    Raises:
        FileNotFoundError: If the cache or a required target index is absent.
        ValueError: If any provenance, membership, shape, finite-value or hash check fails.
    """
    if variant not in MAIN_VARIANTS:
        raise ValueError(f"unknown distillation target variant: {variant}")
    cache_root = cache_root.resolve(strict=True)
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != CACHE_FORMAT or metadata.get("version") != CACHE_VERSION:
        raise ValueError("unsupported heterogeneous distillation cache format")
    if metadata.get("split_manifest_sha256") != split.sha256 or metadata.get("fold") != split.fold:
        raise ValueError("distillation cache was built for a different campaign fold")
    if metadata.get("fit_groups") != {
        "train": list(split.train_groups),
        "valid": list(split.valid_groups),
    }:
        raise ValueError("distillation cache fitting groups disagree with campaign split")
    if metadata.get("evaluation_valid_groups") != list(split.evaluation_groups):
        raise ValueError("distillation cache evaluation groups disagree with campaign split")
    cleanfold_t3 = metadata.get("cache_profile") == CLEANFOLD_T3_CACHE_PROFILE
    if cleanfold_t3:
        _validate_cleanfold_t3_profile(metadata, variant, require_families)
    elif tuple(metadata.get("family_order", ())) != FAMILY_ORDER:
        raise ValueError("distillation family-head order is not the declared four-family order")
    if not cleanfold_t3:
        if not metadata.get("sources") or not metadata.get("source_coefficients"):
            raise ValueError("distillation cache lacks source checkpoint or coefficient provenance")
        # JSON metadata is intentionally dynamic; verify its records rather than trust a summary field.
        coefficients = metadata["source_coefficients"]
        if (
            not isinstance(coefficients, list)
            or len(coefficients) < 4
            or any(not isinstance(item, dict) for item in coefficients)
        ):
            raise ValueError("distillation cache needs at least one teacher per family")
        if {item.get("family") for item in coefficients} != set(FAMILY_ORDER):
            raise ValueError("distillation sources do not cover the four declared families")
        source_runs = [item.get("run") for item in coefficients]
        if len(set(source_runs)) != len(source_runs) or any(not run for run in source_runs):
            raise ValueError("distillation source run names must be present and unique")
        try:
            coefficient_values = [float(item["coefficient"]) for item in coefficients]
            coefficient_sum = math.fsum(coefficient_values)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("distillation source coefficient is invalid") from error
        if any(not np.isfinite(value) or value <= 0 for value in coefficient_values):
            raise ValueError("distillation source coefficients must be finite and positive")
        if (
            abs(coefficient_sum - float(metadata.get("coefficient_sum", 0.0))) > 1e-7
            or abs(coefficient_sum - 0.8125) > 1e-7
        ):
            raise ValueError("source-scale coefficients must sum to 0.8125")
        source_records = metadata["sources"]
        if not isinstance(source_records, list) or any(
            not isinstance(item, dict) for item in source_records
        ):
            raise ValueError("distillation source checkpoint records are invalid")
        records_by_run = {item.get("run"): item for item in source_records}
        if len(records_by_run) != len(source_records) or set(records_by_run) != set(source_runs):
            raise ValueError("distillation source records do not match the declared runs")
        for item in coefficients:
            record = records_by_run[item["run"]]
            try:
                record_coefficient = float(record["coefficient"])
                expected_coefficient = float(item["coefficient"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"distillation source record is invalid: {item['run']}") from error
            if (
                record.get("family") != item["family"]
                or record.get("epoch") != item.get("epoch", 3)
                or abs(record_coefficient - expected_coefficient) > 1e-7
            ):
                raise ValueError(f"distillation source record disagrees: {item['run']}")
        if metadata.get("homogeneous_family") not in FAMILY_ORDER:
            raise ValueError("homogeneous control must name a declared teacher family")
        epoch_policy = metadata.get("epoch_convention", {})
        if (
            epoch_policy.get("friend_at_ep1") != "second additional epoch"
            or epoch_policy.get("student_epochs") != 2
            or epoch_policy.get("student_checkpoint_policy") != "fixed"
            or epoch_policy.get("student_checkpoint_epoch") != 1
        ):
            raise ValueError("distillation cache student checkpoint convention differs")
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise ValueError("distillation cache file manifest is missing")

    verified: dict[str, str] = {}
    for source_split, fitting_groups in (
        ("train", split.train_groups),
        ("valid", split.valid_groups),
    ):
        with np.load(
            data_root / "cache" / f"{source_split}_targets.npz", allow_pickle=False
        ) as targets:
            target_groups = np.asarray(targets["row_group"], dtype=np.int64)
        if len(np.unique(target_groups)) != len(target_groups):
            raise ValueError(f"{source_split} target index repeats a row group")
        fitting = set(map(int, fitting_groups))
        expected_groups = [int(group) for group in target_groups if int(group) in fitting]
        if set(expected_groups) != fitting:
            raise ValueError(f"{source_split} target index omits campaign fitting groups")

        keys = [f"main_{variant}_{source_split}"]
        if require_families:
            keys.append(f"families_{source_split}")
        for key in keys:
            entry = files.get(key)
            if not isinstance(entry, dict):
                raise ValueError(f"distillation cache manifest is missing {key}")
            relative_path = Path(str(entry.get("path", "")))
            if relative_path.is_absolute() or len(relative_path.parts) != 1:
                raise ValueError(f"cache filename must be a single relative path: {key}")
            path = (cache_root / relative_path).resolve(strict=True)
            try:
                path.relative_to(cache_root)
            except ValueError as error:
                raise ValueError(f"cache file escapes the registered cache root: {key}") from error
            if not path.is_file():
                raise FileNotFoundError(path)
            row_groups = list(map(int, entry.get("row_groups", [])))
            if row_groups != expected_groups or row_groups_sha256(row_groups) != entry.get(
                "row_groups_sha256"
            ):
                raise ValueError(f"cache row groups are misaligned for {key}")
            expected_width = MAIN_WIDTH if key.startswith("main_") else FAMILY_WIDTH
            shape = (len(expected_groups), SEQUENCE_ROWS, expected_width)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(
                    f"cache shape or dtype differs for {key}: {array.shape}, {array.dtype}"
                )
            if entry.get("shape") != list(shape) or entry.get("dtype") != "float32":
                raise ValueError(f"cache manifest shape or dtype differs for {key}")
            digest = hashlib.sha256()
            with path.open("rb") as file:
                while chunk := file.read(8 << 20):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
                raise ValueError(f"cache bytes disagree with metadata for {key}")
            for start in range(0, len(array), 64):
                if not np.isfinite(array[start : start + 64]).all():
                    raise ValueError(f"cache contains nonfinite targets for {key}")
            verified[key] = actual_sha256

    return {
        "metadata": metadata,
        "metadata_sha256": file_sha256(metadata_path),
        "verified_files": verified,
    }
