"""Build and validate the all-data-parent history target cache."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from wnn.distillation_cache import (
    CACHE_FORMAT,
    CACHE_VERSION,
    FAMILY_ORDER,
    MAIN_WIDTH,
    SEQUENCE_ROWS,
    file_sha256,
    row_groups_sha256,
)

HISTORY_CACHE_SCHEMA = "wnn.h230.history-swap-cache.v1"
BASE_CACHE_PATH = "cache/bd26_h230_fold0_teacher_family_all_rows_v1"
BASE_METADATA_SHA256 = "b05e652f5fdec4e6b1e4e07d1d3a705655d4588ab8ee9d802ccdf4339fd66500"
SOURCE_MANIFEST_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
    "h230_source_v3_all_data_distill/H230_ALLDATA_DISTILL_SOURCE_MANIFEST_V1_20261008.json"
)
SOURCE_MANIFEST_SHA256 = "9938cf880addd22467dd71fc81fc550ea3ab9c7ba792c84cbd73f7073e7de15d"
STRONG_RUNS = ("c20261003_combo_s3_f0", "c20261003_combo_s4_f0")
PARENT_RUN = "bd26_h230_all_data_parent_b16_ep3_retry3_v1"
PARENT_ATTEMPT = "h230-all-data-parent-fit-b16-ep3-retry3-v1-20261008"
PARENT_CONTRACT_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/"
    "H230_ALLDATA_PARENT_FIT_RETRY3_CONTRACT_V1_20261008.json"
)
PARENT_TERMINAL_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/"
    "h230-all-data-parent-fit-b16-ep3-retry3-v1-20261008.json"
)
PARENT_PROVENANCE_PATH = f"artifacts/runs/{PARENT_RUN}/provenance.json"


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    if value.lower() != value:
        raise ValueError(f"{label} must use lowercase hexadecimal")
    return value


def _repo_file(root: Path, value: object, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a repository-relative path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes the repository root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _row_groups(root: Path, split: str) -> list[int]:
    with np.load(root / "cache" / f"{split}_targets.npz", allow_pickle=False) as index:
        groups = np.asarray(index["row_group"], dtype=np.int64)
    if groups.ndim != 1 or len(groups) != len(np.unique(groups)):
        raise ValueError(f"{split} target index must contain unique row groups")
    return list(map(int, groups))


def combine_history_chunk(
    main_source: np.ndarray,
    strong_seed3: np.ndarray,
    strong_seed4: np.ndarray,
    parent: np.ndarray,
) -> np.ndarray:
    """Apply the frozen replacement with FP64 accumulation and one FP32 cast."""
    arrays = (main_source, strong_seed3, strong_seed4, parent)
    shape = main_source.shape
    if len(shape) != 3 or shape[-1] != MAIN_WIDTH:
        raise ValueError("history target inputs must be rank-three two-column arrays")
    for array in arrays:
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError("history target inputs must share shape and float32 dtype")
        if not np.isfinite(array).all():
            raise ValueError("history target inputs contain nonfinite values")
    total = main_source.astype(np.float64)
    scratch = np.empty_like(total)
    np.multiply(strong_seed3, np.float64(0.15), out=scratch)
    np.subtract(total, scratch, out=total)
    np.multiply(strong_seed4, np.float64(0.15), out=scratch)
    np.subtract(total, scratch, out=total)
    np.multiply(parent, np.float64(0.30), out=scratch)
    np.add(total, scratch, out=total)
    result = total.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("history target result is nonfinite after its float32 cast")
    return result


def validate_prediction_export(
    prediction_path: Path,
    sidecar_path: Path,
    *,
    expected_run: str,
    expected_split: str,
    expected_epoch: int,
    expected_checkpoint_sha256: str,
    expected_groups: list[int],
    sequence_rows: int = SEQUENCE_ROWS,
    required_inference_precision: str | None = "fp32",
    require_features_only: bool = False,
) -> dict[str, Any]:
    """Validate a full-order FP32 two-target prediction and its producer sidecar."""
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("run") != expected_run or sidecar.get("split") != expected_split:
        raise ValueError("parent prediction sidecar run or split differs")
    if sidecar.get("epoch") != expected_epoch:
        raise ValueError("parent prediction sidecar epoch differs")
    if sidecar.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("parent prediction sidecar checkpoint differs")
    inference_precision = sidecar.get("inference_precision")
    if (
        required_inference_precision is not None
        and inference_precision != required_inference_precision
    ):
        raise ValueError("parent predictions must be exported with FP32 inference")
    if required_inference_precision is None and inference_precision not in {"fp16", "fp32"}:
        raise ValueError("source teacher prediction precision is invalid")
    if require_features_only and sidecar.get("features_only") is not True:
        raise ValueError("parent prediction export must read features only")
    if require_features_only and sidecar.get("row_groups_sha256") != row_groups_sha256(
        expected_groups
    ):
        raise ValueError("parent prediction sidecar must bind the exact target-index group order")
    if require_features_only and (
        sidecar.get("shape") != [len(expected_groups), sequence_rows, MAIN_WIDTH]
        or sidecar.get("dtype") != "float32"
    ):
        raise ValueError("parent prediction sidecar shape or dtype differs")
    if list(map(int, sidecar.get("row_groups", []))) != expected_groups:
        raise ValueError("parent predictions do not cover the target-index order exactly")
    expected_group_hash = row_groups_sha256(expected_groups)
    if sidecar.get("row_groups_sha256", expected_group_hash) != expected_group_hash:
        raise ValueError("parent prediction row-group digest differs")
    expected_shape = (len(expected_groups), sequence_rows, MAIN_WIDTH)
    array = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ValueError(f"parent prediction shape/dtype differs: {array.shape}, {array.dtype}")
    digest = file_sha256(prediction_path)
    if digest != sidecar.get("prediction_sha256"):
        raise ValueError("parent prediction bytes differ from the producer sidecar")
    for start in range(0, len(array), 64):
        if not np.isfinite(array[start : start + 64]).all():
            raise ValueError("parent predictions contain nonfinite values")
    return {
        "path": str(prediction_path),
        "sha256": digest,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": file_sha256(sidecar_path),
        "shape": list(expected_shape),
        "dtype": "float32",
        "row_groups_sha256": expected_group_hash,
        "inference_precision": inference_precision,
    }


def _validate_parent_fit(
    data_root: Path,
    expected_groups: dict[str, list[int]],
    parent: dict[str, Any],
) -> dict[str, Any]:
    """Check the exact all-supplied parent run, terminal controller receipt and files."""
    checkpoint_path = _repo_file(data_root, parent.get("checkpoint_path"), "parent checkpoint")
    contract_path = _repo_file(data_root, parent.get("contract_path"), "parent fit contract")
    terminal_path = _repo_file(
        data_root, parent.get("terminal_receipt_path"), "parent terminal receipt"
    )
    provenance_path = _repo_file(data_root, parent.get("provenance_path"), "parent provenance")
    expected_checkpoint_sha = _sha256(parent.get("checkpoint_sha256"), "parent checkpoint hash")
    expected_contract_sha = _sha256(parent.get("contract_sha256"), "parent contract hash")
    expected_terminal_sha = _sha256(parent.get("terminal_receipt_sha256"), "parent terminal hash")
    expected_provenance_sha = _sha256(parent.get("provenance_sha256"), "parent provenance hash")
    if parent.get("run") != PARENT_RUN or parent.get("attempt_id") != PARENT_ATTEMPT:
        raise ValueError("history source is not the frozen H230 all-data parent attempt")
    if file_sha256(checkpoint_path) != expected_checkpoint_sha:
        raise ValueError("all-data parent checkpoint bytes changed")
    if file_sha256(contract_path) != expected_contract_sha:
        raise ValueError("parent fit contract bytes changed")
    if file_sha256(terminal_path) != expected_terminal_sha:
        raise ValueError("parent terminal controller receipt bytes changed")
    if file_sha256(provenance_path) != expected_provenance_sha:
        raise ValueError("parent trainer provenance bytes changed")
    contract = json.loads(contract_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    if contract.get("attempt_id") != PARENT_ATTEMPT or contract.get("contract_id") != (
        "H230-ALLDATA-PARENT-GRU256-B16-EP3-RETRY3-V1"
    ):
        raise ValueError("parent fit contract is not the frozen H230 retry3 contract")
    source = contract.get("source", {})
    source_manifest_path = _repo_file(
        data_root, source.get("manifest_path"), "parent source manifest"
    )
    source_manifest_sha = _sha256(source.get("manifest_sha256"), "parent source manifest hash")
    if file_sha256(source_manifest_path) != source_manifest_sha:
        raise ValueError("parent source manifest bytes changed")
    if terminal.get("id") != PARENT_ATTEMPT or terminal.get("status") != "finished":
        raise ValueError("parent controller attempt is not terminal-success")
    if terminal.get("phase") != "terminal" or terminal.get("exit_status") != 0:
        raise ValueError("parent controller attempt did not exit successfully")
    if terminal.get("contract_sha256") != expected_contract_sha:
        raise ValueError("parent controller receipt binds another fit contract")
    initializer = contract.get("initializer_ancestry")
    if (
        not isinstance(initializer, dict)
        or initializer.get("reference") != "c20261003_combo_s3_f0@ep3"
        or initializer.get("independently_trained_own_checkpoint") is not True
        or initializer.get("teammate_weights_used") is not False
    ):
        raise ValueError(
            "parent initializer ancestry is not the declared own clean-fold checkpoint"
        )
    initializer_path = _repo_file(data_root, initializer.get("path"), "parent initializer")
    initializer_sha = _sha256(initializer.get("sha256"), "parent initializer checkpoint hash")
    if file_sha256(initializer_path) != initializer_sha:
        raise ValueError("parent initializer checkpoint bytes changed")
    args = provenance.get("args", {})
    if (
        args.get("name") != PARENT_RUN
        or args.get("seed") != 3
        or args.get("train_source") != "train+valid"
        or args.get("valid_fold") != -1
        or args.get("include_heldout_valid") is not True
        or args.get("no_eval_during_fit") is not True
        or args.get("epochs") != 4
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != 3
    ):
        raise ValueError(
            "parent trainer provenance does not match the all-data fixed-epoch3 recipe"
        )
    recorded_checkpoint = provenance.get("checkpoint_sha256", {}).get("model_ep3.pt")
    if recorded_checkpoint != expected_checkpoint_sha:
        raise ValueError("parent provenance checkpoint hash differs")
    if provenance.get("initial_checkpoint_sha256") != initializer_sha:
        raise ValueError("parent initial checkpoint differs from the frozen own initializer")
    fit_groups = provenance.get("fit_row_groups")
    if not isinstance(fit_groups, dict):
        raise ValueError("parent provenance lacks exact fitting membership")
    membership: dict[str, Any] = {}
    for split in ("train", "valid"):
        actual = list(map(int, fit_groups.get(split, [])))
        expected = expected_groups[split]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError(f"parent was not fitted on every supplied {split} group")
        membership[split] = {
            "count": len(actual),
            "sorted_row_groups_sha256": row_groups_sha256(sorted(actual)),
        }
    return {
        "checkpoint_path": str(checkpoint_path.relative_to(data_root)),
        "checkpoint_sha256": expected_checkpoint_sha,
        "checkpoint_epoch_zero_based": 3,
        "run": PARENT_RUN,
        "fit_scope": "all_supplied_train_valid",
        "fit_membership": membership,
        "contract_path": str(contract_path.relative_to(data_root)),
        "contract_sha256": expected_contract_sha,
        "terminal_receipt_path": str(terminal_path.relative_to(data_root)),
        "terminal_receipt_sha256": expected_terminal_sha,
        "provenance_path": str(provenance_path.relative_to(data_root)),
        "provenance_sha256": expected_provenance_sha,
        "initial_checkpoint_sha256": provenance.get("initial_checkpoint_sha256"),
        "initializer": {
            "path": str(initializer_path.relative_to(data_root)),
            "sha256": initializer_sha,
            "reference": initializer["reference"],
            "independently_trained_own_checkpoint": True,
            "teammate_weights_used": False,
        },
        "source_manifest_path": str(source_manifest_path.relative_to(data_root)),
        "source_manifest_sha256": source_manifest_sha,
    }


def _verify_output_array(
    root: Path,
    entry: dict[str, Any],
    key: str,
    groups: list[int],
) -> str:
    relative = Path(str(entry.get("path", "")))
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError(f"history target path must be a single relative filename: {key}")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"history target escapes its cache root: {key}")
    shape = (len(groups), SEQUENCE_ROWS, MAIN_WIDTH)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != shape or array.dtype != np.float32:
        raise ValueError(f"history target shape/dtype differs for {key}")
    if entry.get("shape") != list(shape) or entry.get("dtype") != "float32":
        raise ValueError(f"history target metadata shape/dtype differs for {key}")
    if list(map(int, entry.get("row_groups", []))) != groups:
        raise ValueError(f"history target rows are incomplete or reordered for {key}")
    group_hash = row_groups_sha256(groups)
    if entry.get("row_groups_sha256") != group_hash:
        raise ValueError(f"history target row-group digest differs for {key}")
    digest = file_sha256(path)
    if digest != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
        raise ValueError(f"history target bytes differ from metadata for {key}")
    for start in range(0, len(array), 64):
        if not np.isfinite(array[start : start + 64]).all():
            raise ValueError(f"history target contains nonfinite values: {key}")
    return digest


def validate_history_distillation_cache(
    cache_root: Path,
    data_root: Path,
    *,
    require_families: bool,
) -> dict[str, Any]:
    """Validate history targets and their separate, unchanged family-cache dependency."""
    if not require_families:
        raise ValueError("the H230 history swap requires the unchanged family target arrays")
    cache_root = cache_root.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("schema") != HISTORY_CACHE_SCHEMA
        or metadata.get("format") != CACHE_FORMAT
        or metadata.get("version") != CACHE_VERSION
        or metadata.get("variant") != "history"
    ):
        raise ValueError("unsupported H230 history cache schema or variant")
    history = metadata.get("history_swap")
    if not isinstance(history, dict):
        raise ValueError("history cache lacks source and parent provenance")
    base = history.get("base_cache")
    if not isinstance(base, dict) or base.get("path") != BASE_CACHE_PATH:
        raise ValueError("history cache must use the frozen V3 base cache path")
    if (
        base.get("source_manifest_path") != SOURCE_MANIFEST_PATH
        or base.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("history cache source manifest differs from the sealed V3 source")
    source_manifest_path = _repo_file(
        data_root, base.get("source_manifest_path"), "V3 source manifest"
    )
    if file_sha256(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("sealed V3 source manifest bytes changed")
    base_root = _repo_file(
        data_root, base.get("path") + "/metadata.json", "base cache metadata"
    ).parent
    base_sha = _sha256(base.get("metadata_sha256"), "base cache metadata hash")
    if base_sha != BASE_METADATA_SHA256 or file_sha256(base_root / "metadata.json") != base_sha:
        raise ValueError("history cache base metadata identity differs")
    from wnn.all_data_distillation_cache import validate_all_data_distillation_cache

    base_validation = validate_all_data_distillation_cache(
        base_root, data_root, "source", require_families=True
    )
    base_metadata = base_validation["metadata"]
    if base_validation["metadata_sha256"] != base_sha:
        raise ValueError("validated base cache metadata hash differs")
    if history.get("base_verified_files") != base_validation["verified_files"]:
        raise ValueError("history cache family/source inputs differ from the validated base cache")
    if (
        metadata.get("membership_mode") != "all_supplied_train_valid"
        or metadata.get("teacher_training_scope")
        != "mixed_clean_fold_sources_and_all_supplied_history_parent"
        or metadata.get("teacher_prediction_coverage") != "complete_train_valid_target_index"
        or metadata.get("heldout_claim") is not False
        or metadata.get("evaluation_valid_groups") != []
        or tuple(metadata.get("family_order", ())) != FAMILY_ORDER
        or metadata.get("homogeneous_family") != base_metadata.get("homogeneous_family")
    ):
        raise ValueError("history cache must be all-supplied and carry no held-out claim")
    if metadata.get("fit_groups") != base_metadata.get("fit_groups"):
        raise ValueError("history target membership differs from the base cache")
    if metadata.get("split_manifest_sha256") != base_metadata.get("split_manifest_sha256"):
        raise ValueError("history cache clean-split identity differs from the base cache")
    effective = metadata.get("source_coefficients")
    base_coefficients = base_metadata.get("source_coefficients", [])
    expected_effective = [item for item in base_coefficients if item.get("run") not in STRONG_RUNS]
    expected_effective.append(
        {"run": PARENT_RUN, "family": "history_parent", "coefficient": 0.30, "epoch": 3}
    )
    expected_effective.sort(key=lambda item: (item["family"], item["run"]))
    if (
        not isinstance(effective, list)
        or effective != expected_effective
        or history.get("effective_primary_components") != expected_effective
        or abs(math.fsum(float(x["coefficient"]) for x in effective) - 0.8125) > 1e-7
        or abs(float(metadata.get("coefficient_sum", 0)) - 0.8125) > 1e-7
    ):
        raise ValueError("history target coefficients must preserve total primary mass 0.8125")
    formula = history.get("target_formula")
    if formula != {
        "expression": "main_source - 0.15*strong_seed3 - 0.15*strong_seed4 + 0.30*parent_ep3",
        "accumulation": "float64_chunk_order_source_minus_seed3_minus_seed4_plus_parent",
        "output_cast": "one float32 cast after float64 accumulation",
        "family_targets": "unchanged base-cache arrays, no copy",
    }:
        raise ValueError("history cache target formula differs from the frozen H-ADH-R recipe")
    groups = {
        split: list(map(int, base_metadata["fit_groups"][split])) for split in ("train", "valid")
    }
    parent = history.get("parent")
    if not isinstance(parent, dict):
        raise ValueError("history cache lacks its parent ancestry")
    parent_verified = _validate_parent_fit(data_root, groups, parent)
    if parent_verified != history.get("parent_verified"):
        raise ValueError("history cache parent ancestry summary differs")
    exports = history.get("parent_exports")
    if not isinstance(exports, dict):
        raise ValueError("history cache lacks full-order parent prediction exports")
    verified: dict[str, str] = {}
    for split in ("train", "valid"):
        exp = exports.get(split)
        if not isinstance(exp, dict):
            raise ValueError(f"history cache lacks parent {split} export")
        pred_path = _repo_file(data_root, exp.get("path"), f"parent {split} prediction")
        side_path = _repo_file(data_root, exp.get("sidecar_path"), f"parent {split} sidecar")
        export_verified = validate_prediction_export(
            pred_path,
            side_path,
            expected_run=PARENT_RUN,
            expected_split=split,
            expected_epoch=3,
            expected_checkpoint_sha256=parent_verified["checkpoint_sha256"],
            expected_groups=groups[split],
            require_features_only=True,
        )
        for key, value in export_verified.items():
            recorded = exp.get(key)
            if key in {"path", "sidecar_path"}:
                recorded = str(_repo_file(data_root, recorded, f"parent {split} {key}"))
            if recorded != value:
                raise ValueError(f"parent {split} export metadata differs for {key}")
        verified[f"parent_{split}"] = export_verified["sha256"]
        verified[f"parent_{split}_sidecar"] = export_verified["sidecar_sha256"]
    strong_sources = history.get("strong_sources")
    if not isinstance(strong_sources, dict) or set(strong_sources) != set(STRONG_RUNS):
        raise ValueError("history cache must bind both original strong source predictions")
    strong_records = {run: _source_record(base_metadata, run) for run in STRONG_RUNS}
    for run, source_record in strong_records.items():
        split_records = strong_sources.get(run)
        if not isinstance(split_records, dict) or set(split_records) != {"train", "valid"}:
            raise ValueError(f"history cache lacks both splits of strong source {run}")
        for split in ("train", "valid"):
            expected_path, expected_export = _prediction_record(
                data_root, source_record, split, groups[split]
            )
            recorded_export = split_records[split]
            recorded_path = _repo_file(
                data_root, recorded_export.get("path"), f"strong source {run}/{split}"
            )
            if str(recorded_path) != str(expected_path):
                raise ValueError(f"history strong source path differs for {run}/{split}")
            recorded_sidecar = _repo_file(
                data_root,
                recorded_export.get("sidecar_path"),
                f"strong source {run}/{split} sidecar",
            )
            if str(recorded_sidecar) != expected_export["sidecar_path"]:
                raise ValueError(f"history strong source sidecar path differs for {run}/{split}")
            for key in (
                "sha256",
                "sidecar_sha256",
                "shape",
                "dtype",
                "row_groups_sha256",
                "inference_precision",
            ):
                if recorded_export.get(key) != expected_export.get(key):
                    raise ValueError(f"history strong source {run}/{split} differs for {key}")
    files = metadata.get("files")
    if not isinstance(files, dict) or set(files) != {
        "main_history_train",
        "main_history_valid",
    }:
        raise ValueError("history cache must contain only its two new main target arrays")
    for split in ("train", "valid"):
        key = f"main_history_{split}"
        entry = files.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"history cache file manifest is missing {key}")
        verified[key] = _verify_output_array(cache_root, entry, key, groups[split])
    verified.update(
        {f"base/{key}": value for key, value in base_validation["verified_files"].items()}
    )
    return {
        "metadata": metadata,
        "metadata_sha256": file_sha256(metadata_path),
        "verified_files": verified,
        "family_cache_root": str(base_root),
        "family_cache_metadata_sha256": base_sha,
    }


def _source_record(base_metadata: dict[str, Any], run: str) -> dict[str, Any]:
    records = {item.get("run"): item for item in base_metadata.get("sources", [])}
    try:
        record = records[run]
    except KeyError as error:
        raise ValueError(f"base cache does not declare required source {run}") from error
    if record.get("family") != "strong" or record.get("epoch") != 3:
        raise ValueError(
            f"required history subtraction source is not a strong epoch3 teacher: {run}"
        )
    return record


def _prediction_record(
    data_root: Path,
    record: dict[str, Any],
    split: str,
    groups: list[int],
) -> tuple[Path, dict[str, Any]]:
    prediction = record["predictions"][split]
    raw_path = Path(str(prediction["prediction_path"]))
    if raw_path.is_absolute():
        path = raw_path.resolve(strict=True)
        if not path.is_relative_to(data_root):
            raise ValueError("strong teacher prediction path escapes the project root")
    else:
        path = _repo_file(data_root, raw_path, f"strong {split} teacher prediction")
    sidecar_path = path.with_suffix(".json")
    expected_checkpoint = _sha256(record.get("checkpoint_sha256"), "strong teacher checkpoint hash")
    verified = validate_prediction_export(
        path,
        sidecar_path,
        expected_run=str(record["run"]),
        expected_split=split,
        expected_epoch=3,
        expected_checkpoint_sha256=expected_checkpoint,
        expected_groups=groups,
        required_inference_precision=None,
    )
    if verified["sha256"] != prediction.get("prediction_sha256"):
        raise ValueError(f"strong teacher {record['run']}/{split} changed")
    if verified["sidecar_sha256"] != prediction.get("sidecar_sha256"):
        raise ValueError(f"strong teacher {record['run']}/{split} sidecar changed")
    return path, verified


def build_history_cache(
    *,
    data_root: Path,
    base_cache: Path,
    output: Path,
    parent: dict[str, Any],
    parent_prediction_paths: dict[str, Path],
    expected_base_metadata_sha256: str,
    chunk_sequences: int = 16,
    min_free_gb: float = 25.0,
) -> Path:
    """Write only the two new main-history arrays, preserving all base arrays."""
    data_root = data_root.resolve(strict=True)
    base_cache = base_cache.resolve(strict=True)
    output = output.resolve()
    if not output.is_relative_to(data_root):
        raise ValueError("history cache output must remain inside the project root")
    if output == base_cache or base_cache in output.parents:
        raise ValueError("history cache output must not modify the sealed V3 base-cache tree")
    if base_cache.relative_to(data_root).as_posix() != BASE_CACHE_PATH:
        raise ValueError("history cache builder accepts only the frozen V3 base-cache path")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"history cache output parent does not exist: {output.parent}")
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError("refusing to replace an existing history cache or partial output")
    if chunk_sequences < 1 or chunk_sequences > 64:
        raise ValueError("history cache chunk size must be between 1 and 64 sequences")
    expected_base_sha = _sha256(expected_base_metadata_sha256, "expected base cache metadata hash")
    if expected_base_sha != BASE_METADATA_SHA256:
        raise ValueError("expected base cache metadata hash differs from the approved V3 cache")
    from wnn.all_data_distillation_cache import validate_all_data_distillation_cache

    base_validation = validate_all_data_distillation_cache(
        base_cache, data_root, "source", require_families=True
    )
    if base_validation["metadata_sha256"] != expected_base_sha:
        raise ValueError("base cache metadata hash changed")
    base_metadata = base_validation["metadata"]
    if base_metadata.get("membership_mode") != "all_supplied_train_valid":
        raise ValueError("base teacher cache lacks complete all-supplied membership")
    groups = {split: _row_groups(data_root, split) for split in ("train", "valid")}
    if base_metadata.get("fit_groups") != groups:
        raise ValueError("base teacher cache group order differs from target indexes")
    if len(parent_prediction_paths) != 2 or set(parent_prediction_paths) != {"train", "valid"}:
        raise ValueError("both full-order parent train and valid prediction exports are required")
    parent_verified = _validate_parent_fit(data_root, groups, parent)
    parent_exports: dict[str, dict[str, Any]] = {}
    parent_arrays: dict[str, np.ndarray] = {}
    for split in ("train", "valid"):
        pred_path = parent_prediction_paths[split].resolve(strict=True)
        if not pred_path.is_relative_to(data_root):
            raise ValueError("parent prediction export escapes the project root")
        sidecar_path = pred_path.with_suffix(".json")
        export_record = validate_prediction_export(
            pred_path,
            sidecar_path,
            expected_run=PARENT_RUN,
            expected_split=split,
            expected_epoch=3,
            expected_checkpoint_sha256=parent_verified["checkpoint_sha256"],
            expected_groups=groups[split],
            require_features_only=True,
        )
        parent_exports[split] = {
            **export_record,
            "path": str(pred_path.relative_to(data_root)),
            "sidecar_path": str(sidecar_path.relative_to(data_root)),
        }
        parent_arrays[split] = np.load(pred_path, mmap_mode="r", allow_pickle=False)
    strong_records = {run: _source_record(base_metadata, run) for run in STRONG_RUNS}
    strong_arrays: dict[str, dict[str, np.ndarray]] = {run: {} for run in STRONG_RUNS}
    strong_exports: dict[str, dict[str, dict[str, Any]]] = {run: {} for run in STRONG_RUNS}
    for run, record in strong_records.items():
        for split in ("train", "valid"):
            path, verified = _prediction_record(data_root, record, split, groups[split])
            strong_arrays[run][split] = np.load(path, mmap_mode="r", allow_pickle=False)
            strong_exports[run][split] = {
                **verified,
                "path": str(path.relative_to(data_root)),
                "sidecar_path": str(Path(verified["sidecar_path"]).relative_to(data_root)),
                "run": run,
                "checkpoint_sha256": record["checkpoint_sha256"],
            }
    estimated_bytes = sum(len(groups[s]) for s in groups) * SEQUENCE_ROWS * MAIN_WIDTH * 4
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes - estimated_bytes < int(min_free_gb * (1 << 30)):
        raise OSError("history cache output would violate the configured free-disk floor")
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    file_entries: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    for split in ("train", "valid"):
        order = groups[split]
        main_entry = base_metadata["files"][f"main_source_{split}"]
        main_path = (base_cache / str(main_entry["path"])).resolve(strict=True)
        main_array = np.load(main_path, mmap_mode="r", allow_pickle=False)
        shape = (len(order), SEQUENCE_ROWS, MAIN_WIDTH)
        if main_array.shape != shape or main_array.dtype != np.float32:
            raise ValueError(f"base main-source array has a wrong shape or dtype: {split}")
        out = np.lib.format.open_memmap(
            partial / f"main_history_{split}.npy", mode="w+", dtype=np.float32, shape=shape
        )
        for start in range(0, len(order), chunk_sequences):
            stop = min(len(order), start + chunk_sequences)
            chunk = combine_history_chunk(
                np.asarray(main_array[start:stop]),
                np.asarray(strong_arrays[STRONG_RUNS[0]][split][start:stop]),
                np.asarray(strong_arrays[STRONG_RUNS[1]][split][start:stop]),
                np.asarray(parent_arrays[split][start:stop]),
            )
            out[start:stop] = chunk
        out.flush()
        key = f"main_history_{split}"
        path = partial / f"{key}.npy"
        file_entries[key] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "shape": list(shape),
            "dtype": "float32",
            "row_groups": order,
            "row_groups_sha256": row_groups_sha256(order),
        }
        del out, main_array
    effective_sources = [
        {key: item[key] for key in ("run", "family", "coefficient", "epoch")}
        for item in base_metadata["source_coefficients"]
        if item["run"] not in STRONG_RUNS
    ]
    effective_sources.append(
        {"run": PARENT_RUN, "family": "history_parent", "coefficient": 0.30, "epoch": 3}
    )
    effective_sources.sort(key=lambda item: (item["family"], item["run"]))
    metadata = {
        "schema": HISTORY_CACHE_SCHEMA,
        "format": base_metadata["format"],
        "version": base_metadata["version"],
        "variant": "history",
        "membership_mode": "all_supplied_train_valid",
        "teacher_training_scope": "mixed_clean_fold_sources_and_all_supplied_history_parent",
        "teacher_prediction_coverage": "complete_train_valid_target_index",
        "heldout_claim": False,
        "evaluation_valid_groups": [],
        "fit_groups": groups,
        "train_row_groups_sha256": row_groups_sha256(groups["train"]),
        "valid_row_groups_sha256": row_groups_sha256(groups["valid"]),
        "split_manifest": base_metadata["split_manifest"],
        "split_manifest_sha256": base_metadata["split_manifest_sha256"],
        "family_order": list(FAMILY_ORDER),
        "coefficient_sum": 0.8125,
        "source_coefficients": effective_sources,
        "homogeneous_family": base_metadata["homogeneous_family"],
        "epoch_convention": base_metadata["epoch_convention"],
        "files": file_entries,
        "history_swap": {
            "base_cache": {
                "path": str(base_cache.relative_to(data_root)),
                "metadata_sha256": expected_base_sha,
                "source_manifest_path": SOURCE_MANIFEST_PATH,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            },
            "base_verified_files": base_validation["verified_files"],
            "parent": parent,
            "parent_verified": parent_verified,
            "parent_exports": parent_exports,
            "strong_sources": strong_exports,
            "target_formula": {
                "expression": "main_source - 0.15*strong_seed3 - 0.15*strong_seed4 + 0.30*parent_ep3",
                "accumulation": "float64_chunk_order_source_minus_seed3_minus_seed4_plus_parent",
                "output_cast": "one float32 cast after float64 accumulation",
                "family_targets": "unchanged base-cache arrays, no copy",
            },
            "effective_primary_components": effective_sources,
            "builder_source_sha256": file_sha256(Path(__file__).resolve()),
            "build_chunk_sequences": chunk_sequences,
            "created_elapsed_seconds": time.monotonic() - started,
        },
    }
    (partial / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(partial, output)
    return output
