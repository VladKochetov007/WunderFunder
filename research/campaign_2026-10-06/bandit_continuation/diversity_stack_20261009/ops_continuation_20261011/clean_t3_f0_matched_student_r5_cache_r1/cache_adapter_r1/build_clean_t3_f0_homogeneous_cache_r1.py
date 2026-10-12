"""Independently bind clean T3 teacher rows and stream the matched homogeneous cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CACHE_FORMAT = "wnn.heterogeneous-distillation-cache"
CACHE_VERSION = 1
CACHE_PROFILE = "cleanfold_homogeneous_strong_t3_v1"
SEQUENCE_ROWS = 20_000
CHANNELS = 2
COEFFICIENT = np.float32(0.8125)
MEAN_DIVISOR = np.float32(3.0)
SOURCE_COEFFICIENT = 0.1
DEFAULT_BLOCK_SEQUENCES = 8


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    """Hash a file incrementally with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_groups_sha256(groups: Iterable[int]) -> str:
    """Hash an ordered group sequence in stable little-endian int64 form."""
    values = np.asarray(list(map(int, groups)), dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def unique_groups(groups: Iterable[int], label: str) -> list[int]:
    """Normalize group IDs and reject invalid or duplicated members."""
    values = list(map(int, groups))
    if any(group < 0 for group in values):
        raise ValueError(f"{label} contains a negative row-group ID")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate row-group IDs")
    return values


def selected_target_order(
    target_order: Iterable[int], fit_groups: Iterable[int], label: str
) -> list[int]:
    """Preserve the target index order while selecting one source's exact fit set."""
    targets = unique_groups(target_order, f"{label} target index")
    fitting = set(unique_groups(fit_groups, f"{label} fit membership"))
    selected = [group for group in targets if group in fitting]
    if set(selected) != fitting:
        raise ValueError(f"{label} target index omits or adds fit row groups")
    return selected


def row_positions(
    source_order: Iterable[int],
    requested_order: Iterable[int],
    expected_source_membership: Iterable[int],
    label: str,
) -> np.ndarray:
    """Map requested IDs into a producer's ordered array rows for one parquet source."""
    source = unique_groups(source_order, f"{label} producer sidecar")
    requested = unique_groups(requested_order, f"{label} requested output")
    expected = set(unique_groups(expected_source_membership, f"{label} expected membership"))
    if set(source) != expected:
        raise ValueError(f"{label} producer membership differs from its source-qualified contract")
    if not set(requested).issubset(expected):
        raise ValueError(f"{label} requested rows escape the producer's source-qualified membership")
    position = {group: index for index, group in enumerate(source)}
    return np.fromiter((position[group] for group in requested), dtype=np.int64, count=len(requested))


def project_full_valid_rows(
    full_valid_order: Iterable[int],
    fit_valid_groups: Iterable[int],
    evaluation_valid_groups: Iterable[int],
    output_order: Iterable[int],
) -> np.ndarray:
    """Select FIT_VALID rows and reject any evaluation row from full-valid predictions."""
    fit_valid = set(unique_groups(fit_valid_groups, "FIT_VALID"))
    evaluation = set(unique_groups(evaluation_valid_groups, "evaluation valid"))
    if fit_valid & evaluation:
        raise ValueError("FIT_VALID and evaluation valid overlap in the valid.parquet namespace")
    output = unique_groups(output_order, "FIT_VALID output order")
    if set(output) != fit_valid or set(output) & evaluation:
        raise ValueError("projected output must contain exactly FIT_VALID and no evaluation rows")
    return row_positions(
        full_valid_order,
        output,
        fit_valid | evaluation,
        "valid.parquet full-valid projection",
    )


def homogeneous_float32_chunk(
    seed3: np.ndarray, seed4: np.ndarray, seed5: np.ndarray
) -> np.ndarray:
    """Return float32(0.8125 * float32_mean(s3, s4, s5)) without clipping."""
    arrays = (seed3, seed4, seed5)
    if any(array.dtype != np.float32 for array in arrays):
        raise ValueError("teacher chunk inputs must be float32")
    if any(array.shape != seed3.shape for array in arrays):
        raise ValueError("teacher chunk shapes differ")
    if seed3.ndim != 3 or seed3.shape[1:] != (SEQUENCE_ROWS, CHANNELS):
        raise ValueError("teacher chunks must have shape (sequences, 20000, 2)")
    result = np.array(seed3, dtype=np.float32, copy=True)
    np.add(result, seed4, out=result)
    np.add(result, seed5, out=result)
    np.divide(result, MEAN_DIVISOR, out=result)
    np.multiply(result, COEFFICIENT, out=result)
    if not np.isfinite(result).all():
        raise ValueError("homogeneous target chunk contains nonfinite values")
    return result


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    """Read the pinned adapter contract."""
    contract = load_json(path)
    if contract.get("schema") != "wnn.clean-t3-f0-row-group-cache-adapter-contract.r1":
        raise ValueError("unsupported clean T3 cache-adapter contract")
    return contract


def _resolve_lineage_path(root: Path, contract_path: Path, relative: str) -> Path:
    """Resolve a tracked lineage file from the producer root or the adapter worktree."""
    candidate = root / relative
    if candidate.is_file():
        return candidate
    for parent in contract_path.parents:
        if (parent / ".git").exists():
            candidate = parent / relative
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(candidate)


def verify_static_lineage(
    root: Path, contract_path: Path, contract: dict[str, Any]
) -> dict[str, str]:
    """Bind the student recipe, frozen consumer, frozen builder and s3 terminal receipt."""
    lineage = contract["lineage"]
    actual: dict[str, str] = {}
    for path_key, digest_key, label in (
        ("student_recipe_path", "student_recipe_sha256", "student recipe"),
        ("consumer_cache_validator_path", "consumer_cache_validator_sha256", "cache consumer"),
        ("frozen_legacy_builder_path", "frozen_legacy_builder_sha256", "frozen builder"),
        ("terminal_s3_export_receipt_path", "terminal_s3_export_receipt_sha256", "s3 terminal receipt"),
    ):
        path = _resolve_lineage_path(root, contract_path, lineage[path_key])
        digest = sha256_file(path)
        if digest != lineage[digest_key]:
            raise ValueError(f"{label} hash differs from the reviewed contract")
        actual[label] = digest
    return actual


def load_split_membership(
    root: Path, contract: dict[str, Any]
) -> tuple[dict[str, list[int]], list[int], dict[str, Any]]:
    """Read source-qualified F0 fit/evaluation groups and enforce exact counts."""
    membership = contract["source_qualified_membership"]
    path = root / contract["lineage"]["split_manifest_path"]
    digest = sha256_file(path)
    if digest != contract["lineage"]["split_manifest_sha256"]:
        raise ValueError("F0 split manifest hash differs")
    split = load_json(path)
    if split.get("fold") != membership["fold"]:
        raise ValueError("F0 split fold differs")
    fit = {
        "train": unique_groups(split["fit"]["train"], "train.parquet FIT_TRAIN"),
        "valid": unique_groups(split["fit"]["valid"], "valid.parquet FIT_VALID"),
    }
    evaluation_valid = unique_groups(
        split["evaluation"]["valid"], "valid.parquet evaluation"
    )
    if len(fit["train"]) != membership["fit_train_count"]:
        raise ValueError("F0 FIT_TRAIN count differs")
    if len(fit["valid"]) != membership["fit_valid_count"]:
        raise ValueError("F0 FIT_VALID count differs")
    if len(evaluation_valid) != membership["evaluation_valid_count"]:
        raise ValueError("F0 evaluation-valid count differs")
    if set(fit["valid"]) & set(evaluation_valid):
        raise ValueError("valid.parquet FIT_VALID overlaps evaluation membership")
    full_valid = set(range(int(membership["full_valid_prediction_count"])))
    if set(fit["valid"]) | set(evaluation_valid) != full_valid:
        raise ValueError("FIT_VALID and evaluation groups do not exactly cover full valid.parquet")
    return fit, evaluation_valid, {"path": str(path), "sha256": digest, "split": split}


def load_group_target_order(root: Path, source_split: str) -> list[int]:
    """Read only row_group from a source's target index; do not load its labels."""
    if source_split not in ("train", "valid"):
        raise ValueError(f"unknown source split: {source_split}")
    path = root / "cache" / f"{source_split}_targets.npz"
    with np.load(path, allow_pickle=False) as archive:
        if "row_group" not in archive.files:
            raise ValueError(f"target index has no row_group member: {path}")
        groups = np.asarray(archive["row_group"], dtype=np.int64)
    return unique_groups(groups.tolist(), f"{source_split} target index")


def _validate_array(
    path: Path,
    expected_shape: list[int],
    expected_dtype: str,
    expected_bytes: int,
    expected_sha256: str,
    *,
    hash_array: bool,
) -> dict[str, Any]:
    """Check a real producer NPY header and optionally stream its content hash."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"producer NPY byte count differs: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != expected_shape or str(array.dtype) != expected_dtype:
        raise ValueError(f"producer NPY header differs: {path}")
    digest = sha256_file(path) if hash_array else expected_sha256
    if digest != expected_sha256:
        raise ValueError(f"producer array SHA256 differs from its sidecar: {path}")
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "inode": path.stat().st_ino,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "hash_recomputed": hash_array,
    }


def _validate_teacher_provenance(
    root: Path,
    teacher: dict[str, Any],
    fit: dict[str, list[int]],
    evaluation_valid: list[int],
    split_sha256: str,
) -> dict[str, Any]:
    """Verify each seed's own ordered provenance and source-qualified member sets."""
    run = teacher["run"]
    provenance_path = root / "artifacts" / "runs" / run / "provenance.json"
    provenance_sha = sha256_file(provenance_path)
    if provenance_sha != teacher["provenance_sha256"]:
        raise ValueError(f"teacher provenance hash differs for {run}")
    provenance = load_json(provenance_path)
    args = provenance.get("args", {})
    if (
        provenance.get("seed") != teacher["seed"]
        or args.get("seed") != teacher["seed"]
        or provenance.get("split_manifest_sha256") != split_sha256
    ):
        raise ValueError(f"teacher seed or split binding differs for {run}")
    checkpoint_sha = provenance.get("checkpoint_sha256", {}).get("model_ep2.pt")
    checkpoint_path = root / "artifacts" / "runs" / run / "model_ep2.pt"
    if checkpoint_sha != teacher["checkpoint_sha256"] or sha256_file(checkpoint_path) != checkpoint_sha:
        raise ValueError(f"teacher epoch-2 checkpoint hash differs for {run}")
    memberships: dict[str, Any] = {}
    recorded = provenance.get("fit_row_groups", {})
    identity = provenance.get("fit_membership_identity", {})
    for source_split in ("train", "valid"):
        order = unique_groups(recorded.get(source_split, []), f"{run} {source_split} provenance")
        expected = fit[source_split]
        if set(order) != set(expected):
            raise ValueError(f"{run} {source_split} source-qualified fit member set differs")
        expected_identity = teacher["fit_membership_identity"][source_split]
        actual_hash = ordered_groups_sha256(order)
        if (
            len(order) != expected_identity["count"]
            or identity.get(source_split, {}).get("count") != len(order)
            or actual_hash != expected_identity["ordered_int64_le_sha256"]
            or identity.get(source_split, {}).get("ordered_int64_le_sha256") != actual_hash
        ):
            raise ValueError(f"{run} {source_split} ordered provenance fingerprint differs")
        memberships[source_split] = {
            "count": len(order),
            "ordered_sha256": actual_hash,
            "member_set_matches_f0": True,
        }
    if set(fit["valid"]) & set(evaluation_valid):
        raise ValueError(f"{run} valid fit membership overlaps valid evaluation groups")
    return {
        "run": run,
        "seed": teacher["seed"],
        "provenance_path": str(provenance_path),
        "provenance_sha256": provenance_sha,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "fit_memberships": memberships,
    }


def _validate_prediction(
    root: Path,
    teacher: dict[str, Any],
    source_split: str,
    fit_groups: list[int],
    evaluation_valid: list[int],
    split_sha256: str,
    *,
    hash_array: bool,
) -> dict[str, Any] | None:
    """Validate producer scope, IDs, NPY shape and the actual sidecar/array binding."""
    binding = teacher[f"fit_{source_split}_prediction"] if source_split == "train" else teacher[
        "fit_valid_full_valid_prediction"
    ]
    path = root / binding["path"]
    sidecar_path = root / binding["sidecar_path"]
    partial_path = path.with_name(path.name + ".partial")
    if not path.exists() and not sidecar_path.exists():
        return None
    if not path.is_file() or not sidecar_path.is_file() or partial_path.exists():
        raise ValueError(f"{teacher['run']} {source_split} prediction is incomplete")
    sidecar_sha = sha256_file(sidecar_path)
    pinned_sidecar = binding.get("sidecar_sha256")
    if pinned_sidecar and sidecar_sha != pinned_sidecar:
        raise ValueError(f"{teacher['run']} {source_split} sidecar hash differs")
    sidecar = load_json(sidecar_path)
    expected_scope = {
        "run": teacher["run"],
        "split": "train" if source_split == "train" else "valid",
        "epoch": 2,
        "checkpoint_sha256": teacher["checkpoint_sha256"],
        "split_manifest_sha256": split_sha256,
        "campaign_split_manifest_sha256": split_sha256,
        "inference_precision": "fp32",
    }
    if any(sidecar.get(key) != value for key, value in expected_scope.items()):
        raise ValueError(f"{teacher['run']} {source_split} sidecar provenance differs")
    expected_campaign_scope = "train" if source_split == "train" else "full-valid-feature-only"
    if sidecar.get("campaign_fit_scope") != expected_campaign_scope:
        raise ValueError(f"{teacher['run']} {source_split} campaign scope differs")
    groups = unique_groups(sidecar.get("row_groups", []), f"{teacher['run']} {source_split} sidecar")
    if source_split == "train":
        expected_members = set(fit_groups)
        required_order_hash = binding.get("ordered_row_groups_sha256")
    else:
        expected_members = set(fit_groups) | set(evaluation_valid)
        required_order_hash = binding.get("ordered_row_groups_sha256")
    if set(groups) != expected_members:
        raise ValueError(f"{teacher['run']} {source_split} sidecar member set differs")
    group_hash = ordered_groups_sha256(groups)
    if required_order_hash and group_hash != required_order_hash:
        raise ValueError(f"{teacher['run']} {source_split} ordered sidecar fingerprint differs")
    prediction_sha = str(sidecar.get("prediction_sha256", ""))
    expected_prediction_sha = binding.get("prediction_sha256")
    if expected_prediction_sha and prediction_sha != expected_prediction_sha:
        raise ValueError(f"{teacher['run']} {source_split} sidecar prediction SHA differs")
    array_record = _validate_array(
        path,
        binding["expected_shape"],
        binding["expected_dtype"],
        binding["expected_file_bytes"],
        prediction_sha,
        hash_array=hash_array,
    )
    return {
        "source_split": source_split,
        "scope": expected_campaign_scope,
        "path": str(path),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sidecar_sha,
        "prediction_sha256": prediction_sha,
        "ordered_row_groups_sha256": group_hash,
        "row_group_count": len(groups),
        "row_groups": groups,
        "array": array_record,
        "ready": True,
    }


def inspect_inputs(
    root: Path,
    contract_path: Path,
    *,
    require_train: bool = False,
    hash_arrays: bool = False,
) -> dict[str, Any]:
    """Inspect all current teacher sidecars while retaining absent train exports as pending."""
    root = root.resolve(strict=True)
    contract_path = contract_path.resolve(strict=True)
    contract = load_contract(contract_path)
    lineage = verify_static_lineage(root, contract_path, contract)
    fit, evaluation_valid, split_record = load_split_membership(root, contract)
    target_orders = load_target_orders(root)
    output_groups = output_groups_from_target_orders(target_orders, fit, evaluation_valid)
    teacher_records: list[dict[str, Any]] = []
    pending_train: list[str] = []
    for teacher in contract["teachers"]:
        provenance = _validate_teacher_provenance(
            root,
            teacher,
            fit,
            evaluation_valid,
            contract["lineage"]["split_manifest_sha256"],
        )
        predictions: dict[str, Any] = {}
        for source_split in ("train", "valid"):
            record = _validate_prediction(
                root,
                teacher,
                source_split,
                fit[source_split],
                evaluation_valid,
                contract["lineage"]["split_manifest_sha256"],
                hash_array=hash_arrays,
            )
            if record is None and source_split == "train":
                pending_train.append(teacher["run"])
                predictions[source_split] = {"ready": False, "status": "PENDING_TRAIN_EXPORT"}
            elif record is None:
                raise FileNotFoundError(f"{teacher['run']} full-valid sidecar is missing")
            else:
                predictions[source_split] = record
        teacher_records.append({"provenance": provenance, "predictions": predictions})
    if require_train and pending_train:
        raise FileNotFoundError(f"FIT_TRAIN predictions remain pending: {pending_train}")
    ready_train = [
        record["provenance"]["run"]
        for record in teacher_records
        if record["predictions"]["train"].get("ready")
    ]
    status = (
        "PASS_ALL_TEACHER_INPUTS_READY"
        if not pending_train
        else "PARTIAL_INPUTS_VALIDATED_AWAITING_TRAIN_EXPORTS"
    )
    return {
        "schema": "wnn.clean-t3-f0-cache-adapter-input-validation.r1",
        "status": status,
        "root": str(root),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "static_lineage_hashes": lineage,
        "split": {
            "path": split_record["path"],
            "sha256": split_record["sha256"],
            "fit_train_source": "datasets/train.parquet",
            "fit_train_count": len(fit["train"]),
            "fit_train_set_sha256": ordered_groups_sha256(sorted(fit["train"])),
            "fit_valid_source": "datasets/valid.parquet",
            "fit_valid_count": len(fit["valid"]),
            "evaluation_valid_count": len(evaluation_valid),
            "fit_valid_evaluation_overlap": 0,
        },
        "target_indices": {
            name: {
                "path": str(root / "cache" / f"{name}_targets.npz"),
                "member_read": "row_group only; target-label members are not loaded",
                "source_group_count": len(target_orders[name]),
                "source_order_sha256": ordered_groups_sha256(target_orders[name]),
                "selected_fit_count": len(output_groups[name]),
                "selected_fit_order_sha256": ordered_groups_sha256(output_groups[name]),
            }
            for name in ("train", "valid")
        },
        "teacher_records": teacher_records,
        "fit_train_predictions_ready": ready_train,
        "fit_train_predictions_pending": pending_train,
        "array_hashing": "streamed" if hash_arrays else "producer sidecar and pinned receipt only",
        "cache_materialized": False,
        "student_fit_launched": False,
    }


def load_target_orders(root: Path) -> dict[str, list[int]]:
    """Read source-specific target-index row groups without loading target labels."""
    return {
        "train": load_group_target_order(root, "train"),
        "valid": load_group_target_order(root, "valid"),
    }


def output_groups_from_target_orders(
    target_orders: dict[str, list[int]], fit: dict[str, list[int]], evaluation_valid: list[int]
) -> dict[str, list[int]]:
    """Select exact fit rows from source-local target-index order."""
    train_groups = selected_target_order(target_orders["train"], fit["train"], "train.parquet")
    valid_groups = selected_target_order(target_orders["valid"], fit["valid"], "valid.parquet")
    if set(valid_groups) & set(evaluation_valid):
        raise ValueError("valid output order contains an evaluation row")
    return {"train": train_groups, "valid": valid_groups}


def _source_row_map(
    source_record: dict[str, Any], split_name: str, output_groups: list[int], fit: dict[str, list[int]], evaluation: list[int]
) -> np.ndarray:
    """Select output row positions by IDs within one source parquet."""
    groups = source_record["predictions"][split_name]["row_groups"]
    if split_name == "train":
        return row_positions(groups, output_groups, fit["train"], "train.parquet FIT_TRAIN")
    return project_full_valid_rows(groups, fit["valid"], evaluation, output_groups)


def _write_output(
    output_path: Path,
    target_groups: dict[str, list[int]],
    positions: list[dict[str, np.ndarray]],
    arrays: list[dict[str, dict[str, np.ndarray]]],
    block_sequences: int,
) -> dict[str, dict[str, Any]]:
    """Write two target arrays through output memmaps in bounded sequence chunks."""
    partial = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite cache output or partial: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    files: dict[str, dict[str, Any]] = {}
    try:
        for split_name in ("train", "valid"):
            groups = target_groups[split_name]
            key = f"main_homogeneous_{split_name}"
            output_file = partial / f"{key}.npy"
            output = np.lib.format.open_memmap(
                output_file,
                mode="w+",
                dtype=np.float32,
                shape=(len(groups), SEQUENCE_ROWS, CHANNELS),
            )
            for start in range(0, len(groups), block_sequences):
                stop = min(start + block_sequences, len(groups))
                sources = []
                for teacher_index in range(3):
                    source_array = arrays[teacher_index][split_name]["array"]
                    row_indices = positions[teacher_index][split_name][start:stop]
                    block = np.asarray(source_array[row_indices], dtype=np.float32)
                    if not np.isfinite(block).all():
                        raise ValueError(
                            f"nonfinite teacher prediction in {split_name} rows {start}:{stop}"
                        )
                    sources.append(block)
                output[start:stop] = homogeneous_float32_chunk(*sources)
            output.flush()
            del output
            written = np.load(output_file, mmap_mode="r", allow_pickle=False)
            files[key] = {
                "path": output_file.name,
                "sha256": sha256_file(output_file),
                "bytes": output_file.stat().st_size,
                "shape": list(written.shape),
                "dtype": str(written.dtype),
                "row_groups": groups,
                "row_groups_sha256": ordered_groups_sha256(groups),
            }
            del written
        return files
    except Exception:
        # Preserve the partial cache and any written arrays for a diagnosable retry.
        raise


def _metadata(
    contract: dict[str, Any],
    contract_path: Path,
    split_record: dict[str, Any],
    fit: dict[str, list[int]],
    evaluation_valid: list[int],
    teacher_records: list[dict[str, Any]],
    files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the narrow cleanfold T3 metadata accepted by the frozen consumer."""
    coefficients = [
        {
            "family": "strong",
            "run": teacher["provenance"]["run"],
            "epoch": 2,
            "coefficient": SOURCE_COEFFICIENT,
        }
        for teacher in teacher_records
    ]
    source_records = []
    for teacher in teacher_records:
        provenance = teacher["provenance"]
        predictions = teacher["predictions"]
        source_records.append(
            {
                "family": "strong",
                "run": provenance["run"],
                "seed": provenance["seed"],
                "epoch": 2,
                "coefficient": SOURCE_COEFFICIENT,
                "checkpoint_path": provenance["checkpoint_path"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "provenance_path": provenance["provenance_path"],
                "provenance_sha256": provenance["provenance_sha256"],
                "fit_group_memberships": provenance["fit_memberships"],
                "predictions": {
                    name: {
                        key: value
                        for key, value in record.items()
                        if key not in ("row_groups", "array")
                    }
                    for name, record in predictions.items()
                    if record.get("ready")
                },
            }
        )
    return {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "cache_profile": CACHE_PROFILE,
        "fold": 0,
        "split_manifest": contract["lineage"]["split_manifest_path"],
        "split_manifest_sha256": split_record["sha256"],
        "fit_groups": {name: fit[name] for name in ("train", "valid")},
        "evaluation_valid_groups": evaluation_valid,
        "family_order": ["strong"],
        "source_coefficients": coefficients,
        "coefficient_sum": math.fsum(item["coefficient"] for item in coefficients),
        "homogeneous_family": "strong",
        "homogeneous_scale": 0.8125,
        "output_profile": "homogeneous-only",
        "stored_targets": ["main_homogeneous"],
        "target_contract": {
            "formula": "float32(0.8125 * float32_mean(three_epoch2_predictions))",
            "teacher_epoch": 2,
            "teacher_mean": "unweighted arithmetic mean in float32",
            "coefficient_multiplier": 0.8125,
            "clipping": "none during cache construction",
        },
        "student_contract": {
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
        },
        "sources": source_records,
        "files": files,
        "adapter_contract": str(contract_path),
        "adapter_contract_sha256": sha256_file(contract_path),
        "source_qualification": {
            "train": "datasets/train.parquet",
            "valid": "datasets/valid.parquet",
            "teacher_fit_train_order_is_seed_specific": True,
            "prediction_arrays_joined_by_row_group_id": True,
            "evaluation_valid_groups_in_cache": 0,
        },
    }


def materialize_cache(
    root: Path,
    contract_path: Path,
    output_path: Path,
    *,
    root_release_path: Path | None = None,
    block_sequences: int = DEFAULT_BLOCK_SEQUENCES,
    hash_inputs: bool = True,
) -> Path:
    """Create the two clean-F0 cache arrays; call only after root release."""
    if not 1 <= block_sequences <= 32:
        raise ValueError("block_sequences must be in [1, 32]")
    root = root.resolve(strict=True)
    contract_path = contract_path.resolve(strict=True)
    contract = load_contract(contract_path)
    if root_release_path is None:
        raise RuntimeError("cache materialization is pending an explicit root release")
    release = load_json(root_release_path.resolve(strict=True))
    if (
        release.get("schema") != "wnn.clean-t3-f0-cache-adapter-release.r1"
        or release.get("cache_materialization_authorized") is not True
        or release.get("cache_adapter_contract_sha256") != sha256_file(contract_path)
        or release.get("output_cache_root") != contract["output"]["cache_root"]
    ):
        raise ValueError("root release receipt does not authorize this exact cache contract/output")
    if not hash_inputs:
        raise ValueError("materialization requires streamed input SHA256 verification")
    fit, evaluation, split_record = load_split_membership(root, contract)
    inspected = inspect_inputs(root, contract_path, require_train=True, hash_arrays=hash_inputs)
    target_orders = load_target_orders(root)
    target_groups = output_groups_from_target_orders(target_orders, fit, evaluation)
    teacher_records = inspected["teacher_records"]
    if len(teacher_records) != 3:
        raise ValueError("exactly three seed-specific teachers are required")

    opened: list[dict[str, dict[str, Any]]] = []
    positions: list[dict[str, np.ndarray]] = []
    for record in teacher_records:
        opened_by_split: dict[str, dict[str, Any]] = {}
        positions_by_split: dict[str, np.ndarray] = {}
        for source_split in ("train", "valid"):
            binding = record["predictions"][source_split]
            input_stat = Path(binding["path"]).stat()
            array_record = binding["array"]
            if (
                input_stat.st_size != array_record["bytes"]
                or input_stat.st_mtime_ns != array_record["mtime_ns"]
                or input_stat.st_ino != array_record["inode"]
            ):
                raise ValueError("teacher prediction changed after its streamed SHA256 validation")
            array = np.load(binding["path"], mmap_mode="r", allow_pickle=False)
            opened_by_split[source_split] = {"record": binding, "array": array}
            if source_split == "train":
                positions_by_split[source_split] = row_positions(
                    binding["row_groups"],
                    target_groups[source_split],
                    fit["train"],
                    "train.parquet FIT_TRAIN",
                )
            else:
                positions_by_split[source_split] = project_full_valid_rows(
                    binding["row_groups"],
                    fit["valid"],
                    evaluation,
                    target_groups[source_split],
                )
        opened.append(opened_by_split)
        positions.append(positions_by_split)

    output_path = output_path.resolve()
    expected_output = root / contract["output"]["cache_root"]
    if output_path != expected_output.resolve():
        raise ValueError(f"output path differs from unique contract root: {expected_output}")
    total_bytes = sum(
        int(item["expected_bytes_including_npy_header"])
        for item in contract["output"]["files"].values()
    )
    usage = shutil.disk_usage(output_path.parent)
    if usage.free < total_bytes:
        raise OSError(f"cache needs {total_bytes} bytes, only {usage.free} are free")
    files = _write_output(output_path, target_groups, positions, opened, block_sequences)
    for key, expected in contract["output"]["files"].items():
        entry = files[key.removesuffix(".npy")]
        if entry["bytes"] != expected["expected_bytes_including_npy_header"]:
            raise ValueError(f"completed output file size differs for {key}")
        if entry["shape"] != expected["shape"] or entry["dtype"] != expected["dtype"]:
            raise ValueError(f"completed output header differs for {key}")
    metadata = _metadata(
        contract,
        contract_path,
        split_record,
        fit,
        evaluation,
        teacher_records,
        files,
    )
    partial = output_path.with_name(output_path.name + ".partial")
    (partial / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(partial, output_path)
    return output_path


def main() -> None:
    """Inspect producer artifacts or, after root release, build the pending cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--build", action="store_true", help="materialize cache; requires contract release")
    parser.add_argument("--root-release-receipt", type=Path)
    parser.add_argument("--block-sequences", type=int, default=DEFAULT_BLOCK_SEQUENCES)
    parser.add_argument("--skip-array-hash", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    contract = load_contract(args.contract.resolve(strict=True))
    if args.build:
        if args.skip_array_hash:
            parser.error("--build always recomputes every producer-array SHA256")
        output = materialize_cache(
            root,
            args.contract,
            root / contract["output"]["cache_root"],
            root_release_path=args.root_release_receipt,
            block_sequences=args.block_sequences,
            hash_inputs=True,
        )
        result: dict[str, Any] = {"status": "CACHE_MATERIALIZED", "cache_root": str(output)}
    else:
        result = inspect_inputs(
            root,
            args.contract,
            require_train=False,
            hash_arrays=not args.skip_array_hash,
        )
    if args.receipt:
        receipt = args.receipt.resolve()
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "teacher_records"}, indent=2))


if __name__ == "__main__":
    main()
