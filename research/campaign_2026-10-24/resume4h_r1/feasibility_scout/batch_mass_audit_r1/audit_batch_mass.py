"""Describe the batch-normalized fitting measure from transported suffix arrays.

This audit reads only the pinned F0/F1 fit suffixes after a root admission. It
does not import the trainer, model, Torch, or evaluation caches.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np

ROWS_PER_SEQUENCE = 20_000
BATCH_ROWS = 8_192
SUFFIX_WIDTH = 5
SUFFIX_DTYPE = np.dtype("<f4")
READ_BATCHES = 16
BATCH_DENOMINATOR_EPSILON = 1e-8


def read_json(path: Path, *, maximum_bytes: int = 30_000_000) -> dict[str, object]:
    """Read one bounded JSON metadata file and require an object root."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"metadata file size is invalid: {path} ({size} bytes)")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata root must be a JSON object: {path}")
    return cast(dict[str, object], value)


def sha256_file(path: Path) -> str:
    """Hash a source file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def load_array_header(source: BinaryIO) -> tuple[bytes, tuple[int, ...], str, bool]:
    """Read only an NPY v1 header and return bytes, shape, dtype, and order."""
    prefix = source.read(8)
    if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY" or prefix[6:8] != b"\x01\x00":
        raise ValueError("suffix must use the pinned NPY v1.0 format")
    length_bytes = source.read(2)
    if len(length_bytes) != 2:
        raise ValueError("truncated NPY header length")
    header_length = struct.unpack("<H", length_bytes)[0]
    header_text = source.read(header_length)
    if len(header_text) != header_length:
        raise ValueError("truncated NPY header")
    header = ast.literal_eval(header_text.decode("latin1").strip())
    if not isinstance(header, dict):
        raise ValueError("NPY header payload is not a dictionary")
    shape = tuple(int(value) for value in header["shape"])
    return (
        prefix + length_bytes + header_text,
        shape,
        str(header["descr"]),
        bool(header["fortran_order"]),
    )


def row_refs_sha256(row_refs: list[dict[str, object]]) -> str:
    """Reproduce the frozen cache's ordered source-qualified membership digest."""
    normalized = [
        {"split": str(ref["split"]), "row_group": int(ref["row_group"])} for ref in row_refs
    ]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def selected_members_sha256(train_groups: list[int], valid_groups: list[int]) -> str:
    """Reproduce the frozen component-subset source-member digest."""
    members = sorted([("train", int(group)) for group in train_groups])
    members.extend(("valid", int(group)) for group in valid_groups)
    members.sort()
    payload = json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def selected_components_sha256(components: list[int]) -> str:
    """Reproduce the frozen little-endian int64 component digest."""
    payload = b"".join(struct.pack("<q", int(component)) for component in components)
    return hashlib.sha256(payload).hexdigest()


def integer_list(value: object, name: str) -> list[int]:
    """Return a strict nonnegative integer list, excluding booleans."""
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{name} must be a list of nonnegative integers")
    return cast(list[int], value)


def evaluation_groups_sha256(groups: list[int]) -> str:
    """Reproduce the canonical sorted little-endian evaluation-group digest."""
    payload = b"".join(struct.pack("<q", int(group)) for group in groups)
    return hashlib.sha256(payload).hexdigest()


def batch_statistics(
    suffix: np.ndarray, component_indices: np.ndarray, component_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target masses, parent squared-error numerators, and component mass."""
    if suffix.ndim != 2 or suffix.shape[1] != SUFFIX_WIDTH:
        raise ValueError("suffix batch must have shape (rows, 5)")
    if component_indices.shape != (len(suffix),):
        raise ValueError("component indices must align with suffix rows")
    if not np.isfinite(suffix).all() or (suffix[:, 2] < 0).any():
        raise ValueError("suffix contains nonfinite values or negative row weights")
    target = np.clip(suffix[:, :2], -2.0, 2.0).astype(np.float64)
    row_weight = suffix[:, 2].astype(np.float64)
    parent = suffix[:, 3:5].astype(np.float64)
    weights = np.abs(target) * row_weight[:, None]
    numerators = np.sum(weights * np.square(parent - target), axis=0, dtype=np.float64)
    masses = np.sum(weights, axis=0, dtype=np.float64)
    component_mass = np.bincount(
        component_indices, weights=np.sum(weights, axis=1), minlength=component_count
    ).astype(np.float64, copy=False)
    return masses, numerators, component_mass


def component_rows(
    first_row: int, row_count: int, sequence_component_indices: list[int]
) -> np.ndarray:
    """Expand sequence component IDs over one original row-batch interval."""
    result = np.empty(row_count, dtype=np.int32)
    filled = 0
    position = first_row
    while filled < row_count:
        sequence_index, row_in_sequence = divmod(position, ROWS_PER_SEQUENCE)
        take = min(row_count - filled, ROWS_PER_SEQUENCE - row_in_sequence)
        result[filled : filled + take] = sequence_component_indices[sequence_index]
        filled += take
        position += take
    return result


def quantiles(values: list[float | None]) -> dict[str, float | int | None]:
    """Summarize finite scalar values without interpreting zero-mass batches."""
    selected = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(selected):
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    q05, q25, q50, q75, q95 = np.quantile(selected, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(len(selected)),
        "min": float(selected.min()),
        "p05": float(q05),
        "p25": float(q25),
        "median": float(q50),
        "p75": float(q75),
        "p95": float(q95),
        "max": float(selected.max()),
        "mean": float(selected.mean()),
    }


def concentration(shares: np.ndarray) -> dict[str, float | int]:
    """Summarize component shares under one declared measure."""
    ordered = np.sort(shares)[::-1]
    hhi = float(np.dot(shares, shares))
    top_count = max(1, int(np.ceil(len(shares) * 0.10)))
    return {
        "component_count": int(len(shares)),
        "hhi": hhi,
        "effective_component_count": 1.0 / hhi if hhi > 0 else 0.0,
        "largest_component_share": float(ordered[0]) if len(ordered) else 0.0,
        "top_10pct_component_share": float(ordered[:top_count].sum()),
    }


def verify_sources(manifest: dict[str, object], repository: Path) -> None:
    """Verify small source and metadata pins; payloads are checked during scoring."""
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("source manifest files must be a list")
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("source manifest record must be an object")
        record = cast(dict[str, object], item)
        path = resolve_path(repository, str(record["path"]))
        if record.get("kind") == "suffix_payload":
            continue
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"source size changed: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"source hash changed: {path}")


def resolve_path(repository: Path, text: str) -> Path:
    """Resolve one pinned source path against the repository root if needed."""
    value = Path(text)
    return value if value.is_absolute() else (repository / value).resolve()


def validate_admission(
    admission: dict[str, object], recipe_sha: str, manifest_sha: str, recipe: dict[str, object]
) -> None:
    """Require root authorization bound to this frozen source and resource scope."""
    limits = cast(dict[str, object], recipe["resource_limits"])
    if (
        admission.get("schema") != "wnn.f0-f1-batch-mass-admission.v1"
        or admission.get("status") != "admitted"
        or admission.get("root_authorized") is not True
        or admission.get("recipe_sha256") != recipe_sha
        or admission.get("source_manifest_sha256") != manifest_sha
        or admission.get("folds") != [0, 1]
        or admission.get("max_memory_bytes") != limits["memory_bytes"]
        or admission.get("max_cpu_seconds") != limits["cpu_seconds"]
        or admission.get("cpu_cores") != limits["cpu_cores"]
        or admission.get("evaluation_labels_read") is not False
    ):
        raise PermissionError("root admission does not match the frozen audit recipe")


def validate_fold_metadata(
    fold: dict[str, object], repository: Path, transport: dict[str, object]
) -> tuple[list[dict[str, object]], list[int], list[int], dict[str, int]]:
    """Validate suffix, source sidecar, selected members, split mapping, and disjointness."""
    sidecar_path = resolve_path(repository, str(fold["sidecar_path"]))
    subset_path = resolve_path(repository, str(fold["subset_path"]))
    split_path = resolve_path(repository, str(fold["split_path"]))
    if (
        sha256_file(sidecar_path) != fold["sidecar_sha256"]
        or sha256_file(subset_path) != fold["subset_sha256"]
        or sha256_file(split_path) != fold["split_manifest_sha256"]
    ):
        raise ValueError("source-qualified fit metadata differs from frozen hashes")
    cache = read_json(sidecar_path)
    subset = read_json(subset_path)
    split = read_json(split_path)
    expected = cast(dict[str, object], fold["suffix"])
    if (
        cache.get("schema") != "wnn.online-relationships.sequence-cache.v1"
        or cache.get("scope") != "in_sample_fit"
        or cache.get("scope_label") != "in_sample_fit"
        or cache.get("prediction_scope") != "in_sample_fit"
        or cache.get("complete_membership") is not True
        or cache.get("fold") != fold["fold"]
        or cache.get("seed") != fold["seed"]
        or cache.get("recipe_sha256") != fold["original_recipe_sha256"]
        or cache.get("recipe_id") != fold["experiment_id"]
        or cache.get("shape") != [expected["shape"][0], ROWS_PER_SEQUENCE, 117]
        or cache.get("dtype") != "<f4"
        or cache.get("data_sha256") != fold["original_full_cache_sha256"]
        or cache.get("membership_sha256") != fold["membership_sha256"]
        or cache.get("subset_members_sha256") != fold["selected_members_sha256"]
        or cache.get("split_manifest_sha256") != fold["split_manifest_sha256"]
        or cache.get("subset_manifest_sha256") != fold["subset_sha256"]
        or cache.get("read_contract") != {"targets": True, "scored_mask": True}
        or cache.get("data_file") != "fit_sequences.npy"
        or cache.get("split_manifest") != str(split_path)
        or cache.get("subset_manifest") != str(fold["subset_path"])
        or cache.get("layout") != fold["cache_layout"]
        or cache.get("row_weight_contract") != fold["row_weight_contract"]
        or cache.get("row_weight_batch_semantics") != fold["row_weight_batch_semantics"]
        or cache.get("parent") != fold["parent_metadata"]
        or cache.get("propensity") != fold["propensity_metadata"]
        or cache.get("source_datasets") != fold["source_datasets_metadata"]
        or cache.get("source_sha256") != fold["producer_source_sha256"]
        or cache.get("zero_columns") != []
        or expected.get("source_columns_zero_based") != [112, 117]
        or split.get("fold") != fold["fold"]
        or subset.get("split_manifest_sha256") != fold["split_manifest_sha256"]
        or subset.get("experiment_id") != fold["experiment_id"]
        or subset.get("selection_seed") != fold["seed"]
    ):
        raise ValueError(f"fold {fold['fold']} original cache sidecar contract changed")
    refs_raw = cache.get("row_refs")
    if not isinstance(refs_raw, list) or len(refs_raw) != expected["shape"][0]:
        raise ValueError("cache sidecar lacks the exact complete row-reference list")
    refs = [cast(dict[str, object], ref) for ref in refs_raw if isinstance(ref, dict)]
    if len(refs) != len(refs_raw) or row_refs_sha256(refs) != fold["membership_sha256"]:
        raise ValueError("ordered cache membership digest changed")

    selected_groups = cast(dict[str, object], subset["selected_groups"])
    train_groups = integer_list(selected_groups.get("train"), "selected train groups")
    valid_groups = integer_list(selected_groups.get("valid"), "selected valid groups")
    if subset.get("selected_group_counts") != fold["source_group_counts"]:
        raise ValueError("fit subset per-source group counts differ from the frozen recipe")
    split_fit = cast(dict[str, object], split["fit"])
    allowed_groups = {
        source_split: set(integer_list(split_fit.get(source_split), f"{source_split} fit groups"))
        for source_split in ("train", "valid")
    }
    for source_split, groups in (("train", train_groups), ("valid", valid_groups)):
        if len(groups) != len(set(groups)) or not set(groups).issubset(
            allowed_groups[source_split]
        ):
            raise ValueError(f"selected {source_split} groups are duplicated or outside fit scope")
    if selected_members_sha256(train_groups, valid_groups) != fold["selected_members_sha256"]:
        raise ValueError("fit subset member digest changed")
    expected_refs = [{"split": "train", "row_group": int(group)} for group in train_groups] + [
        {"split": "valid", "row_group": int(group)} for group in valid_groups
    ]
    if refs != expected_refs:
        raise ValueError(
            "cache source order differs from the original train-then-valid fitter stream"
        )
    selected_components = integer_list(subset.get("selected_components"), "selected components")
    if (
        selected_components != sorted(set(selected_components))
        or selected_components != fold["selected_components"]
        or len(selected_components) != fold["selected_component_count"]
        or subset.get("selected_component_count") != fold["selected_component_count"]
        or selected_components_sha256(selected_components) != fold["selected_component_sha256"]
    ):
        raise ValueError("selected component identity differs from frozen membership pins")
    component_by_ref: dict[tuple[str, int], int] = {}
    for source_split in ("train", "valid"):
        nodes = cast(
            list[dict[str, object]], cast(dict[str, object], split["sequences"])[source_split]
        )
        for node in nodes:
            key = (source_split, int(node["row_group"]))
            if key in component_by_ref:
                raise ValueError(f"duplicate source-qualified split reference: {key}")
            component_by_ref[key] = int(node["component"])
    if any(
        (source_split, group) not in component_by_ref
        for source_split, groups in allowed_groups.items()
        for group in groups
    ):
        raise ValueError("canonical fit groups are missing from the split sequence map")
    selected_members = {("train", group) for group in train_groups} | {
        ("valid", group) for group in valid_groups
    }
    selected_member_components = {
        component_by_ref[member] for member in selected_members if member in component_by_ref
    }
    if len(selected_member_components) != len(set(selected_components)):
        raise ValueError("selected component identity differs from source-qualified groups")
    expected_component_members = {
        (source_split, group)
        for source_split in ("train", "valid")
        for group in allowed_groups[source_split]
        if component_by_ref[(source_split, group)] in set(selected_components)
    }
    if selected_members != expected_component_members:
        raise ValueError("fit subset splits a source-qualified dependency component")
    evaluation = cast(dict[str, object], split["evaluation"])
    split_eval_groups = sorted(integer_list(evaluation.get("valid"), "evaluation valid groups"))
    subset_eval_groups = integer_list(subset.get("evaluation_groups"), "subset evaluation groups")
    if (
        subset_eval_groups != sorted(set(subset_eval_groups))
        or subset_eval_groups != split_eval_groups
        or subset.get("evaluation_group_count") != len(split_eval_groups)
        or subset.get("evaluation_membership_sha256") != evaluation_groups_sha256(split_eval_groups)
    ):
        raise ValueError("fit subset exact evaluation membership differs from the split")
    sequence_components: list[int] = []
    for ref in refs:
        key = (str(ref["split"]), int(ref["row_group"]))
        if key not in component_by_ref:
            raise ValueError(f"fit source reference is absent from split manifest: {key}")
        sequence_components.append(component_by_ref[key])
    if sorted(set(sequence_components)) != selected_components:
        raise ValueError("selected source-qualified components differ from cache references")
    eval_groups = integer_list(
        cast(dict[str, object], split["evaluation"]).get("valid"), "evaluation valid groups"
    )
    if eval_groups != sorted(set(eval_groups)):
        raise ValueError("canonical evaluation groups are not sorted and unique")
    eval_nodes = cast(list[dict[str, object]], cast(dict[str, object], split["sequences"])["valid"])
    eval_component_by_group = {
        int(node["row_group"]): int(node["component"]) for node in eval_nodes
    }
    eval_components = {eval_component_by_group[int(group)] for group in eval_groups}
    if eval_components.intersection(sequence_components):
        raise ValueError("fit and evaluation dependency components overlap")
    component_labels = selected_components
    component_index = {label: index for index, label in enumerate(component_labels)}
    sequence_indices = [component_index[label] for label in sequence_components]
    group_counts = {"train": len(train_groups), "valid": len(valid_groups)}

    transport_folds = cast(dict[str, object], transport["folds"])
    transport_fold = cast(dict[str, object], transport_folds[str(fold["fold"])])
    transport_suffix = cast(dict[str, object], transport_fold["fit_suffix5"])
    transport_fields = ("bytes", "dtype", "header_bytes", "sha256", "shape")
    if (
        any(transport_suffix.get(key) != expected.get(key) for key in transport_fields)
        or transport_suffix.get("path") != fold["transport_relative_path"]
    ):
        raise ValueError("portable transport metadata differs from frozen suffix pin")
    transport_sidecars = cast(dict[str, object], transport_fold["sidecars"])
    if (
        cast(dict[str, object], transport_sidecars["fit_sequences"])["sha256"]
        != fold["sidecar_sha256"]
    ):
        raise ValueError("portable transport does not bind the frozen fit sidecar")
    return refs, sequence_indices, component_labels, group_counts


def summarize_fold(
    fold: dict[str, object],
    recipe: dict[str, object],
    repository: Path,
    transport: dict[str, object],
) -> dict[str, object]:
    """Stream one admitted suffix in original 8,192-row batch order and summarize it."""
    refs, sequence_indices, component_labels, group_counts = validate_fold_metadata(
        fold, repository, transport
    )
    suffix_pin = cast(dict[str, object], fold["suffix"])
    path = resolve_path(repository, str(suffix_pin["path"]))
    if path.stat().st_size != int(suffix_pin["bytes"]):
        raise ValueError("portable suffix byte length differs from frozen pin")
    total_rows = len(refs) * ROWS_PER_SEQUENCE
    if total_rows != int(suffix_pin["shape"][0]) * ROWS_PER_SEQUENCE:
        raise ValueError("suffix row count differs from exact source membership")
    expected_file_bytes = int(suffix_pin["bytes"])
    digest = hashlib.sha256()
    mass_records: list[tuple[float, float, float]] = []
    distortion_factors: list[float | None] = []
    target_shares: list[float | None] = []
    batch_losses: list[float] = []
    batch_target_numerators = np.zeros(2, dtype=np.float64)
    equal_batch_target_contributions = np.zeros(2, dtype=np.float64)
    pooled_target_mass = np.zeros(2, dtype=np.float64)
    pooled_component_mass = np.zeros(len(component_labels), dtype=np.float64)
    equal_batch_component_mass = np.zeros(len(component_labels), dtype=np.float64)
    row_offset = 0
    with path.open("rb") as source:
        header, shape, dtype, fortran = load_array_header(source)
        if (
            shape != tuple(cast(list[int], suffix_pin["shape"]))
            or dtype != "<f4"
            or fortran
            or len(header) != int(suffix_pin["header_bytes"])
        ):
            raise ValueError("NPY header differs from frozen compact suffix schema")
        digest.update(header)
        while row_offset < total_rows:
            block_rows = min(READ_BATCHES * BATCH_ROWS, total_rows - row_offset)
            block_bytes = block_rows * SUFFIX_WIDTH * SUFFIX_DTYPE.itemsize
            block = source.read(block_bytes)
            if len(block) != block_bytes:
                raise ValueError("truncated compact suffix payload")
            digest.update(block)
            block_values = np.frombuffer(block, dtype=SUFFIX_DTYPE).reshape(
                block_rows, SUFFIX_WIDTH
            )
            block_at = 0
            while block_at < block_rows:
                count = min(BATCH_ROWS, block_rows - block_at)
                suffix = block_values[block_at : block_at + count]
                row_components = component_rows(row_offset + block_at, count, sequence_indices)
                masses, numerators, component_mass = batch_statistics(
                    suffix, row_components, len(component_labels)
                )
                joint_mass = float(masses.sum())
                joint_numerator = float(numerators.sum())
                batch_denominator = max(joint_mass, BATCH_DENOMINATOR_EPSILON)
                mass_records.append((float(masses[0]), float(masses[1]), joint_mass))
                distortion_factors.append(None)
                target_shares.append(float(masses[0] / joint_mass) if joint_mass > 0 else None)
                batch_losses.append(joint_numerator / batch_denominator)
                batch_target_numerators += numerators
                equal_batch_target_contributions += numerators / batch_denominator
                pooled_target_mass += masses
                pooled_component_mass += component_mass
                equal_batch_component_mass += component_mass / batch_denominator
                block_at += count
            row_offset += block_rows
        if source.read(1):
            raise ValueError("compact suffix has trailing bytes beyond its frozen shape")
        if source.tell() != expected_file_bytes:
            raise ValueError("compact suffix file length changed during the admitted read")
    if digest.hexdigest() != suffix_pin["sha256"]:
        raise ValueError("compact suffix payload hash differs from the frozen transport pin")

    batch_count = len(mass_records)
    if batch_count != int(np.ceil(total_rows / BATCH_ROWS)):
        raise ValueError("batch count differs from the canonical fit iterator")
    joint_mass = float(pooled_target_mass.sum())
    pooled_loss = float(batch_target_numerators.sum() / joint_mass) if joint_mass > 0 else 0.0
    equal_batch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
    avg_batch_mass = joint_mass / batch_count if batch_count else 0.0
    distortion_factors = [avg_batch_mass / row[2] if row[2] > 0 else None for row in mass_records]
    masses_array = np.asarray(mass_records, dtype=np.float64)
    pooled_shares = pooled_component_mass / joint_mass if joint_mass > 0 else pooled_component_mass
    positive_mass_batches = sum(row[2] > 0 for row in mass_records)
    equal_component_total = float(equal_batch_component_mass.sum())
    equal_shares = (
        equal_batch_component_mass / equal_component_total
        if equal_component_total > 0
        else equal_batch_component_mass
    )
    component_group_counts: dict[int, dict[str, int]] = {
        component_id: {"train": 0, "valid": 0} for component_id in component_labels
    }
    for ref, component_index_value in zip(refs, sequence_indices, strict=True):
        component_id = component_labels[component_index_value]
        source_split = str(ref["split"])
        counts = component_group_counts.setdefault(component_id, {"train": 0, "valid": 0})
        counts[source_split] += 1
    component_rows_out = [
        {
            "component_id": component_id,
            "source_qualified_group_counts": component_group_counts[component_id],
            "pooled_mass_share": float(pooled_shares[index]),
            "equal_batch_mass_share": float(equal_shares[index]),
        }
        for index, component_id in enumerate(component_labels)
    ]
    return {
        "fold": fold["fold"],
        "seed": fold["seed"],
        "scope": "in_sample_fit_only",
        "source_suffix_sha256_verified": digest.hexdigest(),
        "source_suffix_bytes": expected_file_bytes,
        "source_membership_sha256": fold["membership_sha256"],
        "source_group_counts": group_counts,
        "fit_sequence_count": len(refs),
        "source_qualified_component_count": len(component_labels),
        "row_count": total_rows,
        "batch_rows": BATCH_ROWS,
        "batch_count": batch_count,
        "final_batch_rows": total_rows - BATCH_ROWS * (batch_count - 1),
        "measure": {
            "target_weight_mass_by_target": {
                "t0": quantiles([float(row[0]) for row in mass_records]),
                "t1": quantiles([float(row[1]) for row in mass_records]),
                "joint": quantiles([float(row[2]) for row in mass_records]),
            },
            "total_target_weight_mass": {
                "t0": float(pooled_target_mass[0]),
                "t1": float(pooled_target_mass[1]),
                "joint": joint_mass,
            },
            "global_target_mass_share": {
                "t0": float(pooled_target_mass[0] / joint_mass) if joint_mass else 0.0,
                "t1": float(pooled_target_mass[1] / joint_mass) if joint_mass else 0.0,
            },
            "per_batch_t0_mass_share": quantiles(target_shares),
            "zero_mass_batch_count": int(np.count_nonzero(masses_array[:, 2] == 0)),
            "positive_mass_batch_count": int(positive_mass_batches),
            "distortion_factor_mean_batch_mass_over_batch_mass": quantiles(distortion_factors),
            "mean_batch_joint_mass": avg_batch_mass,
        },
        "parent_squared_error": {
            "prediction_is_unclipped_parent_raw": True,
            "targets_clipped_to": [-2.0, 2.0],
            "weights": "abs(clipped target) * transported row_weight",
            "per_batch_denominator_clamp_min": BATCH_DENOMINATOR_EPSILON,
            "sum_precision": "float64 descriptive reconstruction; no Torch FP32 reduction parity claimed",
            "pooled_joint_loss": pooled_loss,
            "equal_batch_normalized_joint_loss": equal_batch_loss,
            "pooled_by_target_contribution_to_joint_measure": {
                "t0": float(batch_target_numerators[0] / joint_mass) if joint_mass else 0.0,
                "t1": float(batch_target_numerators[1] / joint_mass) if joint_mass else 0.0,
            },
            "equal_batch_by_target_contribution_to_joint_measure": {
                "t0": float(equal_batch_target_contributions[0] / batch_count)
                if batch_count
                else 0.0,
                "t1": float(equal_batch_target_contributions[1] / batch_count)
                if batch_count
                else 0.0,
            },
        },
        "component_concentration": {
            "equal_batch_measure_uses_source_denominator_clamp": BATCH_DENOMINATOR_EPSILON,
            "zero_mass_batches_contribute_zero": True,
            "pooled_mass_measure": concentration(pooled_shares),
            "equal_batch_normalized_measure": concentration(equal_shares),
            "component_shares": component_rows_out,
        },
    }


def main() -> int:
    """Run only after validating the separate root admission receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    recipe_path = args.recipe.resolve()
    manifest_path = args.source_manifest.resolve()
    recipe = read_json(recipe_path)
    manifest = read_json(manifest_path)
    recipe_sha = sha256_file(recipe_path)
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != recipe.get("source_manifest_sha256"):
        raise ValueError("source manifest hash differs from the frozen recipe")
    if manifest_path != resolve_path(
        Path(str(recipe["repository_root"])), str(recipe["source_manifest_path"])
    ):
        raise ValueError("source manifest path differs from the frozen recipe")
    verify_sources(manifest, Path(str(recipe["repository_root"])))
    admission_path = args.admission.resolve()
    if admission_path != resolve_path(
        Path(str(recipe["repository_root"])), str(recipe["admission_path"])
    ):
        raise PermissionError("admission path differs from the frozen recipe")
    admission = read_json(admission_path)
    validate_admission(admission, recipe_sha, manifest_sha, recipe)
    output_path = resolve_path(Path(str(recipe["repository_root"])), str(recipe["output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"output is create-only: {output_path}")
    transport_path = resolve_path(
        Path(str(recipe["repository_root"])), str(recipe["transport_path"])
    )
    transport = read_json(transport_path)
    folds = cast(list[dict[str, object]], recipe["folds"])
    results = [
        summarize_fold(fold, recipe, Path(str(recipe["repository_root"])), transport)
        for fold in folds
    ]
    result: dict[str, object] = {
        "schema": "wnn.f0-f1-training-batch-mass-audit.v1",
        "status": "complete_descriptive_fit_measure_audit",
        "recipe_sha256": recipe_sha,
        "source_manifest_sha256": manifest_sha,
        "admission_sha256": sha256_file(admission_path),
        "metric_and_loss_source_sha256": recipe["primary_pair_wmse_source_sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "new_fit_count": 0,
        "model_imports": 0,
        "evaluation_labels_read": False,
        "torch_imported": False,
        "precision_scope": "float64 descriptive reconstruction from pinned FP32 suffix values",
        "limits": [
            "No Torch FP32 reduction or optimizer trajectory parity is claimed.",
            "The parent squared-error audit is a training-objective diagnostic, not official WP.",
            "Both endpoint folds and their fitting groups are previously exposed development data.",
        ],
        "folds": results,
    }
    encoded = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
