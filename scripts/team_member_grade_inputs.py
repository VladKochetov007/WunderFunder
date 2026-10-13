"""Bounded input validation and row-group access for member-bank grading."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.team_member_grade_support import (
    N_OUTPUTS,
    TARGET_COLUMNS,
    HashState,
    _as_int,
    _as_object,
    _as_sha256,
    _as_string,
    _load_prediction_map,
    _validate_dependencies,
    digest_file,
    read_bounded_bytes,
    stable_file_identity,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.data import SEQUENCE_LENGTH, WARMUP  # noqa: E402


@dataclass
class MemberInput:
    """Validated export metadata and its read-only disk-backed predictions."""

    plan_record: dict[str, object]
    result: dict[str, object]
    result_path: Path
    result_sha256: str
    result_stat: os.stat_result
    prediction_path: Path
    predictions: np.memmap
    prediction_hasher: HashState
    prediction_stat: os.stat_result
    weight_path: Path
    weight_stat: os.stat_result
    group_results: list[dict[str, object]]


def _validate_member_result(
    record: dict[str, object],
    inventory: dict[str, object],
    inventory_path: Path,
    inventory_sha256: str,
    data_path: Path,
    data_sha256: str,
    group_ids: tuple[int, ...],
    bank_root: Path,
) -> tuple[MemberInput, dict[str, str], dict[str, str], dict[str, object]]:
    member_id = _as_string(record.get("id"), "member.id")
    candidate = _as_string(record.get("candidate"), f"{member_id}.candidate")
    member = _as_string(record.get("member"), f"{member_id}.member")
    if Path(member_id).name != member_id or member_id in {".", ".."}:
        raise ValueError("member ID must be a safe directory basename")
    if Path(member).name != member or not member.endswith(".npz"):
        raise ValueError(f"member must be a declared NPZ basename: {member_id}")
    expected_weight_sha = _as_sha256(record.get("sha256"), f"{member_id}.sha256")
    expected_source = _as_object(record.get("source"), f"{member_id}.source")

    packages = _as_object(inventory.get("packages"), "inventory.packages")
    package = _as_object(packages.get(candidate), f"inventory.packages.{candidate}")
    files = _as_object(package.get("files"), f"{candidate}.files")
    weight_record = _as_object(files.get(member), f"{candidate}.{member}")
    if _as_sha256(weight_record.get("sha256"), "inventory weight sha256") != expected_weight_sha:
        raise ValueError(f"plan weight hash differs from inventory for {member_id}")
    package_members = package.get("members")
    if not isinstance(package_members, list):
        raise ValueError(f"inventory package has no active member list: {candidate}")
    inventory_source = next(
        (
            _as_object(item, f"{candidate}.members entry")
            for item in package_members
            if isinstance(item, dict) and item.get("weights") == member
        ),
        None,
    )
    if inventory_source != expected_source:
        raise ValueError(f"plan source/order differs from inventory for {member_id}")
    package_path = Path(_as_string(package.get("package"), f"{candidate}.package"))
    weight_path = package_path / member
    if not weight_path.is_file() or digest_file(weight_path) != expected_weight_sha:
        raise ValueError(f"original weight bytes differ from the plan for {member_id}")
    weight_stat = weight_path.stat()

    result_path = bank_root / member_id / "RESULT.json"
    prediction_path = result_path.parent / "predictions.npy"
    result_bytes, result_stat = read_bounded_bytes(result_path)
    try:
        result = _as_object(json.loads(result_bytes), str(result_path))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read member result {result_path}: {exc}") from exc
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    if stable_file_identity(result_path.stat()) != stable_file_identity(result_stat):
        raise ValueError(f"member RESULT changed while it was read: {member_id}")
    if result.get("status") != "complete":
        raise ValueError(f"member export is not terminal-complete: {member_id}")
    if result.get("candidate") != candidate or result.get("member") != member:
        raise ValueError(f"member result identity differs from plan: {member_id}")
    if result.get("member_source") != expected_source:
        raise ValueError(f"member source/order differs from frozen plan: {member_id}")
    if result.get("weight_sha256") != expected_weight_sha:
        raise ValueError(f"result weight SHA differs from plan: {member_id}")
    if (
        Path(_as_string(result.get("weight_path"), "result.weight_path")).resolve()
        != weight_path.resolve()
    ):
        raise ValueError(f"result weight path differs from inventory: {member_id}")
    if result.get("archive_sha256") != package.get("sha256"):
        raise ValueError(f"result archive SHA differs from inventory: {member_id}")
    if (
        Path(_as_string(result.get("archive_path"), "result.archive_path")).resolve()
        != Path(_as_string(package.get("archive"), f"{candidate}.archive")).resolve()
    ):
        raise ValueError(f"result archive path differs from inventory: {member_id}")
    if result.get("archive_hash_verified_by_this_run") is not True:
        raise ValueError(f"source archive was not verified by exporter: {member_id}")
    if result.get("inventory_sha256") != inventory_sha256:
        raise ValueError(f"result inventory SHA differs: {member_id}")
    if (
        Path(_as_string(result.get("inventory_path"), "result.inventory_path")).resolve()
        != inventory_path.resolve()
    ):
        raise ValueError(f"result inventory path differs: {member_id}")
    if (
        result.get("data_sha256") != data_sha256
        or result.get("data_hash_verified_by_this_run") is not True
    ):
        raise ValueError(f"result did not verify the frozen data SHA: {member_id}")
    if result.get("data_bytes") != data_path.stat().st_size:
        raise ValueError(f"result data size differs: {member_id}")
    if (
        Path(_as_string(result.get("data_path"), "result.data_path")).resolve()
        != data_path.resolve()
    ):
        raise ValueError(f"result data path differs: {member_id}")

    expected_shape = [len(group_ids), SEQUENCE_LENGTH, N_OUTPUTS]
    if (
        result.get("prediction_shape") != expected_shape
        or result.get("prediction_dtype") != "float32"
    ):
        raise ValueError(f"result prediction shape/dtype differs: {member_id}")
    if (
        Path(_as_string(result.get("prediction_path"), "result.prediction_path")).resolve()
        != prediction_path.resolve()
    ):
        raise ValueError(f"result prediction path differs: {member_id}")
    expected_groups = list(group_ids)
    if result.get("row_groups") != expected_groups:
        raise ValueError(f"result row-group order differs: {member_id}")
    raw_group_results = result.get("group_results")
    if not isinstance(raw_group_results, list) or len(raw_group_results) != len(group_ids):
        raise ValueError(f"result lacks per-group provenance: {member_id}")
    group_results: list[dict[str, object]] = []
    for index, raw_group in enumerate(raw_group_results):
        group = _as_object(raw_group, f"{member_id}.group_results[{index}]")
        if _as_int(group.get("row_group"), "group.row_group") != group_ids[index]:
            raise ValueError(f"group result ordering differs: {member_id} at {index}")
        _as_int(group.get("seq_ix"), "group.seq_ix")
        _as_sha256(group.get("feature_sha256"), "group.feature_sha256")
        _as_sha256(group.get("prediction_sha256"), "group.prediction_sha256")
        if group.get("warmup_zeroed") != WARMUP:
            raise ValueError(f"warmup treatment differs: {member_id} at {index}")
        group_results.append(group)

    source_digests, source_paths = _validate_dependencies(result)
    prediction_sha256 = _as_sha256(result.get("prediction_sha256"), "prediction_sha256")
    predictions, prediction_hasher = _load_prediction_map(
        prediction_path, (len(group_ids), SEQUENCE_LENGTH, N_OUTPUTS)
    )
    source_record = {
        "member_id": member_id,
        "candidate": candidate,
        "member": member,
        "member_source": expected_source,
        "weight_sha256": expected_weight_sha,
        "archive_sha256": _as_sha256(package.get("sha256"), "package.sha256"),
        "export_result_sha256": result_sha256,
        "prediction_sha256": prediction_sha256,
    }
    loaded = MemberInput(
        record,
        result,
        result_path,
        result_sha256,
        result_stat,
        prediction_path,
        predictions,
        prediction_hasher,
        prediction_path.stat(),
        weight_path,
        weight_stat,
        group_results,
    )
    return loaded, source_digests, source_paths, source_record


def _read_target_group(parquet: pq.ParquetFile, group: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Read labels and the official prediction/scoring mask for one sequence."""
    columns = ["seq_ix", "step_in_seq", "need_prediction", "is_scored", *TARGET_COLUMNS]
    table = parquet.read_row_group(group, columns=columns, use_threads=False)
    if table.num_rows != SEQUENCE_LENGTH:
        raise ValueError(f"row group {group} has {table.num_rows} rows, expected {SEQUENCE_LENGTH}")
    seq_ids = table.column("seq_ix").to_numpy()
    if not np.all(seq_ids == seq_ids[0]):
        raise ValueError(f"row group {group} contains multiple source sequence IDs")
    steps = table.column("step_in_seq").to_numpy()
    if not np.array_equal(steps, np.arange(SEQUENCE_LENGTH, dtype=steps.dtype)):
        raise ValueError(f"row group {group} is not in complete step order")
    need = np.asarray(table.column("need_prediction").to_numpy(), dtype=np.bool_)
    scored = np.asarray(table.column("is_scored").to_numpy(), dtype=np.bool_)
    mask = np.ascontiguousarray(need & scored)
    if mask[:WARMUP].any():
        raise ValueError(f"official mask selects warmup rows in row group {group}")
    targets = np.ascontiguousarray(
        np.column_stack([table.column(name).to_numpy() for name in TARGET_COLUMNS]).astype(
            np.float32, copy=False
        )
    )
    if targets.shape != (SEQUENCE_LENGTH, N_OUTPUTS):
        raise ValueError(f"row group {group} target shape is invalid")
    return int(seq_ids[0]), targets, mask


def _evict_maps(members: list[MemberInput]) -> None:
    """Discard clean mapped pages after each bounded group block."""
    for member in members:
        mapped = getattr(member.predictions, "_mmap", None)
        if mapped is not None and hasattr(mapped, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
            mapped.madvise(mmap.MADV_DONTNEED)
        if hasattr(os, "posix_fadvise"):
            with member.prediction_path.open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
