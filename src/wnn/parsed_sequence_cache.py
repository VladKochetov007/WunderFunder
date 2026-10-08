"""Build a cohort-scoped, source-pinned parsed sequence cache.

The cache stores complete row groups. Fitting subset membership and fixed
evaluation membership remain separate, source-qualified lists in the sealed
membership artifact. Runtime counters describe observed work; they are never
filled with expected values such as "one parse".
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from wnn import data as data_module
from wnn.data import N_FEATURES, SEQUENCE_LENGTH, SequenceFile

CACHE_SCHEMA = "wnn.parsed-sequence-cache"
MEMBERSHIP_SCHEMA = "wnn.parsed-sequence-cache-membership"
SOURCE_RECEIPT_SCHEMA = "wnn.cohort-source-verification"
FORMAT_VERSION = 1
_ROLES = ("fit_train", "fit_valid", "evaluation_valid")
_SOURCES = {
    "train": "datasets/train.parquet",
    "valid": "datasets/valid.parquet",
}
_ARRAY_DTYPES = {
    "features": np.dtype("<f4"),
    "targets": np.dtype("<f4"),
    "scored": np.dtype("|b1"),
}
_METADATA_RESERVE_BYTES = 1 << 20


class CacheAdmissionError(ValueError):
    """The sealed cache inputs or local capacity fail closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, expected_sha256: str, label: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != expected_sha256:
        raise CacheAdmissionError(f"{label} SHA-256 does not match the frozen pin")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise CacheAdmissionError(f"{label} must be a JSON object")
    return value, digest


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _source_identity(
    split: dict[str, Any], registry: dict[str, Any], logical_name: str
) -> dict[str, Any]:
    identity = split.get("sources", {}).get(logical_name)
    registered = registry.get("sources", {}).get(logical_name)
    if not isinstance(identity, dict) or not isinstance(registered, dict):
        raise CacheAdmissionError(f"missing frozen source identity: {logical_name}")
    if registered.get("original_identity") != identity:
        raise CacheAdmissionError(f"registry identity differs from split manifest: {logical_name}")
    digest = registered.get("sha256")
    if not _valid_sha256(digest):
        raise CacheAdmissionError(f"invalid registry source SHA-256: {logical_name}")
    size = identity.get("bytes")
    if not isinstance(size, int) or size <= 0:
        raise CacheAdmissionError(f"invalid frozen source size: {logical_name}")
    return {"logical_name": logical_name, "bytes": size, "sha256": digest}


def _split_records(split: dict[str, Any], split_name: str) -> dict[int, dict[str, int]]:
    raw = split.get("sequences", {}).get(split_name)
    if not isinstance(raw, list):
        raise CacheAdmissionError(f"split manifest has no {split_name} sequence records")
    records: dict[int, dict[str, int]] = {}
    sequence_ids: set[int] = set()
    for node in raw:
        try:
            row_group = int(node["row_group"])
            component = int(node["component"])
            seq_ix = int(node["seq_ix"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheAdmissionError(f"invalid {split_name} sequence record") from exc
        if row_group < 0 or component < 0 or row_group in records or seq_ix in sequence_ids:
            raise CacheAdmissionError(f"duplicate or negative {split_name} sequence identity")
        records[row_group] = {"component": component, "seq_ix": seq_ix}
        sequence_ids.add(seq_ix)
    return records


def _int_groups(split: dict[str, Any], section: str, source: str) -> list[int]:
    values = split.get(section, {}).get(source)
    if not isinstance(values, list):
        raise CacheAdmissionError(f"split manifest has no {section}.{source} groups")
    try:
        groups = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise CacheAdmissionError(f"invalid {section}.{source} group list") from exc
    if any(group < 0 for group in groups) or len(set(groups)) != len(groups):
        raise CacheAdmissionError(f"duplicate or negative group in {section}.{source}")
    return groups


def _member(
    *, source: dict[str, Any], row_group: int, record: dict[str, int], array_index: int
) -> dict[str, Any]:
    seq_ix = record["seq_ix"]
    return {
        "source": source["logical_name"],
        "source_sha256": source["sha256"],
        "row_group": row_group,
        "seq_ix": seq_ix,
        "sequence_id": f"sha256:{source['sha256']}:seq_ix:{seq_ix}",
        "component_id": record["component"],
        "array_index": array_index,
    }


def prepare_membership(
    *,
    split_manifest_path: Path,
    portable_registry_path: Path,
    expected_split_sha256: str,
    expected_registry_sha256: str,
    fold: int,
    fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Create a deterministic whole-component subset and full eval membership."""
    split, split_sha = _read_json(split_manifest_path, expected_split_sha256, "split manifest")
    registry, registry_sha = _read_json(
        portable_registry_path, expected_registry_sha256, "portable registry"
    )
    if split.get("version") != 1 or registry.get("version") != 1:
        raise CacheAdmissionError("unsupported split or portable registry version")
    if int(split.get("fold", -1)) != fold or not 0.0 < fraction <= 1.0:
        raise CacheAdmissionError("requested fold/fraction does not match the sealed split")
    if registry.get("split_manifest_sha256") != split_sha:
        raise CacheAdmissionError("portable registry does not pin this split manifest")

    sources = {
        name: _source_identity(split, registry, logical_name)
        for name, logical_name in _SOURCES.items()
    }
    records = {name: _split_records(split, name) for name in _SOURCES}
    fit_train = _int_groups(split, "fit", "train")
    fit_valid = _int_groups(split, "fit", "valid")
    evaluation_valid = _int_groups(split, "evaluation", "valid")
    if set(fit_train) - records["train"].keys():
        raise CacheAdmissionError("fit train list names an unknown row group")
    if (set(fit_valid) | set(evaluation_valid)) - records["valid"].keys():
        raise CacheAdmissionError("fit/evaluation valid list names an unknown row group")
    if set(fit_valid) & set(evaluation_valid):
        raise CacheAdmissionError("fit and evaluation valid groups overlap")

    component_of = {
        (name, group): record["component"]
        for name, group_records in records.items()
        for group, record in group_records.items()
    }
    fit_components = {component_of[("train", group)] for group in fit_train} | {
        component_of[("valid", group)] for group in fit_valid
    }
    evaluation_components = {component_of[("valid", group)] for group in evaluation_valid}
    if not fit_components or not evaluation_valid or fit_components & evaluation_components:
        raise CacheAdmissionError("empty membership or fitting/evaluation component overlap")
    eligible = sorted(fit_components)
    selected_count = min(len(eligible), max(1, math.ceil(len(eligible) * fraction)))
    ranked = sorted(
        eligible,
        key=lambda component: hashlib.sha256(f"{seed}:{component}".encode()).hexdigest(),
    )
    selected = set(ranked[:selected_count])
    selected_train = sorted(
        group for group in fit_train if component_of[("train", group)] in selected
    )
    selected_valid = sorted(
        group for group in fit_valid if component_of[("valid", group)] in selected
    )
    if not selected_train or not selected_valid:
        raise CacheAdmissionError("selected components must retain train and fit-valid rows")

    train_indices = {group: index for index, group in enumerate(selected_train)}
    valid_order = sorted(set(selected_valid) | set(evaluation_valid))
    valid_indices = {group: index for index, group in enumerate(valid_order)}
    members = {
        "fit_train": [
            _member(
                source=sources["train"],
                row_group=group,
                record=records["train"][group],
                array_index=train_indices[group],
            )
            for group in selected_train
        ],
        "fit_valid": [
            _member(
                source=sources["valid"],
                row_group=group,
                record=records["valid"][group],
                array_index=valid_indices[group],
            )
            for group in selected_valid
        ],
        "evaluation_valid": [
            _member(
                source=sources["valid"],
                row_group=group,
                record=records["valid"][group],
                array_index=valid_indices[group],
            )
            for group in sorted(evaluation_valid)
        ],
    }
    selection = {
        "fraction": fraction,
        "seed": int(seed),
        "method": "sha256-sort-whole-component-v1",
        "eligible_component_count": len(eligible),
        "selected_component_count": len(selected),
        "selected_component_ids": sorted(selected),
    }
    selection["selected_component_ids_sha256"] = sha256_bytes(
        canonical_json_bytes(selection["selected_component_ids"])
    )
    return {
        "schema": MEMBERSHIP_SCHEMA,
        "format_version": FORMAT_VERSION,
        "fold": fold,
        "split_manifest_sha256": split_sha,
        "portable_registry_sha256": registry_sha,
        "sources": sources,
        "selection": selection,
        "memberships": members,
        "member_counts": {role: len(members[role]) for role in _ROLES},
        "array_shapes": {
            "train": [len(selected_train), SEQUENCE_LENGTH, N_FEATURES],
            "valid": [len(valid_order), SEQUENCE_LENGTH, N_FEATURES],
        },
    }


def _stat_identity(stat: os.stat_result) -> dict[str, int]:
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _proc_io() -> dict[str, int] | None:
    try:
        lines = Path("/proc/self/io").read_text().splitlines()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in lines:
        key, sep, value = line.partition(":")
        if sep:
            try:
                values[key] = int(value.strip())
            except ValueError:
                continue
    return values


def _memory_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, path in (
        ("cgroup_current_bytes", Path("/sys/fs/cgroup/memory.current")),
        ("cgroup_max_bytes", Path("/sys/fs/cgroup/memory.max")),
    ):
        try:
            value = path.read_text().strip()
            if value != "max":
                result[key] = int(value)
        except (OSError, ValueError):
            pass
    for path in (
        Path("/sys/fs/cgroup/memory.stat"),
        Path("/sys/fs/cgroup/wnn-campaign.slice/memory.stat"),
    ):
        try:
            for line in path.read_text().splitlines():
                key, value = line.split()
                if key in {"anon", "file", "inactive_file", "active_file", "slab"}:
                    result[f"cgroup_{key}_bytes"] = int(value)
            if result:
                break
        except (OSError, ValueError):
            continue
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Rss", "Pss", "Anonymous", "Private_Clean", "Private_Dirty"}:
                result[f"process_{key.lower()}_bytes"] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return result


def _io_delta(before: dict[str, int] | None, after: dict[str, int] | None) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    return {key: after[key] - before.get(key, 0) for key in after}


def _phase_snapshot() -> tuple[float, dict[str, int] | None, dict[str, Any]]:
    return time.perf_counter(), _proc_io(), _memory_snapshot()


def _phase_result(start: tuple[float, dict[str, int] | None, dict[str, Any]]) -> dict[str, Any]:
    started, before_io, before_memory = start
    return {
        "wall_seconds": time.perf_counter() - started,
        "process_io_delta": _io_delta(before_io, _proc_io()),
        "memory_before": before_memory,
        "memory_after": _memory_snapshot(),
    }


def _advise_dontneed(fd: int, offset: int = 0, length: int = 0) -> bool:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False


def _hash_source(path: Path, *, chunk_bytes: int = 8 << 20) -> tuple[str, dict[str, Any]]:
    """Hash one source in a bounded stream and advise consumed pages away."""
    fd = os.open(path, os.O_RDONLY)
    digest = hashlib.sha256()
    bytes_hashed = 0
    advice_calls = 0
    start = _phase_snapshot()
    try:
        initial = os.fstat(fd)
        while True:
            block = os.read(fd, chunk_bytes)
            if not block:
                break
            digest.update(block)
            end = bytes_hashed + len(block)
            bytes_hashed = end
            if _advise_dontneed(fd, end - len(block), len(block)):
                advice_calls += 1
        final = os.fstat(fd)
    finally:
        os.close(fd)
    if _stat_identity(initial) != _stat_identity(final):
        raise CacheAdmissionError(f"source changed during verification: {path.name}")
    telemetry = _phase_result(start)
    telemetry.update({"bytes_hashed": bytes_hashed, "fadvise_dontneed_calls": advice_calls})
    return digest.hexdigest(), telemetry


def _verify_sources_once(
    *, source_root: Path, membership: dict[str, Any], registry_sha256: str
) -> dict[str, Any]:
    start = _phase_snapshot()
    observed: dict[str, Any] = {}
    per_source: dict[str, Any] = {}
    for source in membership["sources"].values():
        logical_name = source["logical_name"]
        path = source_root / logical_name
        if not path.is_file():
            raise CacheAdmissionError(f"missing source file: {logical_name}")
        before = path.stat()
        if before.st_size != source["bytes"]:
            raise CacheAdmissionError(f"source size differs from registry: {logical_name}")
        digest, telemetry = _hash_source(path)
        after = path.stat()
        if digest != source["sha256"] or _stat_identity(before) != _stat_identity(after):
            raise CacheAdmissionError(f"source bytes differ from registry: {logical_name}")
        observed[logical_name] = {
            "bytes": source["bytes"],
            "sha256": digest,
            "stat": _stat_identity(after),
        }
        per_source[logical_name] = telemetry
    return {
        "schema": SOURCE_RECEIPT_SCHEMA,
        "format_version": FORMAT_VERSION,
        "portable_registry_sha256": registry_sha256,
        "split_manifest_sha256": membership["split_manifest_sha256"],
        "sources": observed,
        "hash_telemetry": {
            "per_source": per_source,
            "aggregate": _phase_result(start),
        },
    }


def validate_source_verification_receipt(
    *,
    receipt: dict[str, Any],
    expected_receipt_sha256: str,
    receipt_path: Path,
    membership: dict[str, Any],
    source_root: Path,
) -> None:
    """Validate a pinned cohort receipt without rehashing the large inputs."""
    observed_receipt_sha = sha256_file(receipt_path)
    if observed_receipt_sha != expected_receipt_sha256:
        raise CacheAdmissionError("source verification receipt SHA-256 is not pinned")
    if (
        receipt.get("schema") != SOURCE_RECEIPT_SCHEMA
        or receipt.get("format_version") != FORMAT_VERSION
        or receipt.get("portable_registry_sha256") != membership["portable_registry_sha256"]
        or receipt.get("split_manifest_sha256") != membership["split_manifest_sha256"]
    ):
        raise CacheAdmissionError("source verification receipt binds different inputs")
    for source in membership["sources"].values():
        logical_name = source["logical_name"]
        row = receipt.get("sources", {}).get(logical_name)
        if (
            not isinstance(row, dict)
            or row.get("sha256") != source["sha256"]
            or row.get("bytes") != source["bytes"]
            or row.get("stat", {}).get("bytes") != source["bytes"]
        ):
            raise CacheAdmissionError(f"source receipt hash mismatch: {logical_name}")
        path = source_root / logical_name
        if not path.is_file():
            raise CacheAdmissionError(f"receipt source is unavailable: {logical_name}")
        if _stat_identity(path.stat()) != row.get("stat"):
            raise CacheAdmissionError(
                f"source stat identity changed after cohort verification: {logical_name}"
            )


def _npy_header(dtype: np.dtype, shape: tuple[int, ...]) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        buffer,
        {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return buffer.getvalue()


class _NpyStreamWriter:
    def __init__(self, path: Path, dtype: np.dtype, shape: tuple[int, ...]):
        self.path = path
        self.dtype = dtype
        self.shape = shape
        self._file = path.open("xb")
        self._digest = hashlib.sha256()
        self._written = 0
        header = _npy_header(dtype, shape)
        self._file.write(header)
        self._digest.update(header)
        self._written += len(header)
        self._index = 0

    def append(self, value: np.ndarray) -> None:
        if self._index >= self.shape[0]:
            raise CacheAdmissionError(f"too many rows for {self.path.name}")
        expected_shape = self.shape[1:]
        array = np.asarray(value, dtype=self.dtype, order="C")
        if array.shape != expected_shape or not array.flags.c_contiguous:
            raise CacheAdmissionError(f"array shape/layout mismatch for {self.path.name}")
        raw = memoryview(array).cast("B")
        self._file.write(raw)
        self._digest.update(raw)
        self._written += len(raw)
        self._index += 1

    def close(self) -> dict[str, Any]:
        if self._index != self.shape[0]:
            raise CacheAdmissionError(f"incomplete array file: {self.path.name}")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        expected_size = (
            len(_npy_header(self.dtype, self.shape)) + math.prod(self.shape) * self.dtype.itemsize
        )
        if self._written != expected_size or self.path.stat().st_size != expected_size:
            raise CacheAdmissionError(f"array byte count mismatch: {self.path.name}")
        return {
            "path": self.path.name,
            "dtype": self.dtype.str,
            "shape": list(self.shape),
            "size_bytes": self._written,
            "sha256": self._digest.hexdigest(),
            "hash_method": "incremental-header-and-row-write-v1",
        }


def estimate_array_bytes(membership: dict[str, Any]) -> int:
    """Exact NPY payload plus headers for the six arrays written by the builder."""
    total = 0
    for source_name in ("train", "valid"):
        shape = tuple(map(int, membership["array_shapes"][source_name]))
        if len(shape) != 3 or shape[1:] != (SEQUENCE_LENGTH, N_FEATURES):
            raise CacheAdmissionError("unsupported parsed-cache array shape")
        count = shape[0]
        for array_name, dtype in _ARRAY_DTYPES.items():
            array_shape = (
                shape
                if array_name == "features"
                else (count, SEQUENCE_LENGTH, 2)
                if array_name == "targets"
                else (count, SEQUENCE_LENGTH)
            )
            total += len(_npy_header(dtype, array_shape)) + math.prod(array_shape) * dtype.itemsize
    return total


def _validate_membership(membership: dict[str, Any]) -> None:
    if (
        membership.get("schema") != MEMBERSHIP_SCHEMA
        or membership.get("format_version") != FORMAT_VERSION
    ):
        raise CacheAdmissionError("unsupported parsed-cache membership artifact")
    if membership.get("member_counts") != {
        role: len(membership.get("memberships", {}).get(role, [])) for role in _ROLES
    }:
        raise CacheAdmissionError("membership counts disagree with explicit member lists")
    sequence_ids: set[str] = set()
    component_by_role: dict[str, set[int]] = {}
    source_hashes = {
        source["logical_name"]: source["sha256"]
        for source in membership.get("sources", {}).values()
    }
    expected_source_by_role = {
        "fit_train": _SOURCES["train"],
        "fit_valid": _SOURCES["valid"],
        "evaluation_valid": _SOURCES["valid"],
    }
    for role in _ROLES:
        rows = membership["memberships"][role]
        if not rows:
            raise CacheAdmissionError(f"empty required membership: {role}")
        component_by_role[role] = set()
        for row in rows:
            if (
                row.get("source") not in _SOURCES.values()
                or row.get("source") != expected_source_by_role[role]
                or row.get("source_sha256") != source_hashes.get(row.get("source"))
                or not _valid_sha256(row.get("source_sha256"))
                or int(row.get("row_group", -1)) < 0
                or int(row.get("array_index", -1)) < 0
                or row.get("sequence_id")
                != f"sha256:{row['source_sha256']}:seq_ix:{int(row['seq_ix'])}"
            ):
                raise CacheAdmissionError(f"invalid source-qualified member in {role}")
            sequence_id = row["sequence_id"]
            if sequence_id in sequence_ids:
                raise CacheAdmissionError("sequence appears in more than one role")
            sequence_ids.add(sequence_id)
            component_by_role[role].add(int(row["component_id"]))
    if component_by_role["fit_train"] & component_by_role["evaluation_valid"]:
        raise CacheAdmissionError("fit train and evaluation components overlap")
    if component_by_role["fit_valid"] & component_by_role["evaluation_valid"]:
        raise CacheAdmissionError("fit valid and evaluation components overlap")
    for source_name in ("train", "valid"):
        rows = [
            row
            for role in _ROLES
            for row in membership["memberships"][role]
            if row["source"] == _SOURCES[source_name]
        ]
        indices = sorted(int(row["array_index"]) for row in rows)
        expected_count = membership["array_shapes"][source_name][0]
        if indices != list(range(expected_count)):
            raise CacheAdmissionError(f"{source_name} array indices are not complete and unique")


def build_parsed_cache(
    *,
    membership_path: Path,
    expected_membership_sha256: str,
    source_root: Path,
    output_dir: Path,
    output_reserve_bytes: int,
    checkpoint_reserve_bytes: int,
    free_reserve_bytes: int,
    source_verification_path: Path | None = None,
    expected_source_verification_sha256: str | None = None,
) -> dict[str, Any]:
    """Build complete-sequence arrays after a fail-closed disk admission."""
    membership, membership_sha = _read_json(
        membership_path, expected_membership_sha256, "membership artifact"
    )
    _validate_membership(membership)
    if output_dir.exists() or output_dir.is_symlink():
        raise CacheAdmissionError("output directory already exists; cache builds never overwrite")
    if not output_dir.parent.is_dir():
        raise CacheAdmissionError("output parent must already exist on the target filesystem")
    reserves = (output_reserve_bytes, checkpoint_reserve_bytes, free_reserve_bytes)
    if any(not isinstance(value, int) or value < 0 for value in reserves):
        raise CacheAdmissionError("disk reserves must be nonnegative integer bytes")
    estimated_bytes = estimate_array_bytes(membership)
    membership_copy_bytes = membership_path.stat().st_size
    estimated_cache_bytes = estimated_bytes + membership_copy_bytes + _METADATA_RESERVE_BYTES
    required_bytes = estimated_cache_bytes + sum(reserves)
    free_bytes = shutil.disk_usage(output_dir.parent).free
    if free_bytes < required_bytes:
        raise CacheAdmissionError(
            f"insufficient disk headroom: have {free_bytes}, require {required_bytes} bytes"
        )

    if source_verification_path is None:
        source_verification = _verify_sources_once(
            source_root=source_root,
            membership=membership,
            registry_sha256=membership["portable_registry_sha256"],
        )
        source_verification_bytes = canonical_json_bytes(source_verification) + b"\n"
    else:
        if not expected_source_verification_sha256:
            raise CacheAdmissionError("a pinned SHA-256 is required for a supplied source receipt")
        source_verification_bytes = source_verification_path.read_bytes()
        if sha256_bytes(source_verification_bytes) != expected_source_verification_sha256:
            raise CacheAdmissionError("source verification receipt SHA-256 does not match")
        source_verification = json.loads(source_verification_bytes)
        validate_source_verification_receipt(
            receipt=source_verification,
            expected_receipt_sha256=expected_source_verification_sha256,
            receipt_path=source_verification_path,
            membership=membership,
            source_root=source_root,
        )

    output_dir.mkdir()
    status_path = output_dir / "build_status.json"
    status: dict[str, Any] = {
        "schema": "wnn.parsed-sequence-cache-build-status",
        "format_version": FORMAT_VERSION,
        "state": "building",
        "completed_sequence_reads": 0,
        "expected_sequence_reads": sum(membership["member_counts"][role] for role in _ROLES),
    }
    status_path.write_bytes(canonical_json_bytes(status) + b"\n")
    source_receipt_path = output_dir / "source_verification.json"
    source_receipt_path.write_bytes(source_verification_bytes)

    write_start = _phase_snapshot()
    writers: dict[str, dict[str, _NpyStreamWriter]] = {}
    array_meta: dict[str, dict[str, Any]] = {}
    for source_name in ("train", "valid"):
        count = int(membership["array_shapes"][source_name][0])
        shapes = {
            "features": (count, SEQUENCE_LENGTH, N_FEATURES),
            "targets": (count, SEQUENCE_LENGTH, 2),
            "scored": (count, SEQUENCE_LENGTH),
        }
        writers[source_name] = {}
        for array_name, dtype in _ARRAY_DTYPES.items():
            writers[source_name][array_name] = _NpyStreamWriter(
                output_dir / f"{source_name}_{array_name}.npy", dtype, shapes[array_name]
            )

    parsed_count = 0
    unique_keys: set[tuple[str, int]] = set()
    source_layout: dict[str, dict[str, Any]] = {}
    cache_drop_advice_success_count = 0
    parse_start = _phase_snapshot()
    group_members: dict[str, list[dict[str, Any]]] = {
        "train": sorted(membership["memberships"]["fit_train"], key=lambda row: row["row_group"]),
        "valid": sorted(
            membership["memberships"]["fit_valid"] + membership["memberships"]["evaluation_valid"],
            key=lambda row: row["row_group"],
        ),
    }
    for source_name, logical_name in _SOURCES.items():
        source_path = source_root / logical_name
        source_file = SequenceFile(source_path)
        if not source_file.has_targets:
            raise CacheAdmissionError(f"targets are absent from {logical_name}")
        if source_name == "valid" and not source_file.has_mask:
            raise CacheAdmissionError("valid.parquet must contain the frozen is_scored mask")
        source_layout[logical_name] = {
            "has_targets_in_parquet": source_file.has_targets,
            "has_scored_mask_in_parquet": source_file.has_mask,
            "missing_train_mask_policy": (
                "all_false_as_train_rnn_Sequences_getitem" if source_name == "train" else None
            ),
        }
        fd = os.open(source_path, os.O_RDONLY)
        try:
            rows = group_members[source_name]
            if len({int(row["row_group"]) for row in rows}) != len(rows):
                raise CacheAdmissionError(f"duplicate row group in {source_name} membership")
            for row in rows:
                key = (logical_name, int(row["row_group"]))
                if key in unique_keys:
                    raise CacheAdmissionError("a source row group would be parsed more than once")
                seq = source_file.read(int(row["row_group"]))
                if seq.seq_ix != int(row["seq_ix"]):
                    raise CacheAdmissionError(
                        f"sequence id changed at {logical_name} row group {row['row_group']}"
                    )
                if seq.targets is None:
                    raise CacheAdmissionError(
                        f"targets absent at {logical_name} row group {row['row_group']}"
                    )
                scored = seq.scored
                if scored is None:
                    if source_name != "train":
                        raise CacheAdmissionError(
                            f"scored mask absent at {logical_name} row group {row['row_group']}"
                        )
                    scored = np.zeros(len(seq.features), dtype=np.bool_)
                index = int(row["array_index"])
                for array_name, value in (
                    ("features", seq.features),
                    ("targets", seq.targets),
                    ("scored", scored),
                ):
                    writer = writers[source_name][array_name]
                    if writer._index != index:
                        raise CacheAdmissionError(
                            "membership array order differs from stream order"
                        )
                    writer.append(value)
                unique_keys.add(key)
                parsed_count += 1
                if _advise_dontneed(fd):
                    cache_drop_advice_success_count += 1
                if parsed_count % 32 == 0:
                    status["completed_sequence_reads"] = parsed_count
                    status_path.write_bytes(canonical_json_bytes(status) + b"\n")
        finally:
            os.close(fd)
    parsing_telemetry = _phase_result(parse_start)
    parsing_telemetry.update(
        {
            "parquet_file_open_count": len(_SOURCES),
            "sequence_read_invocations": parsed_count,
            "unique_source_row_groups_parsed": len(unique_keys),
            "duplicate_source_row_group_read_count": parsed_count - len(unique_keys),
            "decoded_feature_bytes": parsed_count * SEQUENCE_LENGTH * N_FEATURES * 4,
            "decoded_target_mask_bytes": parsed_count * SEQUENCE_LENGTH * (2 * 4 + 1),
            "source_group_cache_drop_advice": "POSIX_FADV_DONTNEED after each row group when supported",
            "source_group_cache_drop_advice_success_count": cache_drop_advice_success_count,
            "source_group_cache_drop_advice_attempt_count": parsed_count,
            "source_layout": source_layout,
        }
    )

    for source_name in ("train", "valid"):
        for array_name, writer in writers[source_name].items():
            array_meta[f"{source_name}_{array_name}"] = writer.close()
    write_telemetry = _phase_result(write_start)
    write_telemetry["stream_bytes_written"] = sum(row["size_bytes"] for row in array_meta.values())

    mapping_start = _phase_snapshot()
    mapping_count = 0
    mapped_bytes = 0
    sampled_bytes = 0
    for _, entry in array_meta.items():
        mapped = np.load(output_dir / entry["path"], mmap_mode="r", allow_pickle=False)
        try:
            if list(mapped.shape) != entry["shape"] or mapped.dtype.str != entry["dtype"]:
                raise CacheAdmissionError(f"written NPY header mismatch: {entry['path']}")
            mapped_bytes += int(mapped.nbytes)
            if mapped.size:
                first = np.asarray(mapped.reshape(-1)[0])
                last = np.asarray(mapped.reshape(-1)[-1])
                sampled_bytes += first.nbytes + last.nbytes
            mapping_count += 1
        finally:
            mapped._mmap.close()
    mapping_telemetry = _phase_result(mapping_start)
    mapping_telemetry.update(
        {
            "mapped_npy_file_count": mapping_count,
            "virtual_mapping_bytes": mapped_bytes,
            "sampled_element_bytes": sampled_bytes,
            "physical_io_is_reported_separately": True,
        }
    )

    if parsed_count != status["expected_sequence_reads"] or len(unique_keys) != parsed_count:
        raise CacheAdmissionError("measured parse count does not match unique selected membership")
    source_receipt_sha = sha256_file(source_receipt_path)
    cache_key_payload = {
        "schema": CACHE_SCHEMA,
        "format_version": FORMAT_VERSION,
        "fold": membership["fold"],
        "split_manifest_sha256": membership["split_manifest_sha256"],
        "portable_registry_sha256": membership["portable_registry_sha256"],
        "membership_sha256": membership_sha,
        "sources": membership["sources"],
        "feature_layout": {"sequence_length": SEQUENCE_LENGTH, "features": N_FEATURES},
        "array_dtype": {key: dtype.str for key, dtype in _ARRAY_DTYPES.items()},
        "parser_identity": {
            "producer_sha256": sha256_file(Path(__file__)),
            "sequence_reader_sha256": sha256_file(Path(data_module.__file__)),
            "feature_columns_sha256": sha256_bytes(
                canonical_json_bytes(data_module.FEATURE_COLUMNS)
            ),
        },
    }
    cache_key = sha256_bytes(canonical_json_bytes(cache_key_payload))
    manifest = {
        "schema": CACHE_SCHEMA,
        "format_version": FORMAT_VERSION,
        "cache_key": cache_key,
        "scope": "cohort",
        "fold": membership["fold"],
        "split_manifest_sha256": membership["split_manifest_sha256"],
        "portable_registry_sha256": membership["portable_registry_sha256"],
        "parser_identity": cache_key_payload["parser_identity"],
        "membership_artifact": {
            "path": "membership.json",
            "sha256": membership_sha,
            "fit_membership_roles": ["fit_train", "fit_valid"],
            "evaluation_membership_role": "evaluation_valid",
        },
        "source_verification_artifact": {
            "path": "source_verification.json",
            "sha256": source_receipt_sha,
        },
        "sources": membership["sources"],
        "member_counts": membership["member_counts"],
        "arrays": array_meta,
        "estimated_array_bytes_before_admission": estimated_bytes,
        "estimated_cache_bytes_before_admission": estimated_cache_bytes,
        "disk_admission": {
            "free_bytes_observed": free_bytes,
            "required_bytes": required_bytes,
            "array_bytes": estimated_bytes,
            "membership_copy_bytes": membership_copy_bytes,
            "metadata_reserve_bytes": _METADATA_RESERVE_BYTES,
            "estimated_cache_bytes": estimated_cache_bytes,
            "output_reserve_bytes": output_reserve_bytes,
            "checkpoint_reserve_bytes": checkpoint_reserve_bytes,
            "free_reserve_bytes": free_reserve_bytes,
        },
        "telemetry": {
            "source_verification": source_verification.get("hash_telemetry"),
            "parsing": parsing_telemetry,
            "array_write": write_telemetry,
            "mmap_validation": mapping_telemetry,
            "observed_parse_once": parsed_count == len(unique_keys) and parsed_count > 0,
            "parse_once_definition": "one SequenceFile.read call per unique source row group in this cache build",
        },
    }
    membership_bytes = membership_path.read_bytes()
    if sha256_bytes(membership_bytes) != membership_sha:
        raise CacheAdmissionError("membership artifact changed during cache build")
    (output_dir / "membership.json").write_bytes(membership_bytes)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    status.update({"state": "complete", "completed_sequence_reads": parsed_count})
    status_path.write_bytes(canonical_json_bytes(status) + b"\n")
    return manifest
