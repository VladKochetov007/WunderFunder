"""Build small, source-qualified sequence indexes for chronological validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.chronological_splits import _footer_identity
from wnn.splits import file_sha256

SEQUENCE_LENGTH = 20_000
SOURCE_AUDIT_PATH = Path("cache/source_audit.json")
SOURCE_AUDIT_SCHEMA = "wnn.source-audit.v1"
SOURCE_AUDIT_FIELDS = {
    "bytes",
    "sha256",
    "mtime_ns",
    "footer_bytes",
    "footer_sha256",
    "row_group_count",
    "sequence_length",
    "has_is_scored",
}


@dataclass(frozen=True)
class SourcePin:
    """Expected byte identity and complete-sequence coverage for one official file."""

    split: str
    byte_count: int
    sha256: str
    row_group_count: int
    has_scoring_mask: bool


class SourceRowGroupReadError(RuntimeError):
    """Identify an unreadable row group by its source file and source-local index."""

    def __init__(self, split: str, row_group: int, source: str, detail: str) -> None:
        super().__init__(split, row_group, source, detail)
        self.split = split
        self.row_group = row_group
        self.source = source
        self.detail = detail

    def __str__(self) -> str:
        return f"{self.source} row_group={self.row_group} ({self.split}): {self.detail}"


OFFICIAL_SOURCE_PINS = {
    "train": SourcePin(
        "train",
        29_240_219_011,
        "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
        10_607,
        False,
    ),
    "valid": SourcePin(
        "valid",
        5_615_509_128,
        "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
        1_873,
        True,
    ),
}


def _index_map_sha256(row_group: np.ndarray, seq_ix: np.ndarray) -> str:
    """Hash the stable ordered row-group and sequence identifier map."""
    digest = hashlib.sha256()
    for values in (row_group, seq_ix):
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    return digest.hexdigest()


def _read_sequence_ids(
    path: Path, split: str, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read only sequence identifiers and require every parquet group to be complete."""
    parquet = pq.ParquetFile(path)
    if "seq_ix" not in parquet.schema_arrow.names:
        raise ValueError(f"source has no seq_ix column: {path}")
    groups = np.arange(parquet.num_row_groups, dtype=np.int64)
    sequence_ids = np.empty(parquet.num_row_groups, dtype=np.int64)
    for row_group in groups:
        try:
            table = parquet.read_row_group(int(row_group), columns=["seq_ix"], use_threads=False)
            values = np.asarray(table.column("seq_ix").to_numpy())
        except Exception as exc:
            raise SourceRowGroupReadError(
                split, int(row_group), str(path), f"{type(exc).__name__}: {exc}"
            ) from exc
        if table.num_rows != sequence_length:
            raise SourceRowGroupReadError(
                split,
                int(row_group),
                str(path),
                f"has {table.num_rows} rows; expected {sequence_length}",
            )
        if values.ndim != 1 or values.dtype.kind not in "iu" or not np.all(values == values[0]):
            raise SourceRowGroupReadError(
                split, int(row_group), str(path), "seq_ix is not one constant integer sequence"
            )
        sequence_ids[row_group] = int(values[0])
    unique, counts = np.unique(sequence_ids, return_counts=True)
    duplicate_ids = unique[counts > 1]
    if len(duplicate_ids):
        duplicate_id = int(duplicate_ids[0])
        duplicate_group = int(np.flatnonzero(sequence_ids == duplicate_id)[-1])
        raise SourceRowGroupReadError(
            split, duplicate_group, str(path), f"duplicate seq_ix value {duplicate_id}"
        )
    return groups, sequence_ids


def _write_index(path: Path, row_group: np.ndarray, seq_ix: np.ndarray) -> None:
    """Write only the two integer arrays needed by the split producer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != {"row_group", "seq_ix"}:
                raise FileExistsError(f"refusing to replace non-thin source index: {path}")
            if not np.array_equal(existing["row_group"], row_group) or not np.array_equal(
                existing["seq_ix"], seq_ix
            ):
                raise FileExistsError(f"refusing to replace changed source index: {path}")
        return
    with path.open("xb") as stream:
        np.savez(stream, row_group=row_group, seq_ix=seq_ix)


def _write_identity(path: Path, identity: dict[str, object]) -> None:
    """Write a stable JSON receipt, refusing to replace different source evidence."""
    encoded = (json.dumps(identity, sort_keys=True, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace changed source identity receipt: {path}")
        return
    with path.open("xb") as stream:
        stream.write(encoded)


def _load_source_audit(root: Path) -> tuple[dict[str, object], str]:
    """Load the bounded root-issued transfer audit and return its raw-byte hash."""
    audit_path = root / SOURCE_AUDIT_PATH
    size = audit_path.stat().st_size
    if size <= 0 or size > 2_000_000:
        raise ValueError("root-issued source audit must be between 1 byte and 2 MB")
    raw = audit_path.read_bytes()
    if len(raw) != size:
        raise ValueError("root-issued source audit changed while being read")
    audit = json.loads(raw)
    if not isinstance(audit, dict) or set(audit) != {"schema", "sources"}:
        raise ValueError("root-issued source audit has unsupported fields")
    if audit.get("schema") != SOURCE_AUDIT_SCHEMA:
        raise ValueError("root-issued source audit has an unsupported schema")
    sources = audit.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "datasets/train.parquet",
        "datasets/valid.parquet",
    }:
        raise ValueError("root-issued source audit must bind the official train and valid files")
    for name, identity in sources.items():
        if not isinstance(identity, dict) or set(identity) != SOURCE_AUDIT_FIELDS:
            raise ValueError(f"invalid root-issued source audit identity for {name}")
    return audit, hashlib.sha256(raw).hexdigest()


def _verify_source_audit_entry(
    root: Path, split: str, identity: dict[str, object], sequence_length: int
) -> None:
    """Match a root-issued full-file identity to current bounded parquet metadata."""
    source = root / "datasets" / f"{split}.parquet"
    portable = _verify_portable_source_audit(root, split, identity)
    stat = source.stat()
    footer = _footer_identity(source)
    parquet = pq.ParquetFile(source)
    bad_groups = [
        group
        for group in range(parquet.num_row_groups)
        if parquet.metadata.row_group(group).num_rows != sequence_length
    ]
    if bad_groups:
        raise SourceRowGroupReadError(
            split,
            bad_groups[0],
            str(source),
            f"metadata row count is not the expected {sequence_length}",
        )
    actual = {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "footer_bytes": footer["footer_bytes"],
        "footer_sha256": footer["footer_sha256"],
        "row_group_count": parquet.num_row_groups,
        "sequence_length": sequence_length,
        "has_is_scored": "is_scored" in parquet.schema_arrow.names,
    }
    compared = {key: value for key, value in actual.items() if not (portable and key == "mtime_ns")}
    if any(identity.get(key) != value for key, value in compared.items()):
        raise ValueError(f"root-issued source audit metadata no longer matches {source}")
    if not isinstance(identity.get("sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", str(identity["sha256"])
    ):
        raise ValueError(f"root-issued source audit has an invalid SHA-256 for {source}")


def _verify_portable_source_audit(
    root: Path, split: str, audit_identity: dict[str, object]
) -> bool:
    """Verify relocated parquet bytes against the sealed registry before ignoring mtime."""
    registry_name = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    if not registry_name:
        return False
    source_name = f"datasets/{split}.parquet"
    registry = Path(registry_name)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    sources = payload.get("sources") if isinstance(payload, dict) else None
    entry = sources.get(source_name) if isinstance(sources, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"portable source registry is missing {source_name}")
    original_identity, sealed_sha256 = entry.get("original_identity"), entry.get("sha256")
    if not isinstance(original_identity, dict) or sealed_sha256 != audit_identity.get("sha256"):
        raise ValueError(f"portable source registry disagrees with root-issued audit for {split}")
    if any(audit_identity.get(key) != value for key, value in original_identity.items()):
        raise ValueError(f"portable source identity disagrees with root-issued audit for {split}")
    from wnn.portability import verify_portable_source

    verify_portable_source(root / source_name, source_name, original_identity, registry)
    return True


def _write_index_failure_receipt(
    root: Path,
    split: str,
    source_sha256: str,
    source_audit_sha256: str,
    error: SourceRowGroupReadError,
) -> None:
    """Persist the exact row-group read failure and its root-audit binding."""
    _write_identity(
        root / "cache" / f"{split}_index.failure.json",
        {
            "schema": "wnn.official-source-index-failure.v1",
            "split": split,
            "source": f"datasets/{split}.parquet",
            "source_bytes": (root / "datasets" / f"{split}.parquet").stat().st_size,
            "source_sha256": source_sha256,
            "source_audit_sha256": source_audit_sha256,
            "row_group": error.row_group,
            "error": error.detail,
        },
    )


def build_source_index(
    root: Path,
    split: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
    expected_row_groups: int,
    sequence_length: int = SEQUENCE_LENGTH,
) -> Path:
    """Validate one pinned parquet and create its thin row-group/seq_ix index.

    The caller supplies the full source SHA-256 already verified by the source
    transfer audit. This builder checks the byte count and footer identity, then
    reads only ``seq_ix``. No target, scoring-mask, or feature values enter the
    index.
    """
    if split not in OFFICIAL_SOURCE_PINS:
        raise ValueError("split must be train or valid")
    if expected_bytes <= 0 or expected_row_groups <= 0 or sequence_length <= 0:
        raise ValueError("source byte, row-group, and sequence-length pins must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")

    root = root.resolve()
    audit, source_audit_sha256 = _load_source_audit(root)
    audit_entry = audit["sources"][f"datasets/{split}.parquet"]
    if not isinstance(audit_entry, dict):
        raise ValueError(f"root-issued source audit identity is invalid for {split}")
    expected_audit = {
        "bytes": expected_bytes,
        "sha256": expected_sha256,
        "row_group_count": expected_row_groups,
        "sequence_length": sequence_length,
        "has_is_scored": OFFICIAL_SOURCE_PINS[split].has_scoring_mask,
    }
    if any(audit_entry.get(key) != value for key, value in expected_audit.items()):
        raise ValueError(f"root-issued source audit does not match the supplied {split} source pin")
    try:
        _verify_source_audit_entry(root, split, audit_entry, sequence_length)
    except SourceRowGroupReadError as exc:
        _write_index_failure_receipt(root, split, expected_sha256, source_audit_sha256, exc)
        raise
    source = root / "datasets" / f"{split}.parquet"
    before = source.stat()
    if before.st_size != expected_bytes:
        raise ValueError(f"{split} parquet byte count differs from the supplied source pin")
    footer_before = _footer_identity(source)
    if before.st_size != footer_before["bytes"] or before.st_mtime_ns != footer_before["mtime_ns"]:
        raise ValueError(f"{split} parquet changed during source identity validation")

    parquet = pq.ParquetFile(source)
    if parquet.num_row_groups != expected_row_groups:
        raise ValueError(f"{split} parquet row-group count differs from the supplied source pin")
    mask_present = "is_scored" in parquet.schema_arrow.names
    if mask_present != OFFICIAL_SOURCE_PINS[split].has_scoring_mask:
        raise ValueError(f"{split} parquet scoring-mask schema differs from the official source")
    try:
        row_group, seq_ix = _read_sequence_ids(source, split, sequence_length)
    except SourceRowGroupReadError as exc:
        _write_index_failure_receipt(root, split, expected_sha256, source_audit_sha256, exc)
        raise
    after = source.stat()
    footer_after = _footer_identity(source)
    if (
        after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or footer_after != footer_before
    ):
        raise ValueError(f"{split} parquet changed while its sequence index was built")

    index_path = root / "cache" / f"{split}_index.npz"
    _write_index(index_path, row_group, seq_ix)
    index_stat = index_path.stat()
    identity = {
        "schema": "wnn.official-source-index.v1",
        "split": split,
        "source": f"datasets/{split}.parquet",
        "source_bytes": expected_bytes,
        "source_sha256": expected_sha256,
        "source_audit": SOURCE_AUDIT_PATH.as_posix(),
        "source_audit_sha256": source_audit_sha256,
        "source_mtime_ns": audit_entry["mtime_ns"],
        "source_footer_bytes": footer_before["footer_bytes"],
        "source_footer_sha256": footer_before["footer_sha256"],
        "row_group_count": len(row_group),
        "sequence_length": sequence_length,
        "has_is_scored": mask_present,
        "index": f"cache/{split}_index.npz",
        "index_bytes": index_stat.st_size,
        "index_sha256": file_sha256(index_path),
        "index_map_sha256": _index_map_sha256(row_group, seq_ix),
    }
    identity_path = root / "cache" / f"{split}_index.identity.json"
    _write_identity(identity_path, identity)
    return index_path


def validate_source_index_receipt(
    root: Path,
    split: str,
    expected_source_audit_sha256: str,
    *,
    require_official_pin: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one index receipt against the root audit, metadata, and row coverage."""
    if split not in OFFICIAL_SOURCE_PINS:
        raise ValueError("split must be train or valid")
    root = root.resolve()
    pin = OFFICIAL_SOURCE_PINS[split]
    audit, source_audit_sha256 = _load_source_audit(root)
    if source_audit_sha256 != expected_source_audit_sha256:
        raise ValueError("source audit bytes differ from the pinned receipt")
    audit_entry = audit["sources"][f"datasets/{split}.parquet"]
    if not isinstance(audit_entry, dict):
        raise ValueError(f"root-issued source audit identity is invalid for {split}")
    sequence_length = audit_entry.get("sequence_length")
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError(f"root-issued source audit has invalid sequence length for {split}")
    _verify_source_audit_entry(root, split, audit_entry, sequence_length)
    if require_official_pin and (
        audit_entry.get("bytes") != pin.byte_count
        or audit_entry.get("sha256") != pin.sha256
        or audit_entry.get("row_group_count") != pin.row_group_count
        or sequence_length != SEQUENCE_LENGTH
        or audit_entry.get("has_is_scored") != pin.has_scoring_mask
    ):
        raise ValueError(f"root-issued source audit differs from official {split} identity")

    index_path = root / "cache" / f"{split}_index.npz"
    receipt_path = root / "cache" / f"{split}_index.identity.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    parquet_path = root / "datasets" / f"{split}.parquet"
    footer = _footer_identity(parquet_path)
    expected_receipt = {
        "schema": "wnn.official-source-index.v1",
        "split": split,
        "source": f"datasets/{split}.parquet",
        "source_bytes": audit_entry["bytes"],
        "source_sha256": audit_entry["sha256"],
        "source_audit": SOURCE_AUDIT_PATH.as_posix(),
        "source_audit_sha256": source_audit_sha256,
        "source_mtime_ns": audit_entry["mtime_ns"],
        "source_footer_bytes": footer["footer_bytes"],
        "source_footer_sha256": footer["footer_sha256"],
        "row_group_count": audit_entry["row_group_count"],
        "sequence_length": sequence_length,
        "has_is_scored": audit_entry["has_is_scored"],
        "index": f"cache/{split}_index.npz",
        "index_bytes": index_path.stat().st_size,
        "index_sha256": file_sha256(index_path),
    }
    if set(receipt) != set(expected_receipt) | {"index_map_sha256"}:
        raise ValueError(f"{split} source index receipt has unsupported fields")
    for key, value in expected_receipt.items():
        if receipt.get(key) != value:
            raise ValueError(f"{split} source index receipt disagrees with root audit: {key}")
    with np.load(index_path, allow_pickle=False) as index:
        if set(index.files) != {"row_group", "seq_ix"}:
            raise ValueError(f"{split} source index must contain only row_group and seq_ix")
        row_group = np.asarray(index["row_group"])
        seq_ix = np.asarray(index["seq_ix"])
    expected_count = int(audit_entry["row_group_count"])
    if (
        row_group.dtype.kind not in "iu"
        or seq_ix.dtype.kind not in "iu"
        or row_group.shape != (expected_count,)
        or seq_ix.shape != row_group.shape
        or not np.array_equal(row_group, np.arange(expected_count))
        or len(np.unique(seq_ix)) != len(seq_ix)
        or receipt.get("index_map_sha256") != _index_map_sha256(row_group, seq_ix)
    ):
        raise ValueError(f"{split} thin source index has incomplete or invalid sequence coverage")
    return row_group.astype(np.int64, copy=False), seq_ix.astype(np.int64, copy=False)


def verify_source_index_receipt(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Verify the canonical official pin and return its complete thin index."""
    _, source_audit_sha256 = _load_source_audit(root.resolve())
    return validate_source_index_receipt(
        root, split, source_audit_sha256, require_official_pin=True
    )
