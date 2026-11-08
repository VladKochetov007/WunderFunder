"""Source audit and whole-sequence membership guards for evaluation probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from wnn.data import FEATURE_COLUMNS, SEQUENCE_LENGTH

MAX_PROBE_SEQUENCES = 384
AUDIT_SCHEMA = "tabdata.memory-and-storage-audit.v1"
REQUIRED_VALID_SCHEMA = {"seq_ix", "step_in_seq", "is_scored", "t0", "t1"}


def _validate_audit_record(
    audit: object, actual_bytes: int, row_count: int, row_groups: int
) -> dict[str, Any]:
    """Bind the local valid parquet footer to the exact complete-transfer audit."""
    if not isinstance(audit, dict) or audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError("source audit is not the current tabdata server audit")
    data = audit.get("data")
    if not isinstance(data, dict):
        raise ValueError("source audit has no data identity record")
    pin = data.get("valid_sha256_from_complete_transfer_receipt")
    expected_bytes = data.get("valid_bytes")
    expected_rows = data.get("valid_fresh_footer_rows")
    expected_groups = data.get("valid_fresh_footer_row_groups")
    schema = data.get("valid_fresh_schema_includes")
    if (
        not isinstance(pin, str)
        or len(pin) != 64
        or any(char not in "0123456789abcdef" for char in pin)
        or type(expected_bytes) is not int
        or type(expected_rows) is not int
        or type(expected_groups) is not int
        or data.get("valid_fresh_full_hash") is not False
        or not isinstance(schema, list)
        or not REQUIRED_VALID_SCHEMA.issubset(schema)
        or "112 input features" not in schema
    ):
        raise ValueError("source audit lacks the expected complete valid transfer receipt")
    if (actual_bytes, row_count, row_groups) != (expected_bytes, expected_rows, expected_groups):
        raise ValueError("local official valid parquet size or footer differs from the audit")
    if row_count != row_groups * SEQUENCE_LENGTH:
        raise ValueError("official valid row groups are not complete fixed-length sequences")
    return {
        "sha256_receipt_pin": pin,
        "bytes": expected_bytes,
        "rows": expected_rows,
        "row_groups": expected_groups,
        "sha256_verification": (
            "reused exact complete-transfer receipt; local full-file rehash not repeated"
        ),
    }


def _read_valid_source(data_root: Path, audit_path: Path) -> tuple[dict[str, Any], str]:
    """Check the valid parquet footer and source audit without touching train data."""
    audit_path = audit_path.resolve()
    audit_size = audit_path.stat().st_size
    if not 0 < audit_size <= 2_000_000:
        raise ValueError("source audit must be a nonempty JSON file no larger than 2 MB")
    audit_raw = audit_path.read_bytes()
    if len(audit_raw) != audit_size:
        raise ValueError("source audit changed while being read")
    audit = json.loads(audit_raw)
    valid_path = data_root.resolve() / "datasets" / "valid.parquet"
    if not valid_path.is_file():
        raise FileNotFoundError(f"official validation source is missing: {valid_path}")
    metadata = pq.ParquetFile(valid_path).metadata
    required_columns = set(FEATURE_COLUMNS) | REQUIRED_VALID_SCHEMA
    if not required_columns.issubset(metadata.schema.names):
        raise ValueError("local official valid parquet is missing audited model inputs or labels")
    identity = _validate_audit_record(
        audit, valid_path.stat().st_size, metadata.num_rows, metadata.num_row_groups
    )
    for row_group in range(metadata.num_row_groups):
        if metadata.row_group(row_group).num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"valid row group {row_group} is not one full sequence")
    return {
        "path": str(valid_path),
        "audit_path": str(audit_path),
        "audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "identity": identity,
    }, str(valid_path)


def _select_valid_members(row_groups: int, count: int) -> list[tuple[str, int]]:
    """Select whole physical groups without treating their order as chronology."""
    if type(count) is not int or not 1 <= count <= MAX_PROBE_SEQUENCES:
        raise ValueError(f"max sequences must be in [1, {MAX_PROBE_SEQUENCES}]")
    if type(row_groups) is not int or row_groups < count:
        raise ValueError("requested validation subset exceeds official valid membership")
    # Physical row-group order is a deterministic sample, never a time key.
    return [("valid", group) for group in range(count)]
