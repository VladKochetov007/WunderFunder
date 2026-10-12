#!/usr/bin/env python3
"""Validate the exact R2-frozen valid.parquet identity before any prediction load."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET_REL = Path(
    "research/campaign_2026-10-06/bandit_continuation/"
    "diversity_stack_20261009/ops_continuation_20261011/"
    "native_single_member_cleanfold_q8_r3"
)
CONTRACT_REL = PACKET_REL / "CONTRACT_SINGLE_R3.json"
SOURCE_TREE_ROOT = Path(__file__).resolve().parents[6]
IDENTITY_KEYS = (
    "bytes",
    "footer_bytes",
    "footer_sha256_including_8_byte_trailer",
)


def sha256_file(path: Path) -> str:
    """Hash a file incrementally with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject non-object receipts."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def resolve_repo_file(repo_path: str) -> Path:
    """Resolve a strict repository-relative path under the exact MAIN root."""
    relative = Path(repo_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in repo_path:
        raise ValueError(f"unsafe repository-relative path: {repo_path}")
    resolved = (SOURCE_TREE_ROOT / relative).resolve(strict=True)
    try:
        resolved.relative_to(SOURCE_TREE_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"repository path escaped MAIN root: {repo_path}") from exc
    return resolved


def walk_identity_objects(value: Any) -> Iterator[dict[str, Any]]:
    """Yield frozen footer identity objects from nested JSON receipts."""
    if isinstance(value, dict):
        if all(key in value for key in IDENTITY_KEYS):
            yield {key: value[key] for key in IDENTITY_KEYS}
        for child in value.values():
            yield from walk_identity_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_identity_objects(child)


def verify_frozen_receipts(contract: dict[str, Any]) -> None:
    """Verify current-main source receipts and their identical data identity."""
    expected = {key: contract["valid_dataset"][key] for key in IDENTITY_KEYS}
    sources = contract["valid_dataset"]["identity_receipt_sources"]
    found: dict[str, dict[str, Any]] = {}
    for name, binding in sources.items():
        path = resolve_repo_file(binding["repo_path"])
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"frozen identity receipt changed: {name}")
        receipt = read_json(path)
        candidates = list(walk_identity_objects(receipt))
        if expected not in candidates:
            raise ValueError(f"frozen valid.parquet identity differs in receipt: {name}")
        found[name] = expected
    required = {
        "shared_contract_r2",
        "shared_terminal_handoff_r2",
        "shared_root_admission_r2",
        "single_control_contract_r2",
    }
    if set(found) != required:
        raise ValueError("frozen input receipt set differs from the R3 contract")


def file_footer_identity(path: Path, expected_footer_bytes: int) -> dict[str, Any]:
    """Check Parquet magic/trailer and hash only the footer plus its 8-byte trailer."""
    before = path.stat()
    if not path.is_file() or before.st_size < expected_footer_bytes + 12:
        raise ValueError("valid.parquet is absent or smaller than its frozen footer")
    with path.open("rb") as stream:
        if stream.read(4) != b"PAR1":
            raise ValueError("valid.parquet leading magic is not PAR1")
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        declared_footer_bytes = struct.unpack("<I", trailer[:4])[0]
        if trailer[4:] != b"PAR1":
            raise ValueError("valid.parquet trailing magic is not PAR1")
        if declared_footer_bytes != expected_footer_bytes:
            raise ValueError("valid.parquet footer length differs from the frozen receipt")
        start = before.st_size - expected_footer_bytes - 8
        if start < 4:
            raise ValueError("valid.parquet footer range overlaps the leading magic")
        stream.seek(start)
        digest = hashlib.sha256()
        remaining = expected_footer_bytes + 8
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            if not block:
                raise ValueError("short read while hashing valid.parquet footer")
            digest.update(block)
            remaining -= len(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("valid.parquet changed during the footer identity check")
    return {
        "bytes": before.st_size,
        "footer_bytes": declared_footer_bytes,
        "footer_sha256_including_8_byte_trailer": digest.hexdigest(),
        "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns,
    }


def verify_metadata_only(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Read Parquet metadata only; never materialize row-group data pages."""
    import pyarrow.parquet as pq

    sys.path.insert(0, str(MAIN_ROOT))
    import utils

    main_utils = (MAIN_ROOT / "utils.py").resolve(strict=True)
    if Path(utils.__file__).resolve(strict=True) != main_utils:
        raise ValueError("metadata preflight imported utils outside exact MAIN")
    if sha256_file(main_utils) != contract["source_bindings"]["project_utils"]["sha256"]:
        raise ValueError("MAIN utils source differs from the frozen contract")
    feature_columns = utils.FEATURE_COLUMNS

    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    expected = contract["valid_dataset"]
    if metadata.num_rows != expected["rows"]:
        raise ValueError("valid.parquet row count differs from the frozen receipt")
    if parquet.num_row_groups != expected["row_groups"]:
        raise ValueError("valid.parquet row-group count differs from the frozen receipt")
    if any(metadata.row_group(index).num_rows != 20_000 for index in range(parquet.num_row_groups)):
        raise ValueError("valid.parquet has a non-complete sequence row group")
    required = {
        "seq_ix",
        "step_in_seq",
        "need_prediction",
        "is_scored",
        "t0",
        "t1",
        *feature_columns,
    }
    schema_names = set(parquet.schema.names)
    if not required <= schema_names:
        missing = sorted(required - schema_names)
        raise ValueError(f"valid.parquet is missing required columns: {missing[:8]}")
    if len(feature_columns) != contract["feature_column_count"]:
        raise ValueError("local feature-column definition differs from the frozen R2 contract")
    schema_text = str(parquet.schema)
    return {
        "rows": metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "rows_per_group": 20_000,
        "schema_names_count": len(schema_names),
        "schema_names_sha256": hashlib.sha256(
            "\n".join(sorted(schema_names)).encode("utf-8")
        ).hexdigest(),
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "required_columns_present": True,
        "feature_columns": len(feature_columns),
        "row_data_pages_read": False,
    }


def run(contract_path: Path, output_path: Path) -> dict[str, Any]:
    """Validate identity and metadata, then write one create-only receipt."""
    if MAIN_ROOT.resolve(strict=True) != MAIN_ROOT:
        raise RuntimeError("the exact MAIN repository root is unavailable")
    if Path.cwd().resolve(strict=True) != MAIN_ROOT.resolve(strict=True):
        raise RuntimeError("run this admission helper with MAIN as the working directory")
    expected_contract_path = (SOURCE_TREE_ROOT / CONTRACT_REL).resolve(strict=True)
    if contract_path.resolve(strict=True) != expected_contract_path:
        raise ValueError("R3 contract path differs from the frozen MAIN-relative path")
    contract = read_json(contract_path)
    if contract.get("schema") != "wnn.native.single-member-cleanfold-q8-contract.r3":
        raise ValueError("unexpected R3 input-gate contract schema")
    if contract.get("repository", {}).get("absolute_root") != str(MAIN_ROOT):
        raise ValueError("R3 contract is bound to another repository root")
    validator_binding = contract["source_bindings"]["input_validator"]
    validator_path = Path(__file__).resolve(strict=True)
    if validator_binding["repo_path"] != (PACKET_REL / "validate_input_identity_r3.py").as_posix():
        raise ValueError("R3 input-validator source path differs from this implementation")
    if sha256_file(validator_path) != validator_binding["sha256"]:
        raise ValueError("R3 input-validator source hash differs from its contract")
    for name, binding in contract["source_bindings"].items():
        source = resolve_repo_file(binding["repo_path"])
        if sha256_file(source) != binding["sha256"]:
            raise ValueError(f"current-main source binding changed: {name}")
    verify_frozen_receipts(contract)

    data_relative = Path(contract["valid_dataset"]["repo_path"])
    if data_relative.is_absolute() or ".." in data_relative.parts or "\\" in str(data_relative):
        raise ValueError("unsafe MAIN-relative valid dataset path")
    valid_path = (MAIN_ROOT / data_relative).resolve(strict=True)
    expected_path = (MAIN_ROOT / "datasets/valid.parquet").resolve(strict=True)
    if valid_path != expected_path:
        raise ValueError("valid.parquet resolved outside the exact MAIN data path")
    identity = file_footer_identity(valid_path, contract["valid_dataset"]["footer_bytes"])
    for key in IDENTITY_KEYS:
        if identity[key] != contract["valid_dataset"][key]:
            raise ValueError(f"current valid.parquet differs from frozen R2 identity: {key}")
    metadata = verify_metadata_only(valid_path, contract)
    after_metadata = valid_path.stat()
    if (identity["inode"], identity["bytes"], identity["mtime_ns"]) != (
        after_metadata.st_ino,
        after_metadata.st_size,
        after_metadata.st_mtime_ns,
    ):
        raise RuntimeError("valid.parquet changed during metadata inspection")

    output = output_path.resolve()
    packet = (SOURCE_TREE_ROOT / PACKET_REL).resolve(strict=True)
    try:
        output.relative_to(packet)
    except ValueError as exc:
        raise ValueError("preflight receipt must be written inside the R3 packet") from exc
    receipt = {
        "schema": "wnn.native.single-member-input-preflight.r3",
        "status": "PASS_FROZEN_FOOTER_AND_METADATA_ONLY",
        "created_utc": datetime.now(UTC).isoformat(),
        "contract_repo_path": CONTRACT_REL.as_posix(),
        "contract_sha256": sha256_file(contract_path),
        "repository_root": str(MAIN_ROOT),
        "source_tree_base_commit": "bf3ea0740f3e1a4fd4675a87ea5ba1295144bc55",
        "valid_dataset_repo_path": "datasets/valid.parquet",
        "valid_dataset_identity": identity,
        "metadata_only_validation": metadata,
        "prediction_data_loaded": False,
        "model_imported_or_constructed": False,
        "full_fold_score_performed": False,
        "serving_execution_status": "BLOCKED_PENDING_PY311_DEPENDENCIES_AND_ROOT_ADMISSION",
        "python_used_for_static_preflight": sys.version,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Run the root-side pre-inference input gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=SOURCE_TREE_ROOT / CONTRACT_REL)
    parser.add_argument(
        "--output", type=Path, default=SOURCE_TREE_ROOT / PACKET_REL / "INPUT_PREFLIGHT_R3.json"
    )
    args = parser.parse_args()
    receipt = run(args.contract, args.output)
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
