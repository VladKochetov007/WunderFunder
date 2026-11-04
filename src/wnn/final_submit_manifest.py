"""Source-bound manifest contract for the explicit all-labeled-data final fit."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from wnn.data import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS
from wnn.splits import file_sha256

FINAL_FIT_SCHEMA = "wnn.final-fit-membership.v1"
FINAL_FIT_POLICY = "all_official_train_and_valid_labels_fit_no_independent_score_v1"
TRAIN_SEQUENCES = 10_607
VALID_SEQUENCES = 1_873
TOTAL_SEQUENCES = TRAIN_SEQUENCES + VALID_SEQUENCES
SEQUENCE_ROWS = 20_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_MANIFEST_BYTES = 2_000_000


@dataclass(frozen=True)
class FinalFitMembership:
    """Typed final-fit membership, kept separate from chronological split records."""

    path: Path
    manifest_sha256: str
    fit_members_sha256: str
    train_groups: tuple[int, ...]
    valid_groups: tuple[int, ...]
    train_sha256: str
    valid_sha256: str
    source_validation_sha256: str


def _member_identity(groups: dict[str, list[int]]) -> str:
    """Hash ordered, source-qualified whole-sequence row-group membership."""
    members = [[source, group] for source in ("train", "valid") for group in groups[source]]
    encoded = json.dumps(members, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_final_fit_membership(
    train_sha256: str, valid_sha256: str, source_validation_sha256: str
) -> dict[str, object]:
    """Bind complete labeled source coverage without assigning an evaluation set."""
    if not all(
        isinstance(value, str) and _SHA256.fullmatch(value)
        for value in (train_sha256, valid_sha256, source_validation_sha256)
    ):
        raise ValueError(
            "final fit requires lowercase SHA256 source and validation receipt identities"
        )
    groups = {
        "train": list(range(TRAIN_SEQUENCES)),
        "valid": list(range(VALID_SEQUENCES)),
    }
    return {
        "schema": FINAL_FIT_SCHEMA,
        "policy": FINAL_FIT_POLICY,
        "version": 1,
        "sources": {
            "train": {
                "path": "datasets/train.parquet",
                "row_groups": TRAIN_SEQUENCES,
                "sha256": train_sha256,
            },
            "valid": {
                "path": "datasets/valid.parquet",
                "row_groups": VALID_SEQUENCES,
                "sha256": valid_sha256,
            },
        },
        "source_validation": {
            "receipt_sha256": source_validation_sha256,
            "checked_member": ["train", 528],
            "rows": SEQUENCE_ROWS,
            "decoded_feature_and_target_columns": 114,
            "all_decoded_values_finite": True,
        },
        "fit_members": groups,
        "fit_sequences": TOTAL_SEQUENCES,
        "fit_members_sha256": _member_identity(groups),
        "membership_basis": "complete source row-group ordinals; each row group is one whole sequence",
        "feature_assembly": "causal within-sequence features only; never sort or infer time from seq_ix",
        "primary_validation_role": "all official-valid labels are reused for fitting",
        "independent_evaluation_members": [],
        "independent_score_claim": "forbidden",
        "post_fit_primary_valid_scoring": "forbidden",
        "source_coverage": "receiver must verify both source row-group counts and complete membership against the bound parquet identities",
    }


def validate_final_fit_membership(
    value: object, train_sha256: str, valid_sha256: str, source_validation_sha256: str
) -> dict[str, object]:
    """Fail closed unless source identity and complete membership match the contract."""
    if not isinstance(value, dict) or value.get("schema") != FINAL_FIT_SCHEMA:
        raise ValueError("final fit requires its dedicated membership schema")
    expected = build_final_fit_membership(train_sha256, valid_sha256, source_validation_sha256)
    if value.get("sources") != expected["sources"]:
        raise ValueError("final fit source identities differ from the receiver-bound parquets")
    if value != expected:
        raise ValueError("final fit membership differs from its frozen all-labeled-data contract")
    return value


def load_final_fit_membership(
    path: Path, root: Path, expected_source_validation_sha256: str
) -> FinalFitMembership:
    """Verify source files and return complete final-fit membership with no eval partition."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("final-fit manifest must be a regular file, not a symlink")
    if not isinstance(expected_source_validation_sha256, str) or not _SHA256.fullmatch(
        expected_source_validation_sha256
    ):
        raise ValueError("expected source-validation receipt identity is not a lowercase SHA256")
    resolved = path.resolve()
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_MANIFEST_BYTES:
        raise ValueError("final-fit manifest exceeds its bounded read size")
    raw = resolved.read_bytes()
    if len(raw) != size:
        raise ValueError("final-fit manifest changed while being read")
    value = json.loads(raw)
    sources = value.get("sources") if isinstance(value, dict) else None
    source_validation = value.get("source_validation") if isinstance(value, dict) else None
    if not isinstance(sources, dict) or not isinstance(source_validation, dict):
        raise ValueError("final-fit manifest omits source or validation identities")
    if source_validation.get("receipt_sha256") != expected_source_validation_sha256:
        raise ValueError("final-fit source-validation receipt identity changed")
    train_record, valid_record = sources.get("train"), sources.get("valid")
    if not isinstance(train_record, dict) or not isinstance(valid_record, dict):
        raise ValueError("final-fit manifest source records are malformed")
    train_sha, valid_sha = train_record.get("sha256"), valid_record.get("sha256")
    if not isinstance(train_sha, str) or not isinstance(valid_sha, str):
        raise ValueError("final-fit manifest source hashes are malformed")
    validate_final_fit_membership(value, train_sha, valid_sha, expected_source_validation_sha256)
    root = root.resolve()
    for split, expected_groups in (("train", TRAIN_SEQUENCES), ("valid", VALID_SEQUENCES)):
        logical_path = f"datasets/{split}.parquet"
        source = root / logical_path
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"final-fit source is missing or unsafe: {logical_path}")
        if file_sha256(source) != sources[split]["sha256"]:
            raise ValueError(f"final-fit source SHA256 changed: {logical_path}")
        parquet = pq.ParquetFile(source)
        if parquet.metadata.num_row_groups != expected_groups:
            raise ValueError(f"final-fit source row-group count changed: {logical_path}")
        if not set(FEATURE_COLUMNS + list(TARGET_COLUMNS)).issubset(parquet.schema_arrow.names):
            raise ValueError(f"final-fit source lacks labeled feature columns: {logical_path}")
        if any(
            parquet.metadata.row_group(group).num_rows != SEQUENCE_LENGTH
            for group in range(expected_groups)
        ):
            raise ValueError(f"final-fit source contains an incomplete sequence: {logical_path}")
    member_groups = value["fit_members"]
    return FinalFitMembership(
        path=resolved,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        fit_members_sha256=value["fit_members_sha256"],
        train_groups=tuple(member_groups["train"]),
        valid_groups=tuple(member_groups["valid"]),
        train_sha256=sources["train"]["sha256"],
        valid_sha256=sources["valid"]["sha256"],
        source_validation_sha256=source_validation["receipt_sha256"],
    )
