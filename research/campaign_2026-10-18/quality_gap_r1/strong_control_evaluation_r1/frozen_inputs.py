"""Validate the frozen splits, compact bundle and validation-source identity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq
from pins import (
    BUNDLE_SHA256,
    CANONICAL_SPLIT_SHA256,
    DIAGNOSTIC_INPUTS_SHA256,
    EVALUATION_COMPONENTS,
    EVALUATION_GROUPS,
    MEMBERSHIP_COMPARISON_SHA256,
    OLD_SPLIT_SHA256,
    Q8_FULL_GRADE_SHA256,
    ROW_GROUPS,
    ROWS_PER_GROUP,
    SELECTED_ROWS,
    VALID_PARQUET_BYTES,
    VALID_PARQUET_FOOTER_BYTES,
    VALID_PARQUET_FOOTER_SHA256,
    VALID_PARQUET_MTIME_NS,
    VALID_PARQUET_SHA256,
)
from source_alignment import read_pinned_json, resolve_manifest_path, sha256_array, sha256_file


def validate_split_membership(
    repo_root: Path,
) -> tuple[
    dict[int, dict[str, int]],
    list[int],
    dict[str, object],
    dict[str, object],
    Path,
    Path,
    Path,
]:
    """Verify old/canonical split identity and return exact group sequence/component mapping."""
    old_path = repo_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    canonical_path = (
        repo_root
        / "research/campaign_2026-10-18/quality_gap_r1/export_preparation/CANONICAL_F0_REMOTE.json"
    )
    comparison_path = (
        repo_root
        / "research/campaign_2026-10-18/quality_gap_r1/export_preparation/F0_MEMBERSHIP_COMPARISON.json"
    )
    old = read_pinned_json(old_path, OLD_SPLIT_SHA256, "checkpoint-pinned old split")
    canonical = read_pinned_json(canonical_path, CANONICAL_SPLIT_SHA256, "canonical current split")
    comparison = read_pinned_json(
        comparison_path, MEMBERSHIP_COMPARISON_SHA256, "F0 membership comparison"
    )
    if (
        comparison.get("status")
        != "PASS_EXACT_ORDERED_MEMBERSHIP_AND_COMPONENT_MAP; SPLIT_MANIFEST_IDENTITIES_DIFFER"
    ):
        raise ValueError("membership comparison does not record the exact expected pass status")
    inputs = cast(dict[str, dict[str, object]], comparison["inputs"])
    if inputs["main_frozen_split"].get("checkpoint_pinned_sha256") != OLD_SPLIT_SHA256:
        raise ValueError("membership comparison does not pin the checkpoint's old split")
    if inputs["canonical_current_f0"].get("canonical_manifest_sha256") != CANONICAL_SPLIT_SHA256:
        raise ValueError("membership comparison does not pin the canonical split")
    receipt_checks = cast(dict[str, object], comparison["comparison"])
    membership_checks = cast(dict[str, dict[str, object]], receipt_checks["membership_lists"])
    for name in (
        "fit.train",
        "fit.valid",
        "evaluation.valid",
        "excluded.train",
        "excluded.valid",
        "excluded.corrupt_train",
    ):
        if membership_checks[name].get("ordered_equal") is not True:
            raise ValueError(f"membership receipt does not pass ordered comparison for {name}")
    sequence_check = cast(dict[str, object], receipt_checks["sequences_component_mapping"])
    if sequence_check.get("ordered_or_keyed_value_equal") is not True:
        raise ValueError("membership receipt does not pass component-map comparison")
    old_eval = cast(dict[str, list[int]], old["evaluation"])["valid"]
    current_eval = cast(dict[str, list[int]], canonical["evaluation"])["valid"]
    old_valid = cast(dict[str, list[dict[str, int]]], old["sequences"])["valid"]
    current_valid = cast(dict[str, list[dict[str, int]]], canonical["sequences"])["valid"]
    if old_eval != current_eval or old_valid != current_valid:
        raise ValueError("current canonical groups or component map differ from checkpoint split")
    if len(old_eval) != EVALUATION_GROUPS or len(set(old_eval)) != EVALUATION_GROUPS:
        raise ValueError("old split evaluation membership has an unexpected size or duplicates")
    if len(old_valid) != ROW_GROUPS:
        raise ValueError("old split does not identify every validation row group")
    identities: dict[int, dict[str, int]] = {}
    for item in old_valid:
        group = int(item["row_group"])
        if group in identities:
            raise ValueError("old split contains duplicate validation row groups")
        identities[group] = {"seq_ix": int(item["seq_ix"]), "component": int(item["component"])}
    if set(identities) != set(range(ROW_GROUPS)):
        raise ValueError("old split row-group identities are not the full validation source")
    if not set(old_eval).issubset(identities):
        raise ValueError("evaluation membership references an unknown row group")
    if len({identities[group]["component"] for group in old_eval}) != EVALUATION_COMPONENTS:
        raise ValueError("evaluation component coverage differs from the frozen bundle")
    return identities, old_eval, old, comparison, old_path, canonical_path, comparison_path


def validate_diagnostic_manifest(
    path: Path,
) -> tuple[dict[str, object], dict[str, object], Path, dict[str, object], Path]:
    """Verify the frozen diagnostic manifest and resolve the pinned compact bundle."""
    manifest = read_pinned_json(path, DIAGNOSTIC_INPUTS_SHA256, "frozen diagnostic inputs")
    if manifest.get("schema") != "wnn.quality_gap.inputs.v1":
        raise ValueError("unsupported frozen diagnostic manifest schema")
    base = path.resolve().parent
    bundle = cast(dict[str, object], manifest["bundle"])
    if bundle.get("sha256") != BUNDLE_SHA256:
        raise ValueError("diagnostic bundle pin differs from frozen source")
    bundle_path = resolve_manifest_path(base, bundle["path"])
    if (
        bundle_path.stat().st_size != int(bundle["bytes"])
        or sha256_file(bundle_path) != BUNDLE_SHA256
    ):
        raise ValueError("compact scored bundle bytes or SHA-256 mismatch")
    receipts = cast(dict[str, dict[str, object]], manifest["receipts"])
    q8_receipt_path = resolve_manifest_path(base, receipts["Q8_FULL_F0"]["path"])
    q8_receipt = read_pinned_json(q8_receipt_path, Q8_FULL_GRADE_SHA256, "full-panel Q8 grade")
    if (
        q8_receipt.get("status") != "passed_panel_check"
        or q8_receipt.get("panel_sha256") != VALID_PARQUET_SHA256
        or q8_receipt.get("panel_sha256_scope") != "full input file"
        or q8_receipt.get("panel_bytes") != VALID_PARQUET_BYTES
        or q8_receipt.get("panel_mtime_ns_after") != VALID_PARQUET_MTIME_NS
    ):
        raise ValueError("full-panel Q8 receipt does not inherit the pinned validation source")
    return manifest, bundle, bundle_path, q8_receipt, q8_receipt_path


def load_bundle_arrays(bundle: dict[str, object], path: Path) -> dict[str, np.ndarray]:
    """Load and verify the compact labels and row identities used by existing diagnostics."""
    keys = cast(dict[str, str | None], bundle["keys"])
    names = {"target": "target", "row_group": "row_group", "component": "component"}
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in names:
            key = keys[name]
            if key is None or key not in archive.files:
                raise ValueError(f"compact bundle lacks required array: {name}")
            arrays[name] = np.asarray(archive[key])
    hashes = cast(dict[str, str], bundle["array_sha256"])
    for name, array in arrays.items():
        if sha256_array(array) != hashes[name]:
            raise ValueError(f"compact bundle array SHA-256 mismatch: {name}")
    if arrays["target"].shape != (SELECTED_ROWS, 2) or arrays["target"].dtype != np.float32:
        raise ValueError("compact labels must be float32 with the frozen selected-row shape")
    for name in ("row_group", "component"):
        if arrays[name].shape != (SELECTED_ROWS,) or arrays[name].dtype != np.int64:
            raise ValueError(f"compact {name} must be int64 and aligned to all selected rows")
    if not np.isfinite(arrays["target"]).all():
        raise ValueError("compact labels contain nonfinite values")
    return arrays


def validate_parquet_source(
    path: Path, old_fold: dict[str, object], q8_receipt: dict[str, object]
) -> pq.ParquetFile:
    """Verify inherited full-file identity, a fresh footer digest, and every group size."""
    stat = path.stat()
    if stat.st_size != VALID_PARQUET_BYTES or stat.st_mtime_ns != VALID_PARQUET_MTIME_NS:
        raise ValueError(
            "validation parquet size or mtime differs from the pinned full-panel grade"
        )
    source = cast(dict[str, dict[str, object]], old_fold["sources"])["datasets/valid.parquet"]
    footer_bytes = int(source["footer_bytes"])
    if (
        footer_bytes != VALID_PARQUET_FOOTER_BYTES
        or source["footer_sha256"] != VALID_PARQUET_FOOTER_SHA256
    ):
        raise ValueError("old split does not carry the expected validation parquet footer identity")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(-footer_bytes - 8, 2)
        remaining = footer_bytes + 8
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("validation parquet footer is truncated")
            digest.update(block)
            remaining -= len(block)
    if digest.hexdigest() != VALID_PARQUET_FOOTER_SHA256:
        raise ValueError("fresh validation parquet footer SHA-256 mismatch")
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    if metadata.num_row_groups != ROW_GROUPS or metadata.num_rows != ROW_GROUPS * ROWS_PER_GROUP:
        raise ValueError("validation parquet row-group structure differs from the frozen source")
    for group in range(ROW_GROUPS):
        if metadata.row_group(group).num_rows != ROWS_PER_GROUP:
            raise ValueError(f"validation parquet row-group length changed: {group}")
    q8_identity = (q8_receipt.get("panel_sha256"), q8_receipt.get("panel_bytes"))
    if q8_identity != (VALID_PARQUET_SHA256, VALID_PARQUET_BYTES):
        raise ValueError("validation parquet's inherited full-file identity is absent")
    return parquet
