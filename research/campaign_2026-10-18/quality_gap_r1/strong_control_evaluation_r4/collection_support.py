"""Focused helpers for export validation and create-only receipt construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
from export_contract import validate_contract, validate_export_output_paths
from export_receipt import validate_receipt
from h390_sidecar import validate_h390_sidecar
from pins import (
    BUNDLE_SHA256,
    DIAGNOSTIC_INPUTS_SHA256,
    ROW_GROUPS,
    ROWS_PER_GROUP,
    SELECTED_ROWS,
)
from source_alignment import (
    require_sha256,
    resolve_manifest_path,
    sha256_array,
    sha256_file,
    stat_identity,
)


def resolve_create_only_output(path: Path) -> Path:
    """Reject output paths whose final component or existing ancestors use symlinks."""
    absolute = path.absolute()
    if absolute.is_symlink() or absolute.resolve(strict=False) != absolute:
        raise ValueError("output path and existing ancestors must not contain symlinks")
    return absolute


def read_expected_json(path: Path, expected_sha256: str, name: str) -> dict[str, object]:
    """Read a JSON object after enforcing the hash supplied by the arriving export record."""
    if path.is_symlink():
        raise ValueError(f"{name} must be a regular file")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"{name} exceeds the configured JSON byte bound")
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError(f"{name} exceeds the configured JSON byte bound")
    require_sha256(hashlib.sha256(raw).hexdigest(), expected_sha256, name)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(dict[str, object], value)


def validate_export(
    prediction_path: Path,
    sidecar_path: Path,
    export_contract_path: Path,
    expected_export_contract_sha256: str,
    export_receipt_path: Path,
    expected_sidecar_sha256: str,
    expected_export_receipt_sha256: str,
) -> tuple[np.ndarray, dict[str, object], dict[str, object], dict[str, object]]:
    """Verify the frozen export contract, guarded capture, H390 sidecar and bytes."""
    if any(
        path.is_symlink()
        for path in (prediction_path, sidecar_path, export_contract_path, export_receipt_path)
    ):
        raise ValueError("prediction, sidecar, contract, and receipt must be regular files")
    contract, contract_sha256 = validate_contract(
        export_contract_path, expected_export_contract_sha256
    )
    validate_export_output_paths(contract, prediction_path, sidecar_path)
    prediction_resolved = prediction_path.resolve(strict=True)
    sidecar_resolved = sidecar_path.resolve(strict=True)
    export_receipt_resolved = export_receipt_path.resolve(strict=True)
    if prediction_resolved.is_symlink() or sidecar_resolved.is_symlink():
        raise ValueError("prediction and sidecar must be regular files")
    sidecar = read_expected_json(sidecar_path, expected_sidecar_sha256, "prediction sidecar")
    prediction_identity = stat_identity(prediction_resolved)
    prediction_sha256 = sha256_file(prediction_resolved, 1_000_000_000)
    if stat_identity(prediction_resolved) != prediction_identity:
        raise ValueError("full prediction changed while its sealed SHA-256 was computed")
    require_sha256(str(sidecar.get("prediction_sha256", "")), prediction_sha256, "prediction")
    export_receipt, capture_bindings = validate_receipt(
        export_receipt_resolved,
        expected_export_receipt_sha256,
        export_contract_path.resolve(strict=True),
        contract,
        contract_sha256,
        prediction_resolved,
        sidecar_resolved,
        expected_sidecar_sha256,
        prediction_sha256,
    )
    row_map = validate_h390_sidecar(sidecar, export_receipt)
    prediction = np.load(prediction_resolved, mmap_mode="r", allow_pickle=False)
    if prediction.shape != (ROW_GROUPS, ROWS_PER_GROUP, 2) or prediction.dtype != np.float32:
        raise ValueError("full-valid control prediction must be float32 [1873, 20000, 2]")
    for start in range(0, ROW_GROUPS, 32):
        if not np.isfinite(prediction[start : start + 32]).all():
            raise ValueError("full-valid control prediction contains nonfinite values")
    file_bindings = {
        **capture_bindings,
        "export_contract_sha256": contract_sha256,
        "export_receipt_sha256": expected_export_receipt_sha256,
        "sidecar_sha256": expected_sidecar_sha256,
        "prediction_sha256": prediction_sha256,
        "prediction_stat_identity": list(prediction_identity),
        "attempt_id": capture_bindings["attempt_id"],
        "raw_guard_attempt_sha256": capture_bindings["raw_guard_attempt_sha256"],
        "ordered_row_groups_sha256": capture_bindings["ordered_row_groups_sha256"],
        "export_contract_path": str(export_contract_path.resolve()),
        "raw_guard_attempt_path": str(capture_bindings["raw_guard_attempt_path"]),
        "captured_sidecar_path": str(capture_bindings["captured_sidecar_path"]),
    }
    return prediction, sidecar, export_receipt, {**file_bindings, "row_map": row_map}


def compare_bundle_arrays(
    extracted: dict[str, np.ndarray], expected: dict[str, np.ndarray], hashes: dict[str, str]
) -> None:
    """Require source labels, row groups and component identities to match byte-for-byte."""
    for name in ("target", "row_group", "component"):
        actual = extracted[name]
        reference = expected[name]
        if actual.dtype != reference.dtype or actual.shape != reference.shape:
            raise ValueError(f"source-derived {name} shape or dtype differs from compact bundle")
        if not np.array_equal(actual, reference) or actual.tobytes() != reference.tobytes():
            raise ValueError(f"source-derived {name} bytes differ from compact bundle")
        if sha256_array(actual) != hashes[name]:
            raise ValueError(f"source-derived {name} SHA-256 differs from frozen bundle")


def absolute_paths(manifest: dict[str, object], base: Path) -> dict[str, object]:
    """Copy a diagnostic manifest while preserving all existing resolved inputs."""
    derived = json.loads(json.dumps(manifest))
    source = cast(dict[str, object], derived["source"])
    fold = cast(dict[str, object], source["fold_manifest"])
    fold["path"] = str(resolve_manifest_path(base, fold["path"]))
    bundle = cast(dict[str, object], derived["bundle"])
    bundle["path"] = str(resolve_manifest_path(base, bundle["path"]))
    receipts = cast(dict[str, dict[str, object]], derived["receipts"])
    for item in receipts.values():
        item["path"] = str(resolve_manifest_path(base, item["path"]))
    models = cast(dict[str, dict[str, object]], derived["models"])
    for item in models.values():
        if item.get("kind") != "npy":
            continue
        item["path"] = str(resolve_manifest_path(base, item["path"]))
        alignment = cast(dict[str, object], item["alignment_receipt"])
        alignment["path"] = str(resolve_manifest_path(base, alignment["path"]))
    return cast(dict[str, object], derived)


def derive_strong_control_inputs(
    manifest: dict[str, object],
    base: Path,
    prediction_path: Path,
    prediction_sha256: str,
    alignment_path: Path,
    alignment_sha256: str,
) -> dict[str, object]:
    """Add the exact aligned prediction and three fixed comparisons to a manifest copy."""
    derived = absolute_paths(manifest, base)
    models = cast(dict[str, dict[str, object]], derived["models"])
    models["strong_control"] = {
        "kind": "npy",
        "path": str(prediction_path.resolve()),
        "sha256": prediction_sha256,
        "paired_bundle_rows": SELECTED_ROWS,
        "paired_bundle_sha256": BUNDLE_SHA256,
        "alignment_receipt": {
            "path": str(alignment_path.resolve()),
            "sha256": alignment_sha256,
            "fields": {
                "bundle": "paired_bundle_sha256",
                "prediction": "prediction_sha256",
                "rows": "selected_rows",
            },
        },
    }
    comparisons = cast(list[dict[str, str]], derived["comparisons"])
    for baseline in ("wide", "fp32_mix", "q8"):
        comparisons.append({"baseline": baseline, "candidate": "strong_control"})
    derived["derived_comparison"] = {
        "status": "prepared_from_sealed_same_row_alignment",
        "source_inputs_sha256": DIAGNOSTIC_INPUTS_SHA256,
        "alignment_receipt_sha256": alignment_sha256,
        "control_is_historical_strong_pipeline": True,
        "control_is_exact_public_incumbent": False,
        "search_or_tuning": False,
    }
    return derived


def write_create_only_json(path: Path, value: dict[str, object]) -> None:
    """Serialize one JSON receipt without replacing an existing file."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
