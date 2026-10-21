"""Frozen-cache and package setup for the native precision probe."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from probe_config import (
    CP5_CACHE,
    CP5_CACHE_SHA256,
    CP5_SIDECAR_SHA256,
    CP5_ZIP,
    CP5_ZIP_SHA256,
    NATIVE_GRADE,
    NATIVE_GRADE_SHA256,
    ORIGINAL_CACHE,
    ORIGINAL_CACHE_SHA256,
    ORIGINAL_SIDECAR_SHA256,
    ORIGINAL_ZIP,
    ORIGINAL_ZIP_SHA256,
    ROOT,
)
from probe_native import (
    build_ensembles,
    build_single_precision,
    digest_file,
    load_full_cache,
    load_members,
    load_native_module,
    require_hash,
)


def checked_sources() -> tuple[Any, Any, Any, Any]:
    """Check package pins and open immutable full-F0 prediction caches."""
    split = json.loads(
        (ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text(
            encoding="utf-8"
        )
    )
    expected_groups = list(map(int, split["evaluation"]["valid"]))
    if len(expected_groups) != 468 or len(set(expected_groups)) != 468:
        raise ValueError("frozen F0 evaluation membership changed")
    require_hash(ORIGINAL_ZIP, ORIGINAL_ZIP_SHA256)
    require_hash(CP5_ZIP, CP5_ZIP_SHA256)
    require_hash(NATIVE_GRADE, NATIVE_GRADE_SHA256)
    original = load_full_cache(
        ORIGINAL_CACHE,
        ORIGINAL_CACHE.with_suffix(".json"),
        ORIGINAL_CACHE_SHA256,
        ORIGINAL_SIDECAR_SHA256,
        expected_groups,
    )
    cp5 = load_full_cache(
        CP5_CACHE,
        CP5_CACHE.with_suffix(".json"),
        CP5_CACHE_SHA256,
        CP5_SIDECAR_SHA256,
        expected_groups,
    )
    return original[0], original[1], cp5[0], cp5[1]


def extract_packages(
    scratch_dir: Path,
) -> tuple[Path, Path]:
    """Safely extract both immutable ZIPs into private physical scratch."""
    from native_runtime import extract_safe

    old_package, cp5_package = scratch_dir / "original", scratch_dir / "cp5"
    old_package.mkdir()
    cp5_package.mkdir()
    for archive_path, package in ((ORIGINAL_ZIP, old_package), (CP5_ZIP, cp5_package)):
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"ZIP CRC check failed: {archive_path}")
            extract_safe(archive, package)
    return old_package, cp5_package


def load_probe_native(old_package: Path, cp5_package: Path) -> tuple[Any, Any, str, str]:
    """Verify extension identity, then import the pinned candidate native module."""
    old_extensions = list(old_package.rglob("wnn_cpp*.so"))
    cp5_extensions = list(cp5_package.rglob("wnn_cpp*.so"))
    if (
        len(old_extensions) != 1
        or len(cp5_extensions) != 1
        or digest_file(old_extensions[0]) != digest_file(cp5_extensions[0])
    ):
        raise ValueError("native v3 extension bytes differ between frozen ZIPs")
    module, native, native_path, backend = load_native_module(cp5_package)
    return module, native, digest_file(native_path), backend


def make_models(native: Any, old_package: Path, cp5_package: Path) -> dict[str, Any]:
    """Build three H512 precisions, the old student and the fixed retained pair."""
    old_members, cp5_members = load_members(old_package), load_members(cp5_package)
    if [row["name"] for row in old_members[1:]] != [row["name"] for row in cp5_members[1:]]:
        raise ValueError("retained-member roster changed")
    for old, new in zip(old_members[1:], cp5_members[1:], strict=True):
        if bool(old.get("fp32", False)) != bool(new.get("fp32", False)):
            raise ValueError("retained member precision flags differ")
        if digest_file(old_package / old["weights"]) != digest_file(cp5_package / new["weights"]):
            raise ValueError("retained member weights differ between packages")
    student = dict(old_members[0], coef=1.0)
    h512 = cp5_members[0]
    return {
        "h512_dynamic_int8": build_single_precision(
            native, cp5_package, h512, fast=True, dynamic=True
        ),
        "h512_float32_fast": build_single_precision(
            native, cp5_package, h512, fast=True, dynamic=False
        ),
        "h512_float32_exact": build_single_precision(
            native, cp5_package, h512, fast=False, dynamic=False
        ),
        "original_student": build_ensembles(native, old_package, [student], unit_weight=True),
        "retained_pair": build_ensembles(native, old_package, old_members[1:]),
    }


def save_outputs(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    plan_sha: str,
    selection: dict[str, Any],
    native_sha: str,
    backend: str,
) -> Path:
    """Save five feature-only tracks and a pre-label hash manifest."""
    from probe_config import write_json

    output_path = output_dir / "member_predictions.npz"
    with output_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    hashes = {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in arrays.items()
    }
    write_json(
        output_dir / "outputs_manifest.json",
        {
            "schema": "wnn.native-h512-precision-probe-outputs.v1",
            "status": "FEATURE_ONLY_OUTPUTS_SAVED_BEFORE_TARGET_ACCESS",
            "selection_plan_sha256": plan_sha,
            "selected_membership_sha256": selection["selected_membership_sha256"],
            "output_npz": output_path.name,
            "output_npz_sha256": digest_file(output_path),
            "array_sha256": hashes,
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "native_extension_sha256": native_sha,
            "quant_backend": backend,
            "original_zip_sha256": ORIGINAL_ZIP_SHA256,
            "cp5_zip_sha256": CP5_ZIP_SHA256,
            "target_accessed": False,
        },
    )
    return output_path
