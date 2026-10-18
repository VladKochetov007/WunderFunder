"""Pinned local inputs and receipt validation for the R2 B-side preflight."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

EXECUTION_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXECUTION_DIR.parents[4]
EXPORT_PREP = REPO_ROOT / "research/campaign_2026-10-18/quality_gap_r1/export_preparation"
OVERLAY_DIR = EXPORT_PREP / "compatibility_overlay_r2"
OVERLAY_MANIFEST = OVERLAY_DIR / "OVERLAY_MANIFEST_R2.json"
R2_REVIEW = OVERLAY_DIR / "INDEPENDENT_REVIEW_R2.json"
R1_REVIEW = EXPORT_PREP / "compatibility_overlay_r1/INDEPENDENT_REVIEW_R1.json"
SNAPSHOT_DIR = REPO_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot"
)
SPLIT = REPO_ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
REGISTRY = REPO_ROOT / "research/campaign_2026-10-04/portable_inputs.json"
MEMBERSHIP = EXPORT_PREP / "F0_MEMBERSHIP_COMPARISON.json"

EXPECTED_OVERLAY_MANIFEST_SHA256 = (
    "1785e8529a2868e1ca40a0c668c57ccc28ae82b04fda7c13bf7a3c8bc32c409b"
)
EXPECTED_R2_REVIEW_RESULT = "PASS_STATIC_REVIEW_READY_FOR_ROOT_GATED_PREFLIGHT"
EXPECTED_R1_REVIEW_SHA256 = "419495adde1c3e1612efc217d6f69787aa818d727471b2a34e6c51f25d7c8e42"
EXPECTED_SOURCE_MANIFEST_SHA256 = "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa"
EXPECTED_PREDICTOR_SHA256 = "30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0"
EXPECTED_OLD_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_CANONICAL_SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
EXPECTED_REGISTRY_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
EXPECTED_MEMBERSHIP_SHA256 = "6a32f26b2ecd2ea082d2bc6f979ae62533e3360b9cf6169c777fa2f1ce886c5d"
EXPECTED_CACHE_SHA256 = "a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567"
EXPECTED_CACHE_LOGICAL = (
    "/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/chain_folds.npz"
)
EXPECTED_CACHE_RESOLVED = (
    "/home/grinchenkoeg/computer-vision-homework-20261018/receipts/chain_folds_derived.npz"
)

B_BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
B_DATA = B_BASE / "data"
B_PYTHON = Path("/home/grinchenkoeg/WunderFunder/.venv/bin/python")
B_TMP = B_BASE / "research_tmp"
B_PROJECT_ROOT = B_BASE / "quality-gap-r1/export-preparation-r2/root"
B_EXPORT_PREP = B_BASE / "quality-gap-r1/export-preparation-r2"
OLD_SPLIT_REL = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
REGISTRY_REL = Path("research/campaign_2026-10-04/portable_inputs.json")
MEMBERSHIP_REL = Path(
    "research/campaign_2026-10-18/quality_gap_r1/export_preparation/F0_MEMBERSHIP_COMPARISON.json"
)
OVERLAY_REL = Path(
    "research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2"
)
R1_REVIEW_REL = Path(
    "research/campaign_2026-10-18/quality_gap_r1/export_preparation/"
    "compatibility_overlay_r1/INDEPENDENT_REVIEW_R1.json"
)
EXECUTION_REL = Path(
    "research/campaign_2026-10-18/quality_gap_r1/export_preparation/preflight_execution_r2"
)


def sha256_file(path: Path) -> str:
    """Hash a small metadata or source file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _check_file(path: Path, byte_count: int, digest: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required local input is missing or linked: {path}")
    if path.stat().st_size != byte_count or sha256_file(path) != digest:
        raise ValueError(f"required local input identity changed: {path}")


def _check_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    source_binding = manifest["source_snapshot"]
    if source_binding["path"] != str(SNAPSHOT_DIR.relative_to(REPO_ROOT)):
        raise ValueError("R2 manifest points at an unexpected frozen snapshot")
    source_manifest = SNAPSHOT_DIR / "SOURCE_MANIFEST_H390.json"
    if sha256_file(source_manifest) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("frozen source manifest changed")
    snapshot = _read_json(source_manifest)
    if snapshot.get("schema") != "wnn.h390-source-snapshot.v3":
        raise ValueError("unexpected frozen source manifest schema")
    expected = snapshot.get("sources")
    if not isinstance(expected, dict):
        raise ValueError("frozen source file inventory is invalid")
    actual_paths = {
        path.relative_to(SNAPSHOT_DIR).as_posix()
        for path in SNAPSHOT_DIR.rglob("*")
        if path.is_file() and path.name != "SOURCE_MANIFEST_H390.json"
    }
    if actual_paths != set(expected):
        raise ValueError("frozen source snapshot file inventory changed")
    _reject_bytecode(SNAPSHOT_DIR)
    for relative, digest in expected.items():
        source = SNAPSHOT_DIR / relative
        if source.is_symlink() or sha256_file(source) != digest:
            raise ValueError(f"frozen source snapshot file changed: {relative}")
    predictor = SNAPSHOT_DIR / "scripts/predict_run.py"
    if sha256_file(predictor) != EXPECTED_PREDICTOR_SHA256:
        raise ValueError("frozen predictor hash changed")
    if any(path.is_symlink() for path in SNAPSHOT_DIR.rglob("*")):
        raise ValueError("frozen source snapshot contains a symlink")
    return {
        "path": str(SNAPSHOT_DIR.relative_to(REPO_ROOT)),
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "predict_run_sha256": EXPECTED_PREDICTOR_SHA256,
        "file_count": len(expected),
        "total_bytes": sum((SNAPSHOT_DIR / name).stat().st_size for name in expected),
    }


def _check_overlay(manifest: dict[str, Any]) -> None:
    source_entries = manifest.get("source_files")
    support_entries = manifest.get("support_files")
    if not isinstance(source_entries, list) or not isinstance(support_entries, list):
        raise ValueError("R2 source or support inventory is invalid")
    expected_entries = source_entries + support_entries
    expected_paths = {entry["path"] for entry in expected_entries}
    if len(expected_paths) != len(expected_entries):
        raise ValueError("R2 overlay inventory contains duplicate paths")
    actual_paths = {
        path.relative_to(OVERLAY_DIR).as_posix()
        for path in OVERLAY_DIR.rglob("*")
        if path.is_file()
        and path.name not in {"OVERLAY_MANIFEST_R2.json", "INDEPENDENT_REVIEW_R2.json"}
    }
    if actual_paths != expected_paths:
        raise ValueError("R2 overlay file inventory differs from its manifest")
    _reject_bytecode(OVERLAY_DIR)
    for entry in expected_entries:
        path = OVERLAY_DIR / entry["path"]
        _check_file(
            path,
            int(entry["bytes"]),
            str(entry["overlay_sha256"] if "overlay_sha256" in entry else entry["sha256"]),
        )
    if any(path.is_symlink() for path in OVERLAY_DIR.rglob("*")):
        raise ValueError("R2 overlay contains a symlink")
    receipt = OVERLAY_DIR / "COMPATIBILITY_RECEIPT_R2.json"
    if os.path.lexists(receipt):
        raise FileExistsError(f"R2 output receipt already exists: {receipt}")


def _reject_bytecode(root: Path) -> None:
    """Reject cached Python bytecode that could shadow reviewed source files."""
    for path in root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            raise ValueError(f"cached Python bytecode is not allowed: {path}")


def verify_local_prerequisites() -> dict[str, Any]:
    """Verify all reviewable local inputs without reading any large datasets."""
    if not EXECUTION_DIR.resolve().is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("preflight utilities resolved outside the repository")
    manifest_sha = sha256_file(OVERLAY_MANIFEST)
    if manifest_sha != EXPECTED_OVERLAY_MANIFEST_SHA256:
        raise ValueError("R2 manifest digest does not match the independently reviewed digest")
    manifest = _read_json(OVERLAY_MANIFEST)
    _check_overlay(manifest)
    review_sha = sha256_file(R2_REVIEW)
    review = _read_json(R2_REVIEW)
    if (
        review.get("review_result") != EXPECTED_R2_REVIEW_RESULT
        or review.get("reviewed_manifest_sha256") != manifest_sha
        or review.get("admission_state")
        != "NOT_ADMITTED_PENDING_ROOT_MANIFEST_CHECK_AND_REAL_PREFLIGHT"
    ):
        raise ValueError(
            "R2 independent review does not approve this exact manifest for root-gated preflight"
        )
    r1_sha = sha256_file(R1_REVIEW)
    if r1_sha != EXPECTED_R1_REVIEW_SHA256:
        raise ValueError("preserved R1 negative review changed")
    snapshot_binding = _check_snapshot(manifest)
    split_sha = sha256_file(SPLIT)
    registry_sha = sha256_file(REGISTRY)
    membership_sha = sha256_file(MEMBERSHIP)
    if (split_sha, registry_sha, membership_sha) != (
        EXPECTED_OLD_SPLIT_SHA256,
        EXPECTED_REGISTRY_SHA256,
        EXPECTED_MEMBERSHIP_SHA256,
    ):
        raise ValueError("split, portable registry, or membership receipt changed")
    registry = _read_json(REGISTRY)
    if registry.get("split_manifest_sha256") != split_sha:
        raise ValueError("portable registry is not bound to the frozen split")
    membership = _read_json(MEMBERSHIP)
    if membership.get("inputs", {}).get("main_frozen_split", {}).get("sha256") != split_sha:
        raise ValueError("membership receipt does not name the frozen split")
    if (
        membership.get("inputs", {}).get("canonical_current_f0", {}).get("sha256")
        != EXPECTED_CANONICAL_SPLIT_SHA256
    ):
        raise ValueError("membership receipt does not name canonical F0")
    if membership.get("status") != (
        "PASS_EXACT_ORDERED_MEMBERSHIP_AND_COMPONENT_MAP; SPLIT_MANIFEST_IDENTITIES_DIFFER"
    ):
        raise ValueError("membership receipt status is unexpected")
    return {
        "overlay_manifest": {
            "path": str(OVERLAY_MANIFEST.relative_to(REPO_ROOT)),
            "sha256": manifest_sha,
        },
        "r2_review": {"path": str(R2_REVIEW.relative_to(REPO_ROOT)), "sha256": review_sha},
        "r1_negative_review": {"path": str(R1_REVIEW.relative_to(REPO_ROOT)), "sha256": r1_sha},
        "source_snapshot": snapshot_binding,
        "old_split": {
            "path": str(SPLIT.relative_to(REPO_ROOT)),
            "sha256": split_sha,
            "bytes": SPLIT.stat().st_size,
        },
        "portable_registry": {
            "path": str(REGISTRY.relative_to(REPO_ROOT)),
            "sha256": registry_sha,
            "bytes": REGISTRY.stat().st_size,
        },
        "membership_receipt": {
            "path": str(MEMBERSHIP.relative_to(REPO_ROOT)),
            "sha256": membership_sha,
            "bytes": MEMBERSHIP.stat().st_size,
        },
        "overlay_file_count": len(manifest["source_files"]) + len(manifest["support_files"]),
    }


def build_stage_binding(attempt_id: str) -> dict[str, Any]:
    """Create the content-bound, create-only B staging plan."""
    local = verify_local_prerequisites()
    manifest = _read_json(OVERLAY_MANIFEST)
    registry = _read_json(REGISTRY)
    sources: list[dict[str, Any]] = []
    for logical_path, entry in sorted(registry["sources"].items()):
        sources.append(
            {
                "project_path": logical_path,
                "source_path": str(B_DATA / logical_path),
                "registry_entry": entry,
            }
        )
    checkpoint_bindings = manifest["checkpoint_bindings"]
    return {
        "schema": "quality_gap.preflight_stage_binding.r2",
        "attempt_id": attempt_id,
        "status": "PREPARED_CREATE_ONLY_STAGE",
        "review": {
            **local["r2_review"],
            "review_result": EXPECTED_R2_REVIEW_RESULT,
            "reviewed_manifest_sha256": local["overlay_manifest"]["sha256"],
        },
        "pins": local,
        "remote": {
            "base": str(B_BASE),
            "project_root": str(B_PROJECT_ROOT),
            "python": str(B_PYTHON),
            "physical_tmp": str(B_TMP),
            "stage_helper": {
                "path": (EXECUTION_REL / "remote_stage.py").as_posix(),
                "bytes": (EXECUTION_DIR / "remote_stage.py").stat().st_size,
                "sha256": sha256_file(EXECUTION_DIR / "remote_stage.py"),
            },
            "portable_sources": sources,
            "checkpoint_bindings": checkpoint_bindings,
            "cache_folds": {
                "logical_source": EXPECTED_CACHE_LOGICAL,
                "resolved_source": EXPECTED_CACHE_RESOLVED,
                "bytes": 462719,
                "sha256": EXPECTED_CACHE_SHA256,
            },
        },
        "resource_envelope": {
            "cpu_quota_percent": 50,
            "memory_high_bytes": 1300000000,
            "memory_max_bytes": 1500000000,
            "swap_max_bytes": 0,
            "runtime_max_seconds": 900,
            "source_hash_bytes_approx": 37100000000,
            "measured_full_hash_rate_bytes_per_second": None,
            "minimum_effective_rate_for_cap_bytes_per_second": 37100000000 / 900,
            "runtime_estimate": "No prior full-source scan timing is available; the 900-second limit is a hard cap, not a measured forecast.",
        },
        "staged_paths": {
            "frozen_source_snapshot": "frozen_source_snapshot",
            "overlay": OVERLAY_REL.as_posix(),
            "old_split": OLD_SPLIT_REL.as_posix(),
            "portable_registry": REGISTRY_REL.as_posix(),
            "membership_receipt": MEMBERSHIP_REL.as_posix(),
            "prior_r1_review": R1_REVIEW_REL.as_posix(),
            "r2_review": (EXECUTION_REL / "INDEPENDENT_REVIEW_R2.json").as_posix(),
            "stage_binding": (EXECUTION_REL / "STAGE_BINDING_R2.json").as_posix(),
            "stage_receipt": (EXECUTION_REL / "STAGE_RECEIPT_R2.json").as_posix(),
        },
        "no_model_or_target_array_load": True,
        "no_gpu_or_training": True,
    }


def expected_receipt_bindings() -> dict[str, Any]:
    """Return semantic input pins, deliberately excluding any output digest."""
    manifest = _read_json(OVERLAY_MANIFEST)
    membership = _read_json(MEMBERSHIP)
    counts = {
        name: item["count_old"]
        for name, item in membership["comparison"]["membership_lists"].items()
    }
    ordered = {
        name: item["old_ordered_json_sha256"]
        for name, item in membership["comparison"]["membership_lists"].items()
    }
    return {
        "schema": "quality_gap.f0_compatibility_preflight.v2",
        "status": "PASS_CPU_LOADER_PREFLIGHT",
        "old_split_manifest_sha256": EXPECTED_OLD_SPLIT_SHA256,
        "canonical_split_manifest_sha256": EXPECTED_CANONICAL_SPLIT_SHA256,
        "membership_comparison_receipt_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "portable_inputs_registry_sha256": EXPECTED_REGISTRY_SHA256,
        "overlay_manifest_sha256": EXPECTED_OVERLAY_MANIFEST_SHA256,
        "chain_folds": {
            "path": "cache/chain_folds.npz",
            "source_path": EXPECTED_CACHE_LOGICAL,
            "resolved_path": EXPECTED_CACHE_RESOLVED,
            "bytes": 462719,
            "sha256": EXPECTED_CACHE_SHA256,
        },
        "artifact_bindings": manifest["checkpoint_bindings"],
        "membership_counts": counts,
        "membership_ordered_sha256": ordered,
        "checks": {
            "loader_returned": True,
            "component_map_matches_feature_graph": True,
            "evaluation_membership_matches_chain_folds": True,
            "fit_and_evaluation_components_are_disjoint": True,
        },
        "scope": {
            "checkpoint_loaded": False,
            "prediction_or_grade_run": False,
            "training_or_gpu_execution": False,
            "model_or_target_arrays_loaded": False,
        },
    }


def validate_preflight_receipt(payload: dict[str, Any]) -> None:
    """Reject a receipt that does not bind the reviewed inputs and loader scope."""
    expected = expected_receipt_bindings()
    if set(payload) != set(expected) | {"created_utc"}:
        raise ValueError("preflight receipt has an unexpected field inventory")
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"preflight receipt binding mismatch: {field}")
    try:
        datetime.fromisoformat(str(payload["created_utc"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise ValueError("preflight receipt has an invalid timestamp") from exc
