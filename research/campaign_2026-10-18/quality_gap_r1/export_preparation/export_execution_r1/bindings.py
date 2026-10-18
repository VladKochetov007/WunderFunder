"""Fixed source, artifact, split, and execution pins for the F0 export."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXECUTION_DIR = Path(__file__).resolve().parent
RUNTIME_MODULES = frozenset({"bindings.py", "prepare.py", "launch.py", "worker.py", "capture.py"})
REPO_ROOT = EXECUTION_DIR.parents[4]
EXPORT_PREP = REPO_ROOT / "research/campaign_2026-10-18/quality_gap_r1/export_preparation"
OVERLAY_DIR = EXPORT_PREP / "compatibility_overlay_r2"
OVERLAY_MANIFEST = OVERLAY_DIR / "OVERLAY_MANIFEST_R2.json"
OVERLAY_REVIEW = OVERLAY_DIR / "INDEPENDENT_REVIEW_R2.json"
MEMBERSHIP = EXPORT_PREP / "F0_MEMBERSHIP_COMPARISON.json"
SNAPSHOT_DIR = REPO_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot"
)

OLD_SPLIT_REL = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
REGISTRY_REL = Path("research/campaign_2026-10-04/portable_inputs.json")
MEMBERSHIP_REL = Path(
    "research/campaign_2026-10-18/quality_gap_r1/export_preparation/F0_MEMBERSHIP_COMPARISON.json"
)
OVERLAY_REL = Path(
    "research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2"
)
EXECUTION_REL = Path(
    "research/campaign_2026-10-18/quality_gap_r1/export_preparation/export_execution_r1"
)
R2_RECEIPT_REL = OVERLAY_REL / "COMPATIBILITY_RECEIPT_R2.json"

B_BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
B_HOME = Path("/home/grinchenkoeg/WunderFunder")
B_CAMPAIGN_ROOT = B_BASE / "source-a266b036"
B_PROJECT_ROOT = B_BASE / "quality-gap-r1/export-preparation-r2/root"
B_PYTHON = B_HOME / ".venv/bin/python"
B_LIMITED = B_CAMPAIGN_ROOT / "scripts/limited"
B_GUARD = B_CAMPAIGN_ROOT / "scripts/bandit_resources.py"
B_GUARD_STATE = B_BASE / "runtime"
B_TMP = B_BASE / "research_tmp"
R2_REVIEW_REL = Path(
    "research/campaign_2026-10-18/quality_gap_r1/export_preparation/"
    "preflight_execution_r2/INDEPENDENT_REVIEW_R2.json"
)

RUN_NAME = "clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1"
OUTPUT_NAME = "clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1"
EPOCH = 5
GPU_UUID_RE = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)

PINS = {
    "overlay_manifest_sha256": "1785e8529a2868e1ca40a0c668c57ccc28ae82b04fda7c13bf7a3c8bc32c409b",
    "source_manifest_sha256": "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa",
    "predictor_sha256": "30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0",
    "old_split_sha256": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    "canonical_split_sha256": "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7",
    "registry_sha256": "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c",
    "membership_sha256": "6a32f26b2ecd2ea082d2bc6f979ae62533e3360b9cf6169c777fa2f1ce886c5d",
    "cache_sha256": "a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567",
    "cache_logical_source": "/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/chain_folds.npz",
    "cache_resolved_source": "/home/grinchenkoeg/computer-vision-homework-20261018/receipts/chain_folds_derived.npz",
    "bandit_resources_sha256": "3732ffeef79e635c8867d83e618f8d51b2fa8beffd7badf11f60412a864210f7",
    "limited_sha256": "d9bcfc3b671af3237b607e18087fde43fc7451b6482891e4abadd8580bfb8661",
}

RESOURCE_ENVELOPE = {
    "guard_kind": "gpu",
    "cpus": 1,
    "reserve_bytes": 8_000_000_000,
    "allocator_bytes": 4_000_000_000,
    "gpu_free_memory_safety_reserve_bytes": 1_000_000_000,
    "runtime_limit_seconds": 7200,
    "admission_timeout_seconds": 300,
    "profile": "strong-control-export-unmeasured-r1",
    "profile_status": "UNMEASURED_FOR_THIS_CONTROL",
}


def sha256_file(path: Path) -> str:
    """Hash a file with fixed-size reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object without recovering malformed data."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create a JSON evidence file without replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_evidence_path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Create immutable raw evidence without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_evidence_path(path)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def worker_preinference_record_path(attempt_id: str) -> Path:
    """Return the original strict-FP32 record written by the inference worker."""
    return EXECUTION_DIR / "records" / attempt_id / "WORKER_PRE_INFERENCE.json"


def captured_preinference_record_path(attempt_id: str) -> Path:
    """Return the distinct create-only archival copy produced by capture."""
    return EXECUTION_DIR / "records" / attempt_id / "CAPTURED_WORKER_PRE_INFERENCE.json"


def _require_evidence_path(path: Path) -> None:
    """Keep all newly written contract and run evidence inside this utility tree."""
    root = EXECUTION_DIR.resolve(strict=True)
    if not path.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("evidence path resolves outside the export utility directory")


def parse_utc(value: object) -> datetime:
    """Parse a timezone-aware ISO-8601 timestamp."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be ISO-8601 with a timezone") from error
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


def reject_bytecode(root: Path) -> None:
    """Reject bytecode that could shadow the reviewed source."""
    for path in root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            raise ValueError(f"cached bytecode is forbidden: {path}")


def verify_runtime_source_inventory() -> None:
    """Reject unreviewed top-level modules and all cached Python bytecode."""
    actual = {path.name for path in EXECUTION_DIR.glob("*.py") if path.is_file()}
    if actual != RUNTIME_MODULES:
        raise ValueError(
            f"runtime Python inventory differs from the reviewed allowlist: {sorted(actual)}"
        )
    reject_bytecode(EXECUTION_DIR)


def check_overlay(overlay_dir: Path, review_path: Path) -> dict[str, Any]:
    """Validate the complete review-bound R2 source and support inventory."""
    manifest_path = overlay_dir / "OVERLAY_MANIFEST_R2.json"
    if sha256_file(manifest_path) != PINS["overlay_manifest_sha256"]:
        raise ValueError("R2 manifest differs from the independently reviewed digest")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PREPARED_NOT_ADMITTED":
        raise ValueError("unexpected R2 manifest status")
    if review_path.is_symlink() or not review_path.is_file():
        raise ValueError("R2 review record must be staged beside the preflight tools")
    review = read_json(review_path)
    if (
        review.get("review_result") != "PASS_STATIC_REVIEW_READY_FOR_ROOT_GATED_PREFLIGHT"
        or review.get("reviewed_manifest_sha256") != PINS["overlay_manifest_sha256"]
        or review.get("admission_state")
        != "NOT_ADMITTED_PENDING_ROOT_MANIFEST_CHECK_AND_REAL_PREFLIGHT"
    ):
        raise ValueError("R2 static review does not bind the prepared overlay")
    entries = manifest.get("source_files", []) + manifest.get("support_files", [])
    expected: dict[str, tuple[int, str]] = {}
    for entry in entries:
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() in expected:
            raise ValueError("unsafe or duplicate R2 inventory path")
        expected[rel.as_posix()] = (
            int(entry["bytes"]),
            str(entry.get("overlay_sha256", entry.get("sha256"))),
        )
    actual = {
        path.relative_to(overlay_dir).as_posix()
        for path in overlay_dir.rglob("*")
        if path.is_file()
        and path.name
        not in {
            "OVERLAY_MANIFEST_R2.json",
            "COMPATIBILITY_RECEIPT_R2.json",
        }
    }
    if actual != set(expected):
        raise ValueError("R2 inventory has missing or unexpected files")
    reject_bytecode(overlay_dir)
    for relative, (size, digest) in expected.items():
        path = overlay_dir / relative
        if path.is_symlink() or path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"R2 file identity mismatch: {relative}")
    if any(path.is_symlink() for path in overlay_dir.rglob("*")):
        raise ValueError("R2 overlay must not contain symlinks")
    return {
        "manifest_sha256": PINS["overlay_manifest_sha256"],
        "review_sha256": sha256_file(review_path),
        "file_count": len(expected),
        "checkpoint_bindings": manifest["checkpoint_bindings"],
    }


def validate_preflight_receipt(
    receipt_path: Path, *, overlay_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Check actual CPU-preflight evidence against every expected source binding."""
    receipt = read_json(receipt_path)
    membership = read_json(MEMBERSHIP)
    comparison_lists = membership["comparison"]["membership_lists"]
    expected_counts = {name: row["count_old"] for name, row in comparison_lists.items()}
    expected_orders = {
        name: row["old_ordered_json_sha256"] for name, row in comparison_lists.items()
    }
    expected = {
        "schema": "quality_gap.f0_compatibility_preflight.v2",
        "status": "PASS_CPU_LOADER_PREFLIGHT",
        "old_split_manifest_sha256": PINS["old_split_sha256"],
        "canonical_split_manifest_sha256": PINS["canonical_split_sha256"],
        "membership_comparison_receipt_sha256": PINS["membership_sha256"],
        "portable_inputs_registry_sha256": PINS["registry_sha256"],
        "overlay_manifest_sha256": PINS["overlay_manifest_sha256"],
        "chain_folds": {
            "path": "cache/chain_folds.npz",
            "source_path": PINS["cache_logical_source"],
            "resolved_path": PINS["cache_resolved_source"],
            "bytes": 462719,
            "sha256": PINS["cache_sha256"],
        },
        "artifact_bindings": overlay_manifest["checkpoint_bindings"],
        "membership_counts": expected_counts,
        "membership_ordered_sha256": expected_orders,
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
    if set(receipt) != set(expected) | {"created_utc"}:
        raise ValueError("R2 receipt has an unexpected schema or field inventory")
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"R2 receipt binding mismatch: {key}")
    created = parse_utc(receipt["created_utc"])
    return {
        "path": str(receipt_path),
        "bytes": receipt_path.stat().st_size,
        "sha256": sha256_file(receipt_path),
        "created_utc": created.isoformat(),
        "status": receipt["status"],
    }


def validate_root_admission(
    admission_path: Path,
    *,
    attempt_id: str,
    preflight_sha256: str,
    require_fresh: bool = True,
) -> dict[str, Any]:
    """Require a fresh explicit root resource/ownership check for this attempt."""
    if admission_path.is_symlink() or not admission_path.is_file():
        raise ValueError("root admission must be a regular file")
    admission = read_json(admission_path)
    required = {
        "schema": "quality_gap.strong_control_export_admission.r1",
        "status": "ADMITTED_ROOT_ONE_GPU_EXPORT",
        "owner": "root",
        "attempt_id": attempt_id,
        "preflight_receipt_sha256": preflight_sha256,
        "one_gpu_only": True,
    }
    for key, value in required.items():
        if admission.get(key) != value:
            raise ValueError(f"root admission mismatch: {key}")
    gpu_uuid = admission.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or not GPU_UUID_RE.fullmatch(gpu_uuid):
        raise ValueError("root admission must name one full NVIDIA GPU UUID")
    created = parse_utc(admission.get("created_utc"))
    if require_fresh:
        age = (datetime.now(UTC) - created).total_seconds()
        if age < 0 or age > 300:
            raise ValueError("root resource/claim check is older than five minutes")
    resource = admission.get("resource_check")
    expected_resource = {
        "aggregate_cpu_cores": 2,
        "parent_memory_high_bytes": 18_000_000_000,
        "parent_memory_max_bytes": 20_000_000_000,
        "parent_swap_max_bytes": 0,
        "campaign_active_gpu_claims": [],
        "campaign_active_gpu_workers": [],
        "physical_scratch_path": str(B_TMP),
        "selected_gpu_uuid": gpu_uuid,
    }
    if not isinstance(resource, dict) or any(
        resource.get(key) != value for key, value in expected_resource.items()
    ):
        raise ValueError("root admission lacks the exact fresh aggregate/GPU claim check")
    if (
        type(resource.get("physical_scratch_free_bytes")) is not int
        or resource["physical_scratch_free_bytes"] < 2_000_000_000
    ):
        raise ValueError("root admission shows less than 2 GB free physical scratch")
    if (
        type(resource.get("selected_gpu_free_bytes")) is not int
        or resource["selected_gpu_free_bytes"] < 5_000_000_000
    ):
        raise ValueError("selected GPU has less than 5 GB fresh free memory")
    return {
        "path": str(admission_path),
        "bytes": admission_path.stat().st_size,
        "sha256": sha256_file(admission_path),
        "created_utc": created.isoformat(),
        "gpu_uuid": gpu_uuid,
        "resource_check": resource,
    }


def output_paths(project_root: Path) -> dict[str, Path]:
    """Return all paths the frozen predictor may create or remove."""
    final = project_root / "cache" / f"pred_{OUTPUT_NAME}_valid.npy"
    return {
        "prediction": final,
        "sidecar": final.with_suffix(".json"),
        "partial": final.with_name(final.name + ".partial"),
        "family_prediction": project_root / "cache" / f"allout_{OUTPUT_NAME}_valid.npy",
        "stale_seqsum": project_root / "cache" / f"seqsums_{OUTPUT_NAME}.npz",
    }


def require_project_cache_directory(project_root: Path) -> Path:
    """Require the predictor output directory to be a physical project directory."""
    cache = project_root / "cache"
    if cache.is_symlink() or not cache.is_dir():
        raise ValueError("project cache must be a real directory under the staged project root")
    if cache.resolve(strict=True).parent != project_root.resolve(strict=True):
        raise ValueError("project cache resolves outside the staged project root")
    return cache


def require_outputs_absent(project_root: Path) -> dict[str, str]:
    """Refuse launch if any final, sidecar, partial, or stale-score path exists."""
    require_project_cache_directory(project_root)
    paths = output_paths(project_root)
    present = [str(path) for path in paths.values() if os.path.lexists(path)]
    if present:
        raise FileExistsError("export output paths already exist: " + ", ".join(present))
    return {key: str(path) for key, path in paths.items()}


verify_runtime_source_inventory()
