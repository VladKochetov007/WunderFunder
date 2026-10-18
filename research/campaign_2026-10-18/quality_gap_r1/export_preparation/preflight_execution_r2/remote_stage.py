"""Create or verify the isolated B project for the R2 CPU preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
PROJECT_ROOT = BASE / "quality-gap-r1/export-preparation-r2/root"
EXECUTION_REL = Path(
    "research/campaign_2026-10-18/quality_gap_r1/export_preparation/preflight_execution_r2"
)
OVERLAY_MANIFEST_SHA256 = "1785e8529a2868e1ca40a0c668c57ccc28ae82b04fda7c13bf7a3c8bc32c409b"
OLD_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
REGISTRY_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
MEMBERSHIP_SHA256 = "6a32f26b2ecd2ea082d2bc6f979ae62533e3360b9cf6169c777fa2f1ce886c5d"
R1_REVIEW_SHA256 = "419495adde1c3e1612efc217d6f69787aa818d727471b2a34e6c51f25d7c8e42"
R2_REVIEW_RESULT = "PASS_STATIC_REVIEW_READY_FOR_ROOT_GATED_PREFLIGHT"
CACHE_LOGICAL = Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/chain_folds.npz"
)
CACHE_RESOLVED = Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/receipts/chain_folds_derived.npz"
)
CACHE_SHA256 = "a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567"


def sha256_file(path: Path) -> str:
    """Hash small code, metadata, and restored artifact files only."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a staged metadata file."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Write a new JSON evidence file without replacing existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def file_inventory(root: Path, excluded: set[str]) -> set[str]:
    """Return regular-file paths relative to a staged root."""
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def reject_bytecode(root: Path) -> None:
    """Prevent stale bytecode from shadowing any reviewed runtime source."""
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed in the staged source tree: {path}")
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            raise ValueError(f"cached Python bytecode is not allowed: {path}")


def verify_source_snapshot(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    """Verify the complete copied H390 snapshot against its source manifest."""
    snapshot_root = root / "frozen_source_snapshot"
    manifest_path = snapshot_root / "SOURCE_MANIFEST_H390.json"
    manifest_sha = binding["pins"]["source_snapshot"]["source_manifest_sha256"]
    if sha256_file(manifest_path) != manifest_sha:
        raise ValueError("staged frozen source manifest hash changed")
    manifest = read_json(manifest_path)
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("staged source manifest has no source inventory")
    if file_inventory(snapshot_root, {"SOURCE_MANIFEST_H390.json"}) != set(sources):
        raise ValueError("staged frozen source inventory differs from its manifest")
    reject_bytecode(snapshot_root)
    for relative, expected_sha in sources.items():
        path = snapshot_root / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"staged frozen source changed: {relative}")
    predictor = snapshot_root / "scripts/predict_run.py"
    if sha256_file(predictor) != binding["pins"]["source_snapshot"]["predict_run_sha256"]:
        raise ValueError("staged frozen predictor hash changed")
    return {
        "manifest_sha256": manifest_sha,
        "predict_run_sha256": sha256_file(predictor),
        "files_verified": len(sources),
    }


def verify_overlay(root: Path) -> dict[str, Any]:
    """Verify exact source and support file inventories under the runtime overlay."""
    overlay_root = (
        root
        / "research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2"
    )
    manifest_path = overlay_root / "OVERLAY_MANIFEST_R2.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != OVERLAY_MANIFEST_SHA256:
        raise ValueError("staged R2 manifest differs from the independently reviewed digest")
    manifest = read_json(manifest_path)
    entries = manifest.get("source_files", []) + manifest.get("support_files", [])
    expected_paths = {entry["path"] for entry in entries}
    if len(expected_paths) != len(entries):
        raise ValueError("staged R2 inventory contains duplicate paths")
    if file_inventory(overlay_root, {"OVERLAY_MANIFEST_R2.json"}) != expected_paths:
        raise ValueError("staged R2 overlay inventory is incomplete or contains extra files")
    reject_bytecode(overlay_root)
    for entry in entries:
        path = overlay_root / entry["path"]
        expected_sha = entry.get("overlay_sha256", entry.get("sha256"))
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(
                f"staged R2 overlay file has an unexpected type or size: {entry['path']}"
            )
        if sha256_file(path) != expected_sha:
            raise ValueError(f"staged R2 overlay file hash changed: {entry['path']}")
    return {"manifest_sha256": manifest_sha, "files_verified": len(entries)}


def verify_metadata(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    """Verify the old split, registry, membership, and review evidence."""
    staged = binding["staged_paths"]
    checked: dict[str, dict[str, Any]] = {}
    for label, pin_key, relative_key in (
        ("old_split", "old_split", "old_split"),
        ("portable_registry", "portable_registry", "portable_registry"),
        ("membership_receipt", "membership_receipt", "membership_receipt"),
        ("prior_r1_review", "r1_negative_review", "prior_r1_review"),
        ("r2_review", "r2_review", "r2_review"),
    ):
        pin = binding["pins"][pin_key]
        path = root / staged[relative_key]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"staged metadata is missing or linked: {label}")
        digest = sha256_file(path)
        if digest != pin["sha256"]:
            raise ValueError(f"staged metadata hash changed: {label}")
        checked[label] = {
            "path": staged[relative_key],
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    r1_review = read_json(root / staged["prior_r1_review"])
    if r1_review.get("review_result") != "NOT_APPROVED_FOR_ADMISSION_R1":
        raise ValueError("preserved R1 negative review content changed")
    r2_review = read_json(root / staged["r2_review"])
    if (
        r2_review.get("review_result") != R2_REVIEW_RESULT
        or r2_review.get("reviewed_manifest_sha256") != OVERLAY_MANIFEST_SHA256
        or binding["review"]["reviewed_manifest_sha256"] != OVERLAY_MANIFEST_SHA256
    ):
        raise ValueError(
            "staged R2 review does not approve this exact manifest for gated preflight"
        )
    return checked


def create_link(project_root: Path, project_path: str, source_path: str) -> dict[str, Any]:
    """Create a new absolute symlink and verify both its literal and resolved targets."""
    source = Path(source_path)
    destination = project_root / project_path
    if not source.is_file():
        raise FileNotFoundError(f"pinned source is unavailable: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace staged path: {destination}")
    destination.symlink_to(source)
    if not destination.is_symlink() or os.readlink(destination) != str(source):
        raise ValueError(f"staged symlink literal target changed: {project_path}")
    resolved = destination.resolve(strict=True)
    expected_resolved = source.resolve(strict=True)
    if resolved != expected_resolved:
        raise ValueError(f"staged symlink resolved target changed: {project_path}")
    return {
        "project_path": project_path,
        "literal_target": os.readlink(destination),
        "resolved_target": str(resolved),
        "bytes": destination.stat().st_size,
    }


def verify_source_links(root: Path, binding: dict[str, Any], *, create: bool) -> dict[str, Any]:
    """Create or validate input links without reading large arrays or parquet bodies."""
    links: list[dict[str, Any]] = []
    for item in binding["remote"]["portable_sources"]:
        source = Path(item["source_path"])
        project_path = item["project_path"]
        registry_identity = item["registry_entry"]["original_identity"]
        if project_path == "cache/chain_folds.npz":
            if (
                registry_identity.get("bytes") != 550072
                or registry_identity.get("sha256")
                != "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f"
            ):
                raise ValueError("frozen registry no longer records the old chain-fold cache")
            expected_size = 462719
            expected_sha = binding["remote"]["cache_folds"]["sha256"]
        else:
            expected_size = int(registry_identity["bytes"])
            expected_sha = item["registry_entry"].get("sha256")
        if not source.is_file() or source.stat().st_size != expected_size:
            raise ValueError(f"portable source size changed: {project_path}")
        if project_path == "cache/chain_folds.npz" and (
            not source.is_symlink() or source.resolve(strict=True) != CACHE_RESOLVED
        ):
            raise ValueError(
                "canonical cache source no longer follows the observed B symlink chain"
            )
        destination = root / project_path
        if create:
            create_link(root, project_path, str(source))
        if not destination.is_symlink() or os.readlink(destination) != str(source):
            raise ValueError(f"project source link literal target changed: {project_path}")
        if destination.resolve(strict=True) != source.resolve(strict=True):
            raise ValueError(f"project source link resolved target changed: {project_path}")
        if destination.stat().st_size != expected_size:
            raise ValueError(f"project source link size changed: {project_path}")
        should_hash = expected_size <= 1_000_000 or project_path == "cache/chain_folds.npz"
        digest = sha256_file(destination) if should_hash else None
        if should_hash and expected_sha and digest != expected_sha:
            raise ValueError(f"small portable cache digest changed: {project_path}")
        if project_path == "cache/chain_folds.npz" and (
            digest != CACHE_SHA256 or os.readlink(destination) != str(CACHE_LOGICAL)
        ):
            raise ValueError("staged canonical cache identity or logical source changed")
        links.append(
            {
                "project_path": project_path,
                "literal_target": os.readlink(destination),
                "resolved_target": str(destination.resolve(strict=True)),
                "bytes": expected_size,
                "small_sha256_checked": digest,
                "full_hash_deferred_to_preflight": not should_hash,
            }
        )
    return {"links": links, "full_source_hash_deferred": True}


def verify_checkpoint_links(root: Path, binding: dict[str, Any], *, create: bool) -> dict[str, Any]:
    """Create or verify the four source-preserving model/provenance links."""
    result: dict[str, Any] = {}
    for name, item in binding["remote"]["checkpoint_bindings"].items():
        target = str(item["path"])
        source = str(item["restore_path"])
        if create:
            create_link(root, target, source)
        staged = root / target
        if not staged.is_symlink() or os.readlink(staged) != source:
            raise ValueError(f"restored artifact link changed: {name}")
        if staged.resolve(strict=True) != Path(source).resolve(strict=True):
            raise ValueError(f"restored artifact resolved path changed: {name}")
        if staged.stat().st_size != item["bytes"] or sha256_file(staged) != item["sha256"]:
            raise ValueError(f"restored artifact hash or size changed: {name}")
        result[name] = {
            "path": target,
            "restore_path": source,
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
    return result


def verify_project(root: Path, binding: dict[str, Any], *, create: bool) -> dict[str, Any]:
    """Stage idempotently only on the first call, then check every staged binding."""
    if root != PROJECT_ROOT or root.is_symlink() or not root.is_dir():
        raise ValueError("project root must be the exact newly staged directory")
    helper_binding = binding["remote"]["stage_helper"]
    helper_path = root / helper_binding["path"]
    if (
        Path(__file__).resolve(strict=True) != helper_path.resolve(strict=True)
        or helper_path.stat().st_size != helper_binding["bytes"]
        or sha256_file(helper_path) != helper_binding["sha256"]
    ):
        raise ValueError("staged helper differs from its local stage hash pin")
    binding_path = root / binding["staged_paths"]["stage_binding"]
    if binding_path.is_symlink() or read_json(binding_path) != binding:
        raise ValueError("staged binding differs from the local stage plan")
    snapshot = verify_source_snapshot(root, binding)
    overlay = verify_overlay(root)
    metadata = verify_metadata(root, binding)
    portable_registry = read_json(root / binding["staged_paths"]["portable_registry"])
    expected_sources = {
        item["project_path"]: item["registry_entry"]
        for item in binding["remote"]["portable_sources"]
    }
    if portable_registry.get("sources") != expected_sources:
        raise ValueError("staged portable registry disagrees with the source path map")
    for directory in ("datasets", "cache", "artifacts"):
        path = root / directory
        if create:
            if os.path.lexists(path):
                raise FileExistsError(f"refusing to replace staged directory: {path}")
            path.mkdir(mode=0o755)
        elif not path.is_dir() or path.is_symlink():
            raise ValueError(f"staged project directory missing or linked: {directory}")
    links = verify_source_links(root, binding, create=create)
    checkpoints = verify_checkpoint_links(root, binding, create=create)
    expected_project_files = {
        *(item["project_path"] for item in binding["remote"]["portable_sources"]),
        *(item["path"] for item in binding["remote"]["checkpoint_bindings"].values()),
    }
    actual_project_files = {
        path.relative_to(root).as_posix()
        for directory in ("datasets", "cache", "artifacts")
        for path in (root / directory).rglob("*")
        if path.is_symlink() or path.is_file()
    }
    if actual_project_files != expected_project_files:
        raise ValueError("staged data and checkpoint link inventory is not exact")
    cache_dir = root / "cache"
    if not os.access(cache_dir, os.W_OK):
        raise PermissionError("staged cache directory is not writable by the preflight account")
    output = root / binding["staged_paths"]["overlay"] / "COMPATIBILITY_RECEIPT_R2.json"
    if os.path.lexists(output):
        raise FileExistsError("preflight output already exists; refusing to overwrite")
    return {
        "schema": "quality_gap.preflight_stage_receipt.r2",
        "status": "STAGED" if create else "VERIFIED_READY_FOR_CPU_PREFLIGHT",
        "attempt_id": binding["attempt_id"],
        "overlay_manifest_sha256": overlay["manifest_sha256"],
        "r2_review_sha256": binding["review"]["sha256"],
        "source_snapshot": snapshot,
        "metadata": metadata,
        "portable_sources": links,
        "checkpoint_bindings": checkpoints,
        "stage_helper": helper_binding,
        "cache_directory_writable": True,
        "preflight_output_absent": True,
        "full_37gb_source_hash_performed": False,
        "model_or_target_array_loaded": False,
        "gpu_or_training_used": False,
    }


def main() -> int:
    """Apply the create-only stage or verify it without mutation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stage", "verify"), required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    try:
        if args.root != PROJECT_ROOT:
            raise ValueError("unexpected remote project root")
        binding = read_json(args.root / EXECUTION_REL / "STAGE_BINDING_R2.json")
        if binding.get("attempt_id") != args.attempt_id:
            raise ValueError("stage attempt id differs from the staged binding")
        receipt_path = args.root / EXECUTION_REL / "STAGE_RECEIPT_R2.json"
        if args.mode == "stage":
            if os.path.lexists(receipt_path):
                raise FileExistsError("stage receipt already exists; stage is create-only")
            receipt = verify_project(args.root, binding, create=True)
            write_json_exclusive(receipt_path, receipt)
        else:
            receipt = read_json(receipt_path)
            if receipt.get("status") != "STAGED" or receipt.get("attempt_id") != args.attempt_id:
                raise ValueError("existing stage receipt is not for this staged attempt")
            fresh = verify_project(args.root, binding, create=False)
            fresh["status"] = "VERIFIED_READY_FOR_CPU_PREFLIGHT"
            receipt = fresh
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "STAGE_ERROR", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
