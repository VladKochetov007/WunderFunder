"""Pinned F0 compatibility policy for the isolated frozen-source overlay."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from wnn.compatibility_f0_artifacts import verify_f0_checkpoint_bindings


def _file_sha256(path: Path) -> str:
    """Hash a small source or receipt without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_F0_OLD_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
_F0_CANONICAL_SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
_F0_MEMBERSHIP_RECEIPT_SHA256 = "6a32f26b2ecd2ea082d2bc6f979ae62533e3360b9cf6169c777fa2f1ce886c5d"
_F0_PORTABLE_REGISTRY_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
_F0_OLD_CHAIN_FOLDS_IDENTITY = {
    "bytes": 550072,
    "mtime_ns": 1790383861269564648,
    "sha256": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
}
_F0_OLD_CHAIN_FOLDS_RECEIPT_IDENTITY = {
    "bytes": 550072,
    "sha256": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
}
_F0_CURRENT_CHAIN_FOLDS_BYTES = 462719
_F0_CURRENT_CHAIN_FOLDS_SHA256 = "a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567"
_F0_MEMBERSHIP_STATUS = (
    "PASS_EXACT_ORDERED_MEMBERSHIP_AND_COMPONENT_MAP; SPLIT_MANIFEST_IDENTITIES_DIFFER"
)
_F0_MEMBERSHIP_COUNTS = {
    "fit.train": 10535,
    "fit.valid": 1405,
    "evaluation.valid": 468,
    "excluded.train": 71,
    "excluded.valid": 468,
    "excluded.corrupt_train": 1,
}


def _load_f0_membership_receipt(path: Path) -> tuple[dict[str, object], str]:
    """Load the pinned ordered-membership evidence used by the F0 cache overlay."""
    digest = _file_sha256(path)
    if digest != _F0_MEMBERSHIP_RECEIPT_SHA256:
        raise ValueError("F0 membership receipt does not match the reviewed file")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "quality_gap.f0_membership_comparison.v1"
        or receipt.get("status") != _F0_MEMBERSHIP_STATUS
    ):
        raise ValueError("F0 membership receipt has an unexpected status")
    inputs = receipt.get("inputs", {})
    if (
        inputs.get("main_frozen_split", {}).get("sha256") != _F0_OLD_SPLIT_SHA256
        or inputs.get("canonical_current_f0", {}).get("sha256") != _F0_CANONICAL_SPLIT_SHA256
    ):
        raise ValueError("F0 membership receipt names different split manifests")
    comparison = receipt.get("comparison", {})
    membership_lists = comparison.get("membership_lists", {})
    if set(membership_lists) != set(_F0_MEMBERSHIP_COUNTS):
        raise ValueError("F0 membership receipt has an unexpected membership set")
    for name, expected_count in _F0_MEMBERSHIP_COUNTS.items():
        entry = membership_lists[name]
        if (
            entry.get("ordered_equal") is not True
            or entry.get("count_old") != expected_count
            or entry.get("count_canonical") != expected_count
            or entry.get("old_ordered_json_sha256") != entry.get("canonical_ordered_json_sha256")
        ):
            raise ValueError(f"F0 ordered membership differs: {name}")
    if comparison.get("sequences_component_mapping", {}).get(
        "ordered_or_keyed_value_equal"
    ) is not True or not all(comparison.get("shared_fold_metadata", {}).values()):
        raise ValueError("F0 component mapping or shared fold metadata differs")
    identities = comparison.get("content_identity_interpretation", {})
    if (
        identities.get("only_changed_source_content_identity") != "cache/chain_folds.npz"
        or identities.get("chain_folds_old") != _F0_OLD_CHAIN_FOLDS_RECEIPT_IDENTITY
        or identities.get("chain_folds_canonical")
        != {
            "bytes": _F0_CURRENT_CHAIN_FOLDS_BYTES,
            "sha256": _F0_CURRENT_CHAIN_FOLDS_SHA256,
        }
    ):
        raise ValueError("F0 membership receipt names unexpected cache identities")
    return receipt, digest


def _membership_bindings(receipt: dict[str, object]) -> tuple[dict[str, int], dict[str, str]]:
    """Return the exact counts and ordered digests recorded by the pinned comparison."""
    lists = receipt["comparison"]["membership_lists"]
    counts = {name: int(entry["count_old"]) for name, entry in lists.items()}
    digests = {name: str(entry["old_ordered_json_sha256"]) for name, entry in lists.items()}
    return counts, digests


def _validate_f0_compatibility_receipt(
    receipt: dict[str, object],
    *,
    split_sha256: str,
    membership_sha256: str,
    registry_sha256: str,
    overlay_sha256: str,
    artifact_bindings: dict[str, dict[str, object]],
    membership_counts: dict[str, int],
    membership_digests: dict[str, str],
) -> None:
    """Reject stale or incomplete CPU preflight approval for the one F0 cache exception."""
    expected_cache = {
        "path": "cache/chain_folds.npz",
        "bytes": _F0_CURRENT_CHAIN_FOLDS_BYTES,
        "sha256": _F0_CURRENT_CHAIN_FOLDS_SHA256,
    }
    expected_checks = {
        "loader_returned": True,
        "component_map_matches_feature_graph": True,
        "evaluation_membership_matches_chain_folds": True,
        "fit_and_evaluation_components_are_disjoint": True,
    }
    if (
        receipt.get("schema") != "quality_gap.f0_compatibility_preflight.v1"
        or receipt.get("status") != "PASS_CPU_LOADER_PREFLIGHT"
        or receipt.get("old_split_manifest_sha256") != split_sha256
        or receipt.get("old_split_manifest_sha256") != _F0_OLD_SPLIT_SHA256
        or receipt.get("canonical_split_manifest_sha256") != _F0_CANONICAL_SPLIT_SHA256
        or receipt.get("membership_comparison_receipt_sha256") != membership_sha256
        or receipt.get("membership_comparison_receipt_sha256") != _F0_MEMBERSHIP_RECEIPT_SHA256
        or receipt.get("portable_inputs_registry_sha256") != registry_sha256
        or receipt.get("portable_inputs_registry_sha256") != _F0_PORTABLE_REGISTRY_SHA256
        or receipt.get("overlay_manifest_sha256") != overlay_sha256
        or receipt.get("chain_folds") != expected_cache
        or receipt.get("artifact_bindings") != artifact_bindings
        or receipt.get("membership_counts") != membership_counts
        or receipt.get("membership_ordered_sha256") != membership_digests
        or receipt.get("checks") != expected_checks
    ):
        raise ValueError("F0 compatibility preflight receipt is not an exact PASS")


def _verify_overlay_manifest() -> str:
    """Verify every copied package file against the isolated overlay manifest."""
    overlay_root = Path(__file__).resolve().parents[2]
    manifest_path = overlay_root / "OVERLAY_MANIFEST_R1.json"
    manifest_sha256 = _file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "quality_gap.compatibility_overlay_manifest.v1"
        or manifest.get("status") != "PREPARED_NOT_ADMITTED"
        or manifest.get("source_snapshot", {}).get("source_manifest_sha256")
        != "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa"
    ):
        raise ValueError("compatibility overlay manifest is unreviewed or stale")
    files = manifest.get("source_files", [])
    if len(files) != 59:
        raise ValueError("compatibility overlay source inventory is incomplete")
    seen: set[str] = set()
    overlay_only_seen: set[str] = set()
    for entry in files:
        relative = Path(entry["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not str(relative).startswith("src/wnn/")
        ):
            raise ValueError("compatibility overlay manifest has an unsafe path")
        if str(relative) in seen:
            raise ValueError("compatibility overlay manifest repeats a source path")
        seen.add(str(relative))
        path = overlay_root / relative
        stat = path.stat()
        if stat.st_size != entry["bytes"] or _file_sha256(path) != entry["overlay_sha256"]:
            raise ValueError(f"compatibility overlay source changed: {relative}")
        if relative.as_posix() == "src/wnn/splits.py":
            if entry["baseline_sha256"] != (
                "19a31b005710a16b1d3eef7c7a7fa0001b801d8f1faf26c4b4155217b8bd18ea"
            ):
                raise ValueError("compatibility overlay baseline split source changed")
        elif relative.as_posix() in {
            "src/wnn/compatibility_f0.py",
            "src/wnn/compatibility_f0_artifacts.py",
        }:
            if entry.get("baseline_sha256") is not None or entry.get("overlay_only") is not True:
                raise ValueError("compatibility policy module must be marked overlay-only")
            overlay_only_seen.add(relative.as_posix())
        elif entry["baseline_sha256"] != entry["overlay_sha256"]:
            raise ValueError(f"unapproved compatibility overlay change: {relative}")
    if "src/wnn/splits.py" not in seen:
        raise ValueError("compatibility overlay omits the split loader")
    if overlay_only_seen != {
        "src/wnn/compatibility_f0.py",
        "src/wnn/compatibility_f0_artifacts.py",
    }:
        raise ValueError("compatibility overlay omits an isolated policy module")
    return manifest_sha256


def verify_f0_chain_folds_override(
    source: Path,
    identity: dict[str, object],
    split_sha256: str,
    *,
    compatibility_preflight: bool,
) -> None:
    """Allow only the reviewed canonical cache under the unchanged Main F0 split."""
    if split_sha256 != _F0_OLD_SPLIT_SHA256 or identity != _F0_OLD_CHAIN_FOLDS_IDENTITY:
        raise ValueError("F0 compatibility overlay requires the exact frozen split identity")
    registry_value = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    membership_value = os.environ.get("WNN_F0_MEMBERSHIP_RECEIPT")
    if not registry_value or not membership_value:
        raise ValueError("F0 compatibility overlay requires its pinned registry and receipt")
    registry_path, membership_path = Path(registry_value), Path(membership_value)
    registry_sha256 = _file_sha256(registry_path)
    if registry_sha256 != _F0_PORTABLE_REGISTRY_SHA256:
        raise ValueError("portable input registry differs from the frozen Main F0 registry")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    portable_entry = registry.get("sources", {}).get("cache/chain_folds.npz", {})
    if (
        registry.get("version") != 1
        or portable_entry.get("original_identity") != _F0_OLD_CHAIN_FOLDS_IDENTITY
        or portable_entry.get("sha256") != _F0_OLD_CHAIN_FOLDS_IDENTITY["sha256"]
    ):
        raise ValueError("portable registry does not bind the frozen Main F0 cache")
    membership_receipt, membership_sha256 = _load_f0_membership_receipt(membership_path)
    membership_counts, membership_digests = _membership_bindings(membership_receipt)
    overlay_sha256 = _verify_overlay_manifest()
    stat = source.stat()
    if stat.st_size != _F0_CURRENT_CHAIN_FOLDS_BYTES:
        raise ValueError("canonical F0 chain-fold cache has the wrong byte count")
    if _file_sha256(source) != _F0_CURRENT_CHAIN_FOLDS_SHA256:
        raise ValueError("canonical F0 chain-fold cache has the wrong SHA-256")
    artifact_bindings = verify_f0_checkpoint_bindings(source.parent.parent)
    if compatibility_preflight:
        return
    receipt_value = os.environ.get("WNN_F0_COMPATIBILITY_RECEIPT")
    if not receipt_value:
        raise ValueError("F0 compatibility export requires a completed CPU preflight receipt")
    receipt = json.loads(Path(receipt_value).read_text(encoding="utf-8"))
    _validate_f0_compatibility_receipt(
        receipt,
        split_sha256=split_sha256,
        membership_sha256=membership_sha256,
        registry_sha256=registry_sha256,
        overlay_sha256=overlay_sha256,
        artifact_bindings=artifact_bindings,
        membership_counts=membership_counts,
        membership_digests=membership_digests,
    )
