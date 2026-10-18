"""Run only the frozen F0 split loader and emit a compatibility receipt."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

OVERLAY_ROOT = Path(__file__).resolve().parents[1]
UTC = getattr(datetime, "UTC", datetime.timezone(datetime.timedelta(0)))


def _ordered_json_sha256(values: list[int]) -> str:
    """Hash ordered integer membership using the receipt's canonical JSON form."""
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _build_membership_bindings(
    split_manifest_path: Path,
    membership_receipt_path: Path,
    expected_split_sha256: str,
    load_membership_receipt: Callable[[Path], tuple[dict[str, object], str]],
) -> tuple[dict[str, int], dict[str, str]]:
    receipt, _ = load_membership_receipt(membership_receipt_path)
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    sections = {
        "fit.train": ("fit", "train"),
        "fit.valid": ("fit", "valid"),
        "evaluation.valid": ("evaluation", "valid"),
        "excluded.train": ("excluded", "train"),
        "excluded.valid": ("excluded", "valid"),
        "excluded.corrupt_train": ("excluded", "corrupt_train"),
    }
    counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    expected_lists = receipt["comparison"]["membership_lists"]
    for name, (section, key) in sections.items():
        values = manifest[section][key]
        counts[name] = len(values)
        digests[name] = _ordered_json_sha256(values)
        expected = expected_lists[name]
        if (
            counts[name] != expected["count_old"]
            or digests[name] != expected["old_ordered_json_sha256"]
        ):
            raise ValueError(f"staged split membership differs from comparison receipt: {name}")
    if manifest.get("version") != 1 or manifest.get("fold") != 0:
        raise ValueError("unexpected frozen F0 split format")
    if hashlib.sha256(split_manifest_path.read_bytes()).hexdigest() != expected_split_sha256:
        raise ValueError("staged split manifest is not the frozen Main F0 split")
    return counts, digests


def _require_output_absent(output: Path) -> None:
    """Reject any existing output path before starting the expensive loader."""
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing compatibility receipt: {output}")


def _write_receipt_create_only(output: Path, receipt: dict[str, object]) -> None:
    """Publish a small receipt only when its destination is still absent."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def main() -> int:
    """Run the frozen loader preflight and write its source-bound receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--membership-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require_output_absent(args.output)
    sys.path.insert(0, str(OVERLAY_ROOT / "src"))
    compatibility = importlib.import_module("wnn.compatibility_f0")
    splits = importlib.import_module("wnn.splits")
    overlay_sha256_at_start = compatibility._verify_overlay_manifest()

    membership_receipt_path = args.membership_receipt.resolve()
    os.environ["WNN_F0_MEMBERSHIP_RECEIPT"] = str(membership_receipt_path)
    portable_registry = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    if not portable_registry:
        raise ValueError("WNN_PORTABLE_DATA_MANIFEST must name the pinned F0 registry")
    registry_path = Path(portable_registry).resolve()
    registry_sha256 = splits.file_sha256(registry_path)
    if registry_sha256 != compatibility._F0_PORTABLE_REGISTRY_SHA256:
        raise ValueError("portable input registry differs from the frozen F0 registry")

    split_path = args.split_manifest.resolve()
    loaded = splits.load_campaign_split(
        split_path,
        args.project_root.resolve(),
        compatibility_preflight=True,
    )
    membership_counts, membership_digests = _build_membership_bindings(
        split_path,
        membership_receipt_path,
        compatibility._F0_OLD_SPLIT_SHA256,
        compatibility._load_f0_membership_receipt,
    )
    if loaded.sha256 != compatibility._F0_OLD_SPLIT_SHA256:
        raise ValueError("loader returned a split different from frozen Main F0")
    artifact_bindings = compatibility.verify_f0_checkpoint_bindings(args.project_root.resolve())

    chain_folds = args.project_root.resolve() / "cache/chain_folds.npz"
    source_path, resolved_path = compatibility._verify_canonical_chain_folds_link(chain_folds)
    chain_folds_identity = {
        "path": "cache/chain_folds.npz",
        "source_path": source_path,
        "resolved_path": resolved_path,
        "bytes": chain_folds.stat().st_size,
        "sha256": splits.file_sha256(chain_folds),
    }
    if chain_folds_identity != {
        "path": "cache/chain_folds.npz",
        "source_path": str(compatibility._F0_CANONICAL_CHAIN_FOLDS_SOURCE),
        "resolved_path": str(compatibility._F0_CANONICAL_CHAIN_FOLDS_RESOLVED),
        "bytes": compatibility._F0_CURRENT_CHAIN_FOLDS_BYTES,
        "sha256": compatibility._F0_CURRENT_CHAIN_FOLDS_SHA256,
    }:
        raise ValueError("chain-fold source changed after loader validation")
    _, membership_sha256 = compatibility._load_f0_membership_receipt(membership_receipt_path)
    overlay_sha256 = compatibility._verify_overlay_manifest()
    if overlay_sha256 != overlay_sha256_at_start:
        raise ValueError("compatibility overlay changed during CPU preflight")
    receipt = {
        "schema": "quality_gap.f0_compatibility_preflight.v2",
        "status": "PASS_CPU_LOADER_PREFLIGHT",
        "created_utc": datetime.datetime.now(UTC).isoformat(),
        "old_split_manifest_sha256": loaded.sha256,
        "canonical_split_manifest_sha256": compatibility._F0_CANONICAL_SPLIT_SHA256,
        "membership_comparison_receipt_sha256": membership_sha256,
        "portable_inputs_registry_sha256": registry_sha256,
        "overlay_manifest_sha256": overlay_sha256,
        "chain_folds": chain_folds_identity,
        "artifact_bindings": artifact_bindings,
        "membership_counts": membership_counts,
        "membership_ordered_sha256": membership_digests,
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
    _write_receipt_create_only(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
