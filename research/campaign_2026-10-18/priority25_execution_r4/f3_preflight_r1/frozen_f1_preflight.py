#!/usr/bin/env python3
"""Verify fold-1 inputs and frozen split membership without fitting or inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import platform
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

LOG = logging.getLogger("f1_preflight")
REMOTE_ROOT = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
FOLD1_SHA256 = "d8e9748d770bb3af525943b36b1801e6fd72d78259cc6b823d4661ffd78367d9"
REGISTRY_SHA256 = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
ARCHIVE_MANIFEST_SHA256 = "9a90e46958eaed2784f1f12e6a501292e4c865b1bc6d07e3f2bd233271da7341"
SOURCE_REVISION = "a266b036"
SOURCE_ARCHIVE_SHA256 = "288b4c9dd67b48114240bbf593901fc539de15aa709a1aa63388dccf64a4478a"
SPLITS_MODULE_SHA256 = "24024fbb82d33afdc2c709912404f7c1c26504a0816a050428c9f66e04e951fe"
PORTABILITY_MODULE_SHA256 = "a35f84ee5c95fd45cd76aa5f34f7013c8e1c5963a6306b0e4049386515589a44"
SOURCE_NAMES = (
    "datasets/train.parquet",
    "datasets/valid.parquet",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
    "cache/chains.npz",
    "cache/chain_folds.npz",
)
MAX_SMALL_INPUT_BYTES = 2_000_000


def _stat_record(path: Path) -> dict[str, int]:
    """Return stable filesystem identity fields for a read-only input."""
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _read_small_json(
    path: Path, expected_sha256: str, label: str
) -> tuple[dict[str, object], dict[str, object]]:
    """Read and hash a bounded JSON input, requiring a stable file generation."""
    before = _stat_record(path)
    if int(before["bytes"]) <= 0 or int(before["bytes"]) > MAX_SMALL_INPUT_BYTES:
        raise ValueError(f"{label} exceeds the bounded JSON read size")
    raw = path.read_bytes()
    after = _stat_record(path)
    digest = hashlib.sha256(raw).hexdigest()
    if before != after or len(raw) != before["bytes"]:
        raise RuntimeError(f"{label} changed while being read")
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch: {digest}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value), {
        "path": str(path),
        "bytes": len(raw),
        "sha256": digest,
        "stat_before": before,
        "stat_after": after,
    }


def _same_file_generation(path: Path, before: Mapping[str, int]) -> dict[str, int]:
    """Require an input's identity to remain unchanged after verification."""
    after = _stat_record(path)
    if dict(before) != after:
        raise RuntimeError(f"input changed during preflight: {path}")
    return after


def _hash_observer() -> tuple[dict[str, object], Callable[[int, str | None], None]]:
    """Capture the size and digest emitted by the portable verifier."""
    observed: dict[str, object] = {}

    def on_hashed(size: int, digest: str | None) -> None:
        observed["bytes"] = size
        observed["sha256"] = digest

    return observed, on_hashed


def _hash_registered_inputs(
    data_root: Path,
    registry_path: Path,
    manifest: Mapping[str, object],
    registry: Mapping[str, object],
    verify_portable_source: Callable[..., str],
) -> dict[str, dict[str, object]]:
    """Stream-hash all six registered data inputs and preserve before/after stats."""
    manifest_sources = cast(dict[str, dict[str, object]], manifest["sources"])
    registry_sources = cast(dict[str, dict[str, object]], registry["sources"])
    if set(manifest_sources) != set(SOURCE_NAMES) or set(registry_sources) != set(SOURCE_NAMES):
        raise ValueError("fold and portable registry must bind exactly the six canonical inputs")

    results: dict[str, dict[str, object]] = {}
    for name in SOURCE_NAMES:
        path = (data_root / name).resolve(strict=True)
        entry = registry_sources[name]
        identity = cast(dict[str, object], entry["original_identity"])
        if manifest_sources[name] != identity:
            raise ValueError(f"fold/portable identity mismatch for {name}")
        if int(path.stat().st_size) != int(identity["bytes"]):
            raise ValueError(f"portable input byte count mismatch for {name}")
        before = _stat_record(path)
        if int(before["mtime_ns"]) != int(identity["mtime_ns"]):
            raise ValueError(f"native input mtime differs from fold identity for {name}")
        observed, on_hashed = _hash_observer()

        LOG.info("Hashing %s (%s bytes)", name, before["bytes"])
        digest = verify_portable_source(
            path,
            name,
            identity,
            registry_path,
            observation_callback=on_hashed,
        )
        after = _stat_record(path)
        if before != after:
            raise RuntimeError(f"portable input changed during hash: {name}")
        expected_digest = str(entry["sha256"])
        if digest != expected_digest or observed.get("sha256") != expected_digest:
            raise ValueError(f"portable input SHA256 mismatch for {name}")
        if int(observed.get("bytes", -1)) != before["bytes"]:
            raise ValueError(f"portable input size changed during hash: {name}")
        results[name] = {
            "path": str(path),
            "bytes": int(before["bytes"]),
            "sha256": digest,
            "stat_before": before,
            "stat_after_hash": after,
        }
        LOG.info("Verified %s sha256=%s", name, digest)
    return results


def _validate_archive_manifest(path: Path) -> dict[str, object]:
    """Pin the portable source archive inventory to the canonical trainer archive."""
    manifest, evidence = _read_small_json(path, ARCHIVE_MANIFEST_SHA256, "archive manifest")
    if manifest.get("source_revision") != SOURCE_REVISION:
        raise ValueError("portable archive manifest source revision mismatch")
    archives = cast(list[dict[str, object]], manifest["archives"])
    if not any(row.get("sha256") == SOURCE_ARCHIVE_SHA256 for row in archives):
        raise ValueError("canonical source archive SHA is absent from archive manifest")
    return evidence


def _load_and_validate_split(
    source_root: Path,
    data_root: Path,
    fold_path: Path,
) -> dict[str, object]:
    """Use the frozen loader for graph, native cache, and component-disjointness checks."""
    source_dir = (source_root / "src").resolve(strict=True)
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    splits_module = importlib.import_module("wnn.splits")
    split_module_path = Path(cast(str, splits_module.__file__)).resolve(strict=True)
    expected_splits_path = (source_dir / "wnn/splits.py").resolve(strict=True)
    if split_module_path != expected_splits_path:
        raise ValueError("loaded wnn.splits is not from the frozen source root")
    splits_sha256 = hashlib.sha256(split_module_path.read_bytes()).hexdigest()
    if splits_sha256 != SPLITS_MODULE_SHA256:
        raise ValueError("frozen wnn.splits source hash mismatch")

    portability_module = importlib.import_module("wnn.portability")
    portability_path = Path(cast(str, portability_module.__file__)).resolve(strict=True)
    if portability_path != (source_dir / "wnn/portability.py").resolve(strict=True):
        raise ValueError("loaded wnn.portability is not from the frozen source root")
    portability_sha256 = hashlib.sha256(portability_path.read_bytes()).hexdigest()
    if portability_sha256 != PORTABILITY_MODULE_SHA256:
        raise ValueError("frozen wnn.portability source hash mismatch")

    # Preserve the plan's native stat-identity route. Full portable content
    # hashes were checked separately against the matching registry above.
    os.environ.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    split = splits_module.load_campaign_split(fold_path, data_root)
    if (int(split.fold), int(split.n_folds), split.sha256) != (1, 4, FOLD1_SHA256):
        raise ValueError("canonical loader returned the wrong fold identity")
    expected_counts = (10_533, 1_416, 457)
    actual_counts = (len(split.train_groups), len(split.valid_groups), len(split.evaluation_groups))
    if actual_counts != expected_counts:
        raise ValueError(f"canonical loader membership counts differ: {actual_counts}")
    component_by_group = {
        (split_name, group): component for split_name, group, component in split.components
    }
    evaluation_components = {
        component_by_group[("valid", group)] for group in split.evaluation_groups
    }
    if len(evaluation_components) != 149:
        raise ValueError("canonical loader evaluation component count differs")
    return {
        "fold": int(split.fold),
        "n_folds": int(split.n_folds),
        "manifest_sha256": split.sha256,
        "fit_train_groups": len(split.train_groups),
        "fit_valid_groups": len(split.valid_groups),
        "evaluation_valid_groups": len(split.evaluation_groups),
        "evaluation_components": len(evaluation_components),
        "component_disjointness": "passed by frozen load_campaign_split.assert_fit_groups",
        "loader_modules": {
            "splits_path": str(split_module_path),
            "splits_sha256": splits_sha256,
            "portability_path": str(portability_path),
            "portability_sha256": portability_sha256,
        },
        "portable_registry_env_used": False,
        "native_source_identity_route": True,
        "portable_registry_original_identities_match_fold": True,
    }


def _schedule_evidence(counts: Mapping[str, object]) -> dict[str, object]:
    """Derive the unobserved F1 schedule from verified membership counts."""
    fit_sequences = int(counts["fit_train_groups"]) + int(counts["fit_valid_groups"])
    batches = fit_sequences // 8
    updates_per_batch = (20_000 + 2_000 - 1) // 2_000
    updates_per_epoch = batches * updates_per_batch
    return {
        "classification": "static expectation, not an observed optimizer result",
        "fit_sequences": fit_sequences,
        "batch_size": 8,
        "drop_last": True,
        "batches_per_epoch": batches,
        "updates_per_batch": updates_per_batch,
        "updates_per_epoch": updates_per_epoch,
        "stage1_epochs": 2,
        "stage1_updates": 2 * updates_per_epoch,
        "stage1_batches": 2 * batches,
        "stage1_next_epoch": 2,
        "terminal_epochs": 6,
        "terminal_updates": 6 * updates_per_epoch,
        "terminal_batches": 6 * batches,
        "terminal_next_epoch": 6,
        "selected_epoch": 5,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Hash fold-1 data and run the frozen split checks without GPU work."""
    inherited_portable_manifest = os.environ.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    campaign_root = args.campaign_root.resolve(strict=True)
    data_root = (campaign_root / "data").resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    fold_path = args.fold_manifest.resolve(strict=True)
    registry_path = args.portable_registry.resolve(strict=True)
    archive_path = args.archive_manifest.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"receipt parent directory is missing: {output.parent}")

    fold_manifest, fold_evidence = _read_small_json(fold_path, FOLD1_SHA256, "fold1 manifest")
    registry, registry_evidence = _read_small_json(
        registry_path, REGISTRY_SHA256, "portable source registry"
    )
    archive_evidence = _validate_archive_manifest(archive_path)
    registry_sources = cast(dict[str, dict[str, object]], registry["sources"])
    fold_sources = cast(dict[str, dict[str, object]], fold_manifest["sources"])
    if set(fold_sources) != set(SOURCE_NAMES) or set(registry_sources) != set(SOURCE_NAMES):
        raise ValueError("fold or portable registry source list is incomplete")
    for name in SOURCE_NAMES:
        if fold_sources[name] != registry_sources[name].get("original_identity"):
            raise ValueError(f"portable original identity does not match fold manifest: {name}")

    source_dir = (source_root / "src").resolve(strict=True)
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    splits_module = importlib.import_module("wnn.splits")
    splits_path = Path(cast(str, splits_module.__file__)).resolve(strict=True)
    if splits_path != (source_dir / "wnn/splits.py").resolve(strict=True):
        raise ValueError("split loader import is not from the requested frozen source")
    if hashlib.sha256(splits_path.read_bytes()).hexdigest() != SPLITS_MODULE_SHA256:
        raise ValueError("frozen wnn.splits source hash mismatch")

    portability_module = importlib.import_module("wnn.portability")
    expected_portability = (source_root / "src/wnn/portability.py").resolve(strict=True)
    portability_path = Path(cast(str, portability_module.__file__)).resolve(strict=True)
    if portability_path != expected_portability:
        raise ValueError("portable verifier import is not from the requested frozen source")
    if hashlib.sha256(portability_path.read_bytes()).hexdigest() != PORTABILITY_MODULE_SHA256:
        raise ValueError("frozen wnn.portability source hash mismatch")
    files = _hash_registered_inputs(
        data_root,
        registry_path,
        fold_manifest,
        registry,
        portability_module.verify_portable_source,
    )
    membership = _load_and_validate_split(
        source_root,
        data_root,
        fold_path,
    )
    for _name, item in files.items():
        item["stat_after_loader"] = _same_file_generation(
            Path(str(item["path"])), cast(Mapping[str, int], item["stat_after_hash"])
        )
    _read_small_json(fold_path, FOLD1_SHA256, "fold1 manifest after data checks")
    _read_small_json(registry_path, REGISTRY_SHA256, "portable registry after data checks")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    cuda_initialized = bool(torch.cuda.is_initialized())
    if cuda_initialized or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("preflight did not keep CUDA hidden and uninitialized")
    schedule = _schedule_evidence(membership)
    return {
        "schema": "priority25.f1_beleriand_data_preflight.v1",
        "kind": "fold1_data_and_membership_preflight",
        "status": "passed",
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "read-only data identity/membership validation; no fit, inference, plan staging, or claim",
        "campaign_root": str(campaign_root),
        "data_root": str(data_root),
        "source_root": str(source_root),
        "source_revision": SOURCE_REVISION,
        "fold_manifest": fold_evidence,
        "portable_registry": registry_evidence,
        "portable_archive_manifest": archive_evidence,
        "data_files": files,
        "membership": membership,
        "static_schedule": schedule,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_cuda_build": torch.version.cuda,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_is_initialized": cuda_initialized,
            "torch_threads": torch.get_num_threads(),
        },
        "portable_manifest_environment": {
            "used_by_canonical_loader": False,
            "inherited_value_cleared": inherited_portable_manifest is not None,
            "basis": "B source size/mtime metadata matched the sealed fold identity; canonical native identity route retained.",
        },
        "source_map_check": {
            "status": "deferred_to_root",
            "reason": "The local deep-terminal receipt reports 138 source files verified but omits the checkpoint contract source map; this helper does not load checkpoint payloads.",
        },
        "limitations": [
            "Static update counts are not observed F1 training progress.",
            "No source-map rehash of the 138 checkpoint contract entries was performed here.",
            "No provider/cell claim, runtime resource admission, training, prediction, or quality grade is established.",
        ],
    }


def parse_args() -> argparse.Namespace:
    """Parse pinned campaign input paths and the create-only receipt destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=REMOTE_ROOT)
    parser.add_argument("--source-root", type=Path, default=REMOTE_ROOT / "source-a266b036")
    parser.add_argument("--fold-manifest", type=Path, default=REMOTE_ROOT / "splits/fold1.json")
    parser.add_argument(
        "--portable-registry",
        type=Path,
        default=REMOTE_ROOT / "support-source/terminal-v1/PORTABLE_SOURCES.json",
    )
    parser.add_argument(
        "--archive-manifest",
        type=Path,
        default=REMOTE_ROOT / "support-source/terminal-v1/PORTABLE_ARCHIVES.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REMOTE_ROOT / "receipts/F1_BELERIAND_PREFLIGHT_V1.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the preflight and create its receipt only after all checks pass."""
    sys.dont_write_bytecode = True
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        receipt = run_preflight(args)
        with args.output.resolve().open("x", encoding="utf-8") as output:
            json.dump(receipt, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as error:
        LOG.error("F1 preflight failed: %s: %s", type(error).__name__, error)
        raise
    LOG.info("Created create-only F1 preflight receipt: %s", args.output)


if __name__ == "__main__":
    main()
