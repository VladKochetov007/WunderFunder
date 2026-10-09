from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PACKET_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PACKET_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "src"))

from wnn.parsed_sequence_cache_reader import ParsedSequenceCache  # noqa: E402

from wnn import fold0_assets  # noqa: E402
from wnn.splits import connected_components, load_campaign_split  # noqa: E402

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "parsed_cache_source_stage"
SPLIT_RELATIVE = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
REGISTRY_RELATIVE = Path("research/campaign_2026-10-04/portable_inputs.json")
MEMBERSHIP_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "loss_pair_r1_membership_fold0_f03_s20261009.json"
)


def _write_sparse_cache_arrays(cache_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Create header-valid sparse NPY mappings and bind their local stat records."""
    validation = json.loads((FIXTURE_ROOT / "reader_validation.json").read_text())
    for name, entry in manifest["arrays"].items():
        path = cache_root / entry["path"]
        mapped = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.dtype(entry["dtype"]),
            shape=tuple(entry["shape"]),
        )
        mapped._mmap.close()
        observed = path.stat()
        validation["arrays"][name]["stat"] = {
            "device": int(observed.st_dev),
            "inode": int(observed.st_ino),
            "bytes": int(observed.st_size),
            "mtime_ns": int(observed.st_mtime_ns),
            "ctime_ns": int(observed.st_ctime_ns),
        }
    allocated_bytes = sum(path.stat().st_blocks * 512 for path in cache_root.glob("*.npy"))
    assert allocated_bytes < 64 * 1024 * 1024
    return validation


def _write_minimal_split_sources(source_root: Path, split: dict[str, Any]) -> None:
    """Write compact graph/index fixtures consistent with the pinned split."""
    cache_root = source_root / "cache"
    cache_root.mkdir(parents=True)
    nodes = [
        (split_name, int(row["row_group"]), int(row["component"]))
        for split_name in ("train", "valid")
        for row in split["sequences"][split_name]
    ]
    nodes.sort(key=lambda node: (node[2], node[0], node[1]))
    by_component: dict[int, list[int]] = defaultdict(list)
    for index, (_, _, component) in enumerate(nodes):
        by_component[component].append(index)
    successors = np.full(len(nodes), -1, dtype=np.int64)
    for indices in by_component.values():
        for left, right in zip(indices, indices[1:], strict=False):
            successors[left] = right
    labels = connected_components(successors)
    expected = np.asarray([component for _, _, component in nodes], dtype=np.int64)
    if not np.array_equal(labels, expected):
        raise AssertionError("split component IDs are not contiguous canonical labels")
    np.savez_compressed(
        cache_root / "chains.npz",
        **{
            "next": successors,
            "split": np.asarray([node[0] for node in nodes]),
            "row_group": np.asarray([node[1] for node in nodes], dtype=np.int64),
        },
    )

    evaluation_groups = set(map(int, split["evaluation"]["valid"]))
    valid_groups = sorted(int(row["row_group"]) for row in split["sequences"]["valid"])
    np.savez_compressed(
        cache_root / "chain_folds.npz",
        split=np.asarray(["valid"] * len(valid_groups)),
        row_group=np.asarray(valid_groups, dtype=np.int64),
        fold=np.asarray(
            [0 if group in evaluation_groups else 1 for group in valid_groups],
            dtype=np.int64,
        ),
    )

    for split_name in ("train", "valid"):
        rows = sorted(split["sequences"][split_name], key=lambda row: int(row["row_group"]))
        row_groups = np.asarray([int(row["row_group"]) for row in rows], dtype="<i8")
        seq_ix = np.asarray([int(row["seq_ix"]) for row in rows], dtype="<i8")
        digest = hashlib.sha256()
        digest.update(row_groups.tobytes())
        digest.update(seq_ix.tobytes())
        expected_sha = split["sources"][f"cache/{split_name}_targets.npz"]["index_sha256"]
        if digest.hexdigest() != expected_sha:
            raise AssertionError(f"{split_name} target index order differs from the split")
        np.savez_compressed(
            cache_root / f"{split_name}_targets.npz",
            row_group=row_groups,
            seq_ix=seq_ix,
        )


def test_actual_cache_open_hashes_receipt_before_split_loader(tmp_path, monkeypatch) -> None:
    """Bridge the actual CPU cache fixture into the strict campaign split loader.

    Cache arrays are sparse, header-valid mappings because the source-stage fixture
    carries the real array hashes and sizes; this regression verifies receipt/hash
    propagation without copying, parsing, or hashing the 7.7 GB cache payload.
    """
    cache_root = tmp_path / "parsed_cache"
    cache_root.mkdir()
    manifest_path = cache_root / "manifest.json"
    source_verification_path = cache_root / "source_verification.json"
    membership_path = cache_root / "membership.json"
    validation_path = cache_root / "reader_validation.json"
    shutil.copyfile(FIXTURE_ROOT / "manifest.json", manifest_path)
    shutil.copyfile(FIXTURE_ROOT / "source_verification.json", source_verification_path)
    shutil.copyfile(CODE_ROOT / MEMBERSHIP_RELATIVE, membership_path)

    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    original_validation_raw = (FIXTURE_ROOT / "reader_validation.json").read_bytes()
    source_stage_receipt = json.loads((FIXTURE_ROOT / "source_stage_receipt.json").read_text())
    assert hashlib.sha256(manifest_raw).hexdigest() == source_stage_receipt["cache_manifest_sha256"]
    assert (
        hashlib.sha256(original_validation_raw).hexdigest()
        == source_stage_receipt["reader_validation_sha256"]
    )
    assert (
        hashlib.sha256(source_verification_path.read_bytes()).hexdigest()
        == (source_stage_receipt["raw_source_verification_sha256"])
    )
    assert (
        hashlib.sha256(membership_path.read_bytes()).hexdigest()
        == (manifest["membership_artifact"]["sha256"])
    )

    validation = _write_sparse_cache_arrays(cache_root, manifest)
    validation_raw = (
        json.dumps(validation, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    validation_path.write_bytes(validation_raw)
    validation_sha = hashlib.sha256(validation_raw).hexdigest()

    parsed_cache = ParsedSequenceCache.open(
        manifest_path,
        expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        validation_receipt_path=validation_path,
        expected_validation_receipt_sha256=validation_sha,
        expected_cache_key=manifest["cache_key"],
        expected_split_manifest_sha256=manifest["split_manifest_sha256"],
        expected_fold=0,
    )
    try:
        assert parsed_cache.preverified_sources.validation_receipt_sha256 == validation_sha
        assert len(parsed_cache.preverified_sources.validation_receipt_sha256) == 64

        split_path = CODE_ROOT / SPLIT_RELATIVE
        split_manifest = json.loads(split_path.read_text())
        source_root = tmp_path / "campaign_source"
        _write_minimal_split_sources(source_root, split_manifest)

        receipt_path = tmp_path / "fold0_assets_validation.json"
        receipt_path.write_text("{}\n")
        registry_path = CODE_ROOT / REGISTRY_RELATIVE
        monkeypatch.setenv("WNN_FOLD0_ASSET_VALIDATION_RECEIPT", str(receipt_path))
        monkeypatch.setenv(
            "WNN_FOLD0_ASSET_VALIDATION_SHA256",
            hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        )
        monkeypatch.setenv("WNN_PORTABLE_DATA_MANIFEST", str(registry_path))
        monkeypatch.setattr(
            fold0_assets,
            "validate_preverified_fold0_assets",
            lambda *_args, **_kwargs: {},
        )
        loaded_split = load_campaign_split(
            split_path,
            source_root,
            preverified_sources=parsed_cache.preverified_sources,
        )
        assert loaded_split.fold == 0
        assert loaded_split.sha256 == manifest["split_manifest_sha256"]
        assert len(loaded_split.evaluation_groups) == 468
    finally:
        parsed_cache.close()
