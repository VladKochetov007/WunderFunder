#!/usr/bin/env python3
"""Fail-closed source and recipe gate for preregistered C1-C4 student fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "research/campaign_2026-10-06/bandit_continuation"
DEFAULT_SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"


def sha256_file(path: Path) -> str:
    """Hash one source/checkpoint file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> tuple[dict[str, object], bytes]:
    """Read one JSON object and return its exact source bytes for hashing."""
    payload = path.read_bytes()
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, object], decoded), payload


def resolve_from_root(path: Path) -> Path:
    """Resolve command-line paths against this checkout without accepting cwd drift."""
    return path if path.is_absolute() else ROOT / path


def actual_artifact_hashes(
    families: dict[str, object], artifacts_root: Path
) -> dict[str, dict[str, str]]:
    """Hash each registered checkpoint and provenance file without loading tensors."""
    entries = families.get("families")
    if not isinstance(entries, list):
        raise ValueError("source-family snapshot has no family list")
    result: dict[str, dict[str, str]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("source-family snapshot contains an invalid family row")
        entry = cast(dict[str, object], raw_entry)
        run = entry.get("run")
        epoch = entry.get("epoch_zero_based")
        if not isinstance(run, str) or not isinstance(epoch, int) or isinstance(epoch, bool):
            raise ValueError("source-family snapshot has invalid run/epoch fields")
        run_root = artifacts_root / run
        result[run] = {
            "checkpoint_sha256": sha256_file(run_root / f"model_ep{epoch}.pt"),
            "provenance_sha256": sha256_file(run_root / "provenance.json"),
        }
    return result


def validate_files(
    recipe_path: Path,
    source_families_path: Path,
    realized_snapshot_path: Path,
    split_manifest_path: Path,
    expected_snapshot_sha256: str,
    artifacts_root: Path,
) -> dict[str, object]:
    """Load the sealed inputs and run the exact campaign validator."""
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.bandit_campaign_recipe import validate_bandit_recipe

    recipe, _ = read_object(recipe_path)
    source_families, source_bytes = read_object(source_families_path)
    realized_snapshot, realized_bytes = read_object(realized_snapshot_path)
    split_manifest, split_bytes = read_object(split_manifest_path)
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    if split_sha256 != "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d":
        raise ValueError("fold-0 split manifest bytes differ from the sealed SHA-256")
    cache_root_value = recipe.get("cache_root")
    if not isinstance(cache_root_value, str):
        raise ValueError("recipe cache_root must be a relative path")
    cache_root = Path(cache_root_value)
    if cache_root.is_absolute() or len(cache_root.parts) == 0:
        raise ValueError("recipe cache_root must stay inside the repository")
    cache_metadata, cache_bytes = read_object(ROOT / cache_root / "metadata.json")
    result = validate_bandit_recipe(
        recipe,
        source_families,
        realized_snapshot,
        cache_metadata,
        split_manifest=split_manifest,
        sealed_families_sha256=hashlib.sha256(source_bytes).hexdigest(),
        realized_snapshot_sha256=hashlib.sha256(realized_bytes).hexdigest(),
        expected_realized_snapshot_sha256=expected_snapshot_sha256,
        actual_artifact_hashes=actual_artifact_hashes(realized_snapshot, artifacts_root),
    )
    return {
        **result,
        "recipe_path": str(recipe_path),
        "cache_metadata_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        "source_snapshot_path": str(realized_snapshot_path),
        "split_manifest_sha256": split_sha256,
    }


def main() -> None:
    """Validate one preregistered arm before issuing either scientific fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument(
        "--source-families",
        type=Path,
        default=DEFAULT_CAMPAIGN / "SOURCE_FAMILIES.json",
    )
    parser.add_argument("--realized-source-snapshot", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--artifacts-root", type=Path, default=ROOT / "artifacts" / "runs")
    args = parser.parse_args()
    try:
        report = validate_files(
            resolve_from_root(args.recipe),
            resolve_from_root(args.source_families),
            resolve_from_root(args.realized_source_snapshot),
            resolve_from_root(args.split_manifest),
            args.source_snapshot_sha256,
            resolve_from_root(args.artifacts_root),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"campaign recipe validation failed: {error}") from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
