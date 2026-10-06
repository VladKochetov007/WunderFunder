#!/usr/bin/env python3
"""Fail-closed source and recipe gate for preregistered C1-C4 student fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np

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


def resolve_registered_directory(
    path: Path, relative: str, label: str, *, repository_root: Path = ROOT
) -> Path:
    """Require an input directory to resolve to its exact registered repository path."""
    repository = repository_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    expected = (repository / relative).resolve(strict=True)
    if resolved != expected:
        raise ValueError(f"{label} must resolve to the registered path {relative}")
    try:
        resolved.relative_to(repository)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    return resolved


def actual_artifact_hashes(
    families: dict[str, object], artifacts_root: Path
) -> dict[str, dict[str, str]]:
    """Hash each registered checkpoint and provenance file without loading tensors."""
    artifacts_root = artifacts_root.resolve(strict=True)
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
        run_root = (artifacts_root / run).resolve(strict=True)
        try:
            run_root.relative_to(artifacts_root)
        except ValueError as error:
            raise ValueError(f"registered run directory escapes artifacts root: {run}") from error
        checkpoint_path = (run_root / f"model_ep{epoch}.pt").resolve(strict=True)
        provenance_path = (run_root / "provenance.json").resolve(strict=True)
        for artifact_path in (checkpoint_path, provenance_path):
            try:
                artifact_path.relative_to(run_root)
            except ValueError as error:
                raise ValueError(
                    f"registered source artifact escapes run directory: {run}"
                ) from error
        result[run] = {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "provenance_sha256": sha256_file(provenance_path),
        }
    return result


def validate_files(
    recipe_path: Path,
    source_families_path: Path,
    realized_snapshot_path: Path,
    split_manifest_path: Path,
    expected_snapshot_sha256: str,
    expected_cache_metadata_sha256: str,
    artifacts_root: Path,
    teacher_cache_root: Path,
) -> dict[str, object]:
    """Load the sealed inputs and run the exact campaign validator."""
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.bandit_campaign_recipe import (
        RECIPE_VARIANTS,
        validate_bandit_recipe,
        verify_derived_distillation_targets,
    )
    from wnn.distillation_cache import validate_distillation_cache
    from wnn.splits import load_campaign_split

    recipe, recipe_bytes = read_object(recipe_path)
    arm = recipe.get("arm")
    if not isinstance(arm, str) or arm not in RECIPE_VARIANTS:
        raise ValueError("recipe does not name a registered C1-C4 arm")
    root_path = ROOT.resolve(strict=True)
    resolved_recipe = recipe_path.resolve(strict=True)
    try:
        relative_recipe_path = resolved_recipe.relative_to(root_path).as_posix()
    except ValueError as error:
        raise ValueError("recipe path must resolve within this checkout") from error
    expected_recipe_path = str(RECIPE_VARIANTS[arm]["recipe_path"])
    if relative_recipe_path != expected_recipe_path:
        raise ValueError("recipe path differs from the frozen arm preregistration")
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
    expected_cache_root = str(RECIPE_VARIANTS[arm]["cache_root"])
    if cache_root.as_posix() != expected_cache_root:
        raise ValueError("recipe cache root differs from its frozen campaign root")
    resolved_cache_root = resolve_registered_directory(
        ROOT / cache_root, expected_cache_root, "recipe cache root"
    )
    resolved_teacher_cache_root = resolve_registered_directory(
        teacher_cache_root, "cache", "teacher prediction cache root"
    )
    resolved_artifacts_root = resolve_registered_directory(
        artifacts_root, "artifacts/runs", "source artifact root"
    )
    metadata_path = (resolved_cache_root / "metadata.json").resolve(strict=True)
    try:
        metadata_path.relative_to(resolved_cache_root)
    except ValueError as error:
        raise ValueError("cache metadata escapes its registered cache root") from error
    cache_metadata, cache_bytes = read_object(metadata_path)
    cache_metadata_sha256 = hashlib.sha256(cache_bytes).hexdigest()
    if cache_metadata_sha256 != expected_cache_metadata_sha256:
        raise ValueError(
            "cache metadata and its derived-array digest manifest are not controller-pinned"
        )
    realized_sha256 = hashlib.sha256(realized_bytes).hexdigest()
    recipe_report = validate_bandit_recipe(
        recipe,
        source_families,
        realized_snapshot,
        cache_metadata,
        split_manifest=split_manifest,
        recipe_path=relative_recipe_path,
        recipe_sha256=hashlib.sha256(recipe_bytes).hexdigest(),
        sealed_families_sha256=hashlib.sha256(source_bytes).hexdigest(),
        realized_snapshot_sha256=realized_sha256,
        expected_realized_snapshot_sha256=expected_snapshot_sha256,
        actual_artifact_hashes=actual_artifact_hashes(realized_snapshot, resolved_artifacts_root),
    )
    split = load_campaign_split(split_manifest_path, ROOT)
    recipe_options = RECIPE_VARIANTS[arm]
    variant = str(recipe_options["variant"])
    require_families = float(recipe_options["family_weight"]) > 0
    cache_validation = validate_distillation_cache(
        resolved_cache_root,
        ROOT,
        split,
        variant,
        require_families=require_families,
    )
    if (
        cache_validation["metadata_sha256"] != expected_cache_metadata_sha256
        or cache_validation["metadata"] != cache_metadata
    ):
        raise ValueError("cache metadata changed during the pinned array validation")
    target_orders: dict[str, list[int]] = {}
    for split_name in ("train", "valid"):
        with np.load(ROOT / "cache" / f"{split_name}_targets.npz", allow_pickle=False) as targets:
            target_orders[split_name] = list(map(int, targets["row_group"]))
    derived_validation = verify_derived_distillation_targets(
        resolved_cache_root,
        resolved_teacher_cache_root,
        cache_metadata,
        realized_snapshot,
        target_orders,
        {"train": list(split.train_groups), "valid": list(split.valid_groups)},
        variant=variant,
        require_families=require_families,
        chunk_rows=8,
    )
    return {
        **recipe_report,
        "recipe_path": str(recipe_path),
        "cache_metadata_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        "verified_cache_files": cache_validation["verified_files"],
        "derived_target_closure": derived_validation,
        "controller_pinned_cache_metadata_sha256": expected_cache_metadata_sha256,
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
    parser.add_argument("--cache-metadata-sha256", required=True)
    parser.add_argument("--artifacts-root", type=Path, default=ROOT / "artifacts" / "runs")
    parser.add_argument("--teacher-cache-root", type=Path, default=ROOT / "cache")
    args = parser.parse_args()
    try:
        report = validate_files(
            resolve_from_root(args.recipe),
            resolve_from_root(args.source_families),
            resolve_from_root(args.realized_source_snapshot),
            resolve_from_root(args.split_manifest),
            args.source_snapshot_sha256,
            args.cache_metadata_sha256,
            resolve_from_root(args.artifacts_root),
            resolve_from_root(args.teacher_cache_root),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"campaign recipe validation failed: {error}") from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
