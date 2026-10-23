from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from wnn.data import SEQUENCE_LENGTH
from wnn.splits import CampaignSplit, file_sha256, load_campaign_split
from wnn.training.sequence_cache import ENSEMBLE_KIND, MAX_CACHE_MANIFEST_BYTES, TeacherCache
from wnn.training.sequence_cache_ensemble_lineage import common_provenance

PARENT_COUNT = 3


def _canonical_sha256(value: object) -> str:
    """Return a stable digest for JSON-compatible cache provenance."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _teacher_recipe(producer: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Extract a seed and complete normalized teacher recipe from one producer."""
    if producer.get("kind") == ENSEMBLE_KIND:
        raise ValueError("mean cache inputs must be three single-checkpoint teachers")
    if "checkpoint_sha256" not in producer or not isinstance(producer.get("config"), dict):
        raise ValueError("teacher producer must identify a checkpoint and recipe")
    config = producer["config"]
    seed = config.get("seed")
    model = config.get("model")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("teacher recipe must contain an unsigned seed")
    if (
        not isinstance(model, dict)
        or type(model.get("hint_dim")) is not int
        or model.get("hint_dim") != 0
    ):
        raise ValueError("teacher recipe must use zero representation hints")
    recipe = dict(config)
    recipe.pop("seed")
    return recipe, seed


def _parent_records(caches: list[TeacherCache]) -> dict[str, Any]:
    """Bind each source cache, physical checkpoint, recipe and provenance into the output."""
    parents: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    recipes: list[dict[str, Any]] = []
    common_provenances: list[dict[str, Any]] = []
    for cache in caches:
        producer = cache.metadata["producer"]
        recipe, seed = _teacher_recipe(producer)
        provenance = producer.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("teacher producer provenance must be a JSON object")
        common_provenances.append(common_provenance(producer, provenance))
        checkpoint_hash = producer["checkpoint_sha256"]
        recipe_hash = _canonical_sha256(recipe)
        producer_hash = _canonical_sha256(producer)
        provenance_hash = _canonical_sha256(provenance)
        manifest_hash = cache.identity["manifest_sha256"]
        parents.append(
            {
                "manifest_sha256": manifest_hash,
                "checkpoint_sha256": checkpoint_hash,
                "seed": seed,
                "recipe_sha256": recipe_hash,
                "producer_sha256": producer_hash,
                "provenance_sha256": provenance_hash,
                "producer": producer,
                "provenance": provenance,
            }
        )
        references.append(
            {
                "manifest_sha256": manifest_hash,
                "checkpoint_sha256": checkpoint_hash,
                "seed": seed,
                "recipe_sha256": recipe_hash,
                "producer_sha256": producer_hash,
                "provenance_sha256": provenance_hash,
            }
        )
        recipes.append(recipe)
    if len({parent["checkpoint_sha256"] for parent in parents}) != PARENT_COUNT:
        raise ValueError("teacher caches must come from three distinct checkpoints")
    if len({parent["manifest_sha256"] for parent in parents}) != PARENT_COUNT:
        raise ValueError("teacher caches must have three distinct manifests")
    if len({parent["seed"] for parent in parents}) != PARENT_COUNT:
        raise ValueError("teacher caches must use three distinct seeds")
    if recipes[1:] != recipes[:1] * 2:
        raise ValueError("teacher recipes must match exactly except for seed")
    if common_provenances[1:] != common_provenances[:1] * 2:
        raise ValueError("teacher source, runtime, data or normalizer provenance differs")
    shared_provenance_hash = _canonical_sha256(common_provenances[0])
    recipe = recipes[0]
    composition = {
        "schema": "sequence-teacher-composition-v1",
        "method": ENSEMBLE_KIND,
        "clip_before_mean": False,
        "shared_provenance_sha256": shared_provenance_hash,
        "parents": references,
    }
    producer = {
        "kind": ENSEMBLE_KIND,
        "split_sha256": caches[0].metadata["split_sha256"],
        "fit_members": caches[0].metadata["producer"]["fit_members"],
        "recipe": recipe,
        "recipe_sha256": _canonical_sha256(recipe),
        "shared_provenance_sha256": shared_provenance_hash,
        "composition": composition,
        "composition_sha256": _canonical_sha256(composition),
        "parents": parents,
    }
    return producer


def _cache_values(cache: TeacherCache, member: tuple[str, int]) -> np.ndarray:
    """Read and validate one full FP32 two-target sequence from a teacher cache."""
    entry = cache.entries[member]
    values = np.load(cache._entry_path(entry), mmap_mode="r", allow_pickle=False)
    if values.dtype != np.float32 or values.shape != (SEQUENCE_LENGTH, 2):
        raise ValueError("teacher arrays must be exactly float32 20000x2")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite teacher cache")
    return values


def _mean_sequence(caches: list[TeacherCache], member: tuple[str, int]) -> np.ndarray:
    """Accumulate three sequence predictions in FP32 without clipping members."""
    mean = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for cache in caches:
            np.add(mean, _cache_values(cache, member), out=mean)
        np.divide(mean, np.float32(PARENT_COUNT), out=mean)
    if not np.isfinite(mean).all():
        raise ValueError("mean teacher cache contains nonfinite values")
    return mean


def _write_sequence(root: Path, member: tuple[str, int], values: np.ndarray) -> dict[str, Any]:
    """Write one complete sequence and return its manifest checksum record."""
    source, group = member
    name = f"{source}-{group}.npy"
    path = root / name
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"split": source, "row_group": group, "file": name, "sha256": file_sha256(path)}


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Publish a cache manifest only after all referenced sequence hashes exist."""
    payload = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if len(payload) > MAX_CACHE_MANIFEST_BYTES:
        raise ValueError("merged teacher cache manifest exceeds the consumer size limit")
    temporary = path / "manifest.json.partial"
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path / "manifest.json")


def _verify_data_unchanged(data_root: Path, split_manifest: Path, expected: CampaignSplit) -> None:
    """Reload the frozen split and source identities after all parent arrays were streamed."""
    current = load_campaign_split(split_manifest, data_root)
    if current.sha256 != expected.sha256 or current != expected:
        raise ValueError("split manifest or raw data changed during cache merge")


def merge_teacher_caches(
    input_roots: list[Path],
    data_root: Path,
    split_manifest: Path,
    output: Path,
) -> Path:
    """Create a provenance-bound equal mean of exactly three full fit teacher caches."""
    if len(input_roots) != PARENT_COUNT:
        raise ValueError("exactly three --input cache roots are required")
    split = load_campaign_split(split_manifest, data_root)
    members = [("train", group) for group in split.train_groups] + [
        ("valid", group) for group in split.valid_groups
    ]
    caches = [TeacherCache(root, split, members) for root in input_roots]
    for cache in caches:
        if cache.metadata["hint_dim"] != 0:
            raise ValueError("mean teacher caches require hint_dim=0")
    producer = _parent_records(caches)
    target = output.parent.resolve() / output.name
    if target.name in ("", ".", ".."):
        raise ValueError("output path must name a new cache directory")
    input_paths = [root.resolve() for root in input_roots]
    if any(target == root or root in target.parents for root in input_paths):
        raise ValueError("output path must be outside every input cache")
    if not target.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    lock = target.with_name(f".{target.name}.merge.lock")
    stage: Path | None = None
    lock_created = False
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write("merge in progress\n")
        lock_created = True
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output already exists: {target}")
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.merge-", dir=target.parent))
        entries = [
            _write_sequence(stage, member, _mean_sequence(caches, member)) for member in members
        ]
        for cache in caches:
            cache.verify_unchanged()
        _verify_data_unchanged(data_root, split_manifest, split)
        manifest = {
            "schema": "sequence-teacher-cache-v1",
            "complete": True,
            "input_schema": "raw112-v1",
            "membership": "fit",
            "split_sha256": split.sha256,
            "hint_dim": 0,
            "sequence_length": SEQUENCE_LENGTH,
            "producer": producer,
            "sequences": entries,
        }
        _write_manifest(stage, manifest)
        for cache in caches:
            cache.verify_unchanged()
        _verify_data_unchanged(data_root, split_manifest, split)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output appeared during merge: {target}")
        stage.rename(target)
        stage = None
        return target
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        if lock_created:
            lock.unlink(missing_ok=True)


def main() -> None:
    """Merge three independently seeded full fit teacher exports into one cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) != PARENT_COUNT:
        parser.error("--input must be provided exactly three times")
    merge_teacher_caches(args.input, args.data_root, args.split_manifest, args.output)
