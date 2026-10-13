from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast


def sha256_file(path: Path) -> str:
    """Hash a mounted input in bounded blocks without loading its payload."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    """Read only bounded metadata at the cloud trust boundary."""
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ValueError(f"metadata exceeds {maximum_bytes} bytes: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def npy_header(path: Path) -> tuple[tuple[int, ...], str]:
    """Read the bounded NPY header only; never touch a target-array payload."""
    import numpy as np

    with path.open("rb") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(source)
        elif version in {(2, 0), (3, 0)}:
            shape, _, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError(f"unsupported NPY version {version}: {path}")
    return tuple(int(value) for value in shape), str(dtype)


def write_json(path: Path, value: object) -> None:
    """Write the preflight receipt atomically inside the worker root."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_external_inputs(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Verify exact Kaggle mount slugs, filenames, lengths and content hashes."""
    observations: list[dict[str, Any]] = []
    for item in manifest["external_inputs"]:
        logical = str(item["path"])
        path = root / logical
        if not path.is_symlink():
            raise ValueError(f"external input is not the kernel-created mount symlink: {logical}")
        resolved = path.resolve(strict=True)
        dataset_slug = str(item["dataset"]).split("/", 1)[1]
        if resolved.parent.name != dataset_slug or resolved.name != item["filename"]:
            raise ValueError(
                f"mounted dataset identity differs for {logical}: "
                f"expected {dataset_slug}/{item['filename']}, got {resolved}"
            )
        started = time.monotonic()
        size = resolved.stat().st_size
        digest = sha256_file(resolved)
        observation = {
            "path": logical,
            "dataset": item["dataset"],
            "filename": item["filename"],
            "size": size,
            "sha256": digest,
            "expected_size": int(item["size"]),
            "expected_sha256": str(item["sha256"]),
            "resolved_path": str(resolved),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        observations.append(observation)
        if size != item["size"] or digest != item["sha256"]:
            raise ValueError(f"mounted input content identity differs: {logical}")
    return observations


def validate_recipe_inputs(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check raw-data, F0 split, CP5, propensity and cache ancestry before profiles."""
    portable_path = root / str(manifest["portable_registry_path"])
    portable = read_json(portable_path, maximum_bytes=1_000_000)
    registry_digest = sha256_file(portable_path)
    if registry_digest != manifest["portable_registry_sha256"]:
        raise ValueError("portable raw-data registry digest differs")

    by_path = {item["path"]: item for item in manifest["external_inputs"]}
    for logical in ("datasets/train.parquet", "datasets/valid.parquet"):
        registry_entry = portable["sources"].get(logical)
        observed_entry = by_path.get(logical)
        if registry_entry is None or observed_entry is None:
            raise ValueError(f"portable registry or mount is missing {logical}")
        original = registry_entry.get("original_identity", {})
        if (
            int(original.get("bytes", -1)) != observed_entry["size"]
            or registry_entry.get("sha256") != observed_entry["sha256"]
        ):
            raise ValueError(f"portable registry and mounted raw source disagree: {logical}")

    split_path = root / str(manifest["split_manifest_path"])
    split = read_json(split_path, maximum_bytes=5_000_000)
    split_digest = sha256_file(split_path)
    if split_digest != manifest["split_manifest_sha256"] or split.get("fold") != 0:
        raise ValueError("source-qualified fold-0 split identity differs")
    fit = split.get("fit", {})
    evaluation = split.get("evaluation", {})
    if (
        len(fit.get("train", [])) != 10535
        or len(fit.get("valid", [])) != 1405
        or len(evaluation.get("valid", [])) != 468
    ):
        raise ValueError("fold-0 fitting or fixed evaluation membership count differs")

    subset = read_json(root / str(manifest["fit_subset_path"]), maximum_bytes=5_000_000)
    if (
        subset.get("experiment_id") != manifest["experiment_id"]
        or subset.get("split_manifest_sha256") != manifest["split_manifest_sha256"]
        or subset.get("selected_group_counts") != {"train": 10535, "valid": 1405}
        or subset.get("evaluation_group_count") != 468
        or subset.get("selected_component_count") != subset.get("eligible_component_count")
    ):
        raise ValueError("full-component fit manifest is not the complete clean F0 membership")

    parent_path = root / str(manifest["parent_checkpoint_path"])
    parent_prov_path = root / str(manifest["parent_provenance_path"])
    parent = read_json(parent_prov_path, maximum_bytes=2_000_000)
    parent_args = parent.get("args", {})
    expected_parent = manifest["parent"]
    if (
        parent.get("seed") != 4
        or parent.get("split_manifest_sha256") != manifest["split_manifest_sha256"]
        or parent_args.get("init") != expected_parent["initializer"]
        or parent_args.get("checkpoint_epoch") != 5
        or parent_args.get("hidden") != 256
        or parent_args.get("layers") != 2
        or parent_args.get("aux_struct") != 0.3
        or parent_args.get("aux_next") != 0.3
        or parent_args.get("propensity") != "c20260929_prop_f0"
        or parent_args.get("prop_cache") is not True
        or parent.get("initial_checkpoint_sha256") != expected_parent["initializer_sha256"]
        or parent.get("propensity_checkpoint_sha256") != expected_parent["propensity_sha256"]
        or parent.get("distillation_cache", {}).get("metadata_sha256")
        != manifest["teacher_cache"]["metadata_sha256"]
    ):
        raise ValueError("seed-4 clean-T3 CP5 provenance differs from the reviewed parent")
    if not parent_path.is_symlink() or parent_path.stat().st_size != expected_parent["checkpoint_bytes"]:
        raise ValueError("seed-4 CP5 initializer is missing or has the wrong length")
    import torch

    checkpoint = torch.load(parent_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", {})
    mean_digest = hashlib.sha256(state["mean"].contiguous().numpy().tobytes()).hexdigest()
    std_digest = hashlib.sha256(state["std"].contiguous().numpy().tobytes()).hexdigest()
    head_shape = tuple(state["head.2.weight"].shape)
    if (
        checkpoint.get("n_in") != expected_parent["n_in"]
        or tuple(state["mean"].shape) != (expected_parent["n_in"],)
        or tuple(state["std"].shape) != (expected_parent["n_in"],)
        or mean_digest != expected_parent["mean_sha256"]
        or std_digest != expected_parent["std_sha256"]
        or head_shape != (expected_parent["aux_width"], 256)
    ):
        raise ValueError("seed-4 CP5 model dimensions or normalization buffers differ")

    cache = manifest["teacher_cache"]
    metadata_path = root / cache["metadata_path"]
    metadata = read_json(metadata_path, maximum_bytes=2_000_000)
    if sha256_file(metadata_path) != cache["metadata_sha256"]:
        raise ValueError("clean-T3 target-cache metadata digest differs")
    if (
        metadata.get("split_manifest_sha256") != manifest["split_manifest_sha256"]
        or metadata.get("cache_profile") != "cleanfold_homogeneous_strong_t3_v1"
        or metadata.get("target_contract", {}).get("formula")
        != "float32(0.8125 * float32_mean(three_epoch2_predictions))"
    ):
        raise ValueError("clean-T3 target-cache profile or target arithmetic differs")
    cache_arrays: list[dict[str, Any]] = []
    for key, expected_shape in (
        ("main_homogeneous_train", (10535, 20000, 2)),
        ("main_homogeneous_valid", (1405, 20000, 2)),
    ):
        entry = metadata["files"][key]
        logical = str(Path(cache["root"]) / entry["path"])
        descriptor = by_path[logical]
        if (
            tuple(entry["shape"]) != expected_shape
            or entry.get("dtype") != "float32"
            or int(entry["bytes"]) != descriptor["size"]
            or entry.get("sha256") != descriptor["sha256"]
        ):
            raise ValueError(f"clean-T3 target descriptor differs: {key}")
        observed_shape, observed_dtype = npy_header(root / logical)
        if observed_shape != expected_shape or observed_dtype != "float32":
            raise ValueError(f"clean-T3 target NPY header differs: {logical}")
        cache_arrays.append(
            {"path": logical, "shape": observed_shape, "dtype": observed_dtype, "bytes": descriptor["size"]}
        )

    if len(metadata.get("sources", [])) != 3 or len(metadata.get("source_coefficients", [])) != 3:
        raise ValueError("clean-T3 cache does not name exactly three teacher sources")
    for teacher in metadata["sources"]:
        for field, digest_field in (("checkpoint_path", "checkpoint_sha256"),
                                    ("provenance_path", "provenance_sha256")):
            source_path = Path(teacher[field])
            marker = "artifacts/runs/"
            if marker not in source_path.as_posix():
                raise ValueError("teacher ancestry path is not campaign-relative")
            logical = marker + source_path.as_posix().split(marker, 1)[1]
            item = by_path.get(logical)
            if item is None or item["sha256"] != teacher[digest_field]:
                raise ValueError(f"clean-T3 teacher source identity is not mounted: {logical}")

    return {
        "portable_registry_sha256": registry_digest,
        "split_manifest_sha256": split_digest,
        "parent_run": expected_parent["run"],
        "parent_checkpoint_sha256": expected_parent["checkpoint_sha256"],
        "parent_mean_sha256": expected_parent["mean_sha256"],
        "parent_std_sha256": expected_parent["std_sha256"],
        "parent_n_in": expected_parent["n_in"],
        "parent_aux_width": expected_parent["aux_width"],
        "cache_metadata_sha256": cache["metadata_sha256"],
        "cache_arrays": cache_arrays,
        "fit_components": subset["selected_component_count"],
        "fixed_evaluation_groups": 468,
    }


def main() -> None:
    """Verify actual Kaggle inputs before any profile or fit worker starts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": "qat-kaggle-actual-input-preflight-r1",
        "status": "running",
        "training_source_revision": "173cce08e9c7f13caccd1c3090419dd044aa702d",
        "started_unix_seconds": time.time(),
        "inputs": [],
    }
    try:
        manifest = read_json(args.repo_root / args.identity_manifest, maximum_bytes=2_000_000)
        receipt["identity_manifest_sha256"] = sha256_file(args.repo_root / args.identity_manifest)
        receipt["inputs"] = verify_external_inputs(args.repo_root, manifest)
        receipt["recipe_validation"] = validate_recipe_inputs(args.repo_root, manifest)
        receipt["status"] = "pass"
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        receipt["bytes_hashed"] = sum(int(item["size"]) for item in receipt["inputs"])
        write_json(args.output, receipt)
    except Exception as error:
        receipt["status"] = "fail"
        receipt["failure"] = f"{type(error).__name__}: {error}"
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(args.output, receipt)
        raise


if __name__ == "__main__":
    main()
