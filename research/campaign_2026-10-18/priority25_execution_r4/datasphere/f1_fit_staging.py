"""Fold-1 input identity and data staging for the one-pair DataSphere fit."""

from __future__ import annotations

import json
import shutil
import tarfile
from argparse import Namespace
from pathlib import Path
from typing import Any

from f1_fit_recipe import FOLD1_SHA256, PROFILE_SPEC_SHA256, SOURCE_COMMIT
from profile_staging import (
    assemble,
    copy_mounted,
    discover_mounted_files,
    relative_path,
    safe_extract,
    sealed_hash,
    sha256_file,
)

# Parsed packet and registry objects are intentionally unstructured JSON.


def read_object(path: Path, label: str) -> dict[str, Any]:
    """Read one JSON object from an operator-controlled sealed packet file."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def check_packet_binding(profile: dict[str, Any], fit: dict[str, Any], profile_path: Path) -> None:
    """Bind the follow-on fit to the exact sealed profile, source and fold-1 identity."""
    if sha256_file(profile_path) != PROFILE_SPEC_SHA256:
        raise ValueError("base profile specification identity differs from the frozen packet")
    if profile.get("state") != "SEALED_FOR_PROFILE":
        raise ValueError("base profile specification is not sealed")
    if profile.get("source_commit") != SOURCE_COMMIT or fit.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("fit source commit differs from the canonical trainer")
    if fit.get("base_profile_spec_sha256") != PROFILE_SPEC_SHA256:
        raise ValueError("fit spec does not bind to the frozen base profile specification")
    for key in ("code_archive_sha256", "supplemental_archive_sha256"):
        if fit.get(key) != profile.get(key):
            raise ValueError(f"fit spec changes the sealed {key}")
    members = profile.get("supplemental_members")
    if not isinstance(members, dict) or members.get("splits/fold1.json") != FOLD1_SHA256:
        raise ValueError("base profile packet lacks the authorized fold-1 manifest")
    if fit.get("fold_manifest_sha256") != members["splits/fold1.json"]:
        raise ValueError("fit spec fold-1 manifest differs from the supplemental archive")
    if fit.get("fold_manifest_path") != "splits/fold1.json":
        raise ValueError("fit spec must select the sealed fold-1 manifest")


def _archive_expanded_size(path: Path) -> int:
    """Inspect archive member sizes to reserve extraction space without reading payloads."""
    with tarfile.open(path, "r:*") as archive:
        return sum(member.size for member in archive.getmembers() if member.isfile())


def prepare_fit(
    profile: dict[str, Any], fit: dict[str, Any], args: Namespace
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    """Stage hash-bound source, fold-1 metadata, parquet and target indexes."""
    if set(profile["dataset_slots"]) != {"DS1", "DS2", "DS3", "DS4"}:
        raise ValueError("base profile dataset slots differ from the sealed packet")
    if set(profile["mounted_inputs"]) != {"train", "valid"}:
        raise ValueError("fit requires only the mounted train and valid parquet splits")
    if set(profile["mounted_files"]) != {"cache/train_targets.npz", "cache/valid_targets.npz"}:
        raise ValueError("fit requires only the canonical train and valid target indexes")
    if len(args.datasets) != 4:
        raise ValueError("the sealed data packet requires exactly four dataset mounts")
    dataset_roots = dict(zip(profile["dataset_slots"], map(Path, args.datasets), strict=True))
    expected_names = {
        part["filename"] for item in profile["mounted_inputs"].values() for part in item["parts"]
    }
    expected_names.update(item["source_filename"] for item in profile["mounted_files"].values())
    discovered, mount_scans = discover_mounted_files(dataset_roots, expected_names)
    for split, entry in profile["mounted_inputs"].items():
        mounted_bytes = sum(int(discovered[part["filename"]]["bytes"]) for part in entry["parts"])
        if mounted_bytes != int(entry["bytes"]):
            raise ValueError(f"mounted {split} parquet parts differ from the sealed byte count")
    for logical, entry in profile["mounted_files"].items():
        if int(discovered[entry["source_filename"]]["bytes"]) != int(entry["bytes"]):
            raise ValueError(f"mounted target index differs from its sealed byte count: {logical}")

    staged_bytes = sum(
        int(item["bytes"])
        for item in (*profile["mounted_inputs"].values(), *profile["mounted_files"].values())
    )
    projected_bytes = staged_bytes + sum(
        _archive_expanded_size(path) for path in (args.code, args.supplemental)
    )
    reserve = int(profile["minimum_free_bytes_after_staging"])
    disk_before = shutil.disk_usage(Path.cwd()).free
    if disk_before < projected_bytes + reserve:
        raise OSError("insufficient free disk for fit staging and the required reserve")

    work = Path.cwd() / "rank01-f1-fit-work"
    if work.exists():
        raise FileExistsError("fit staging directory already exists; refusing stale-state reuse")
    source_root = work.joinpath(*relative_path(profile["source_root"]).parts)
    data_root = work.joinpath(*relative_path(profile["data_root"]).parts)
    source_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    safe_extract(args.code, source_root)
    required = (
        "scripts/train_rnn.py",
        "scripts/predict_run.py",
        "src/wnn/training/runner.py",
        "pyproject.toml",
        "uv.lock",
    )
    if not all((source_root / name).is_file() for name in required):
        raise ValueError("source archive lacks the canonical trainer or prediction entrypoint")

    from profile_recipe import check_runtime

    runtime = check_runtime(profile["expected_environment"])
    safe_extract(args.supplemental, data_root, profile["supplemental_members"])
    manifest_path = data_root.joinpath(*relative_path(fit["fold_manifest_path"]).parts)
    registry_path = data_root.joinpath(*relative_path(profile["portable_registry"]).parts)
    for path, expected in (
        (manifest_path, FOLD1_SHA256),
        (registry_path, profile["expected_portable_registry_sha256"]),
    ):
        if sha256_file(path) != sealed_hash(expected, path.name):
            raise ValueError(f"staged input identity mismatch: {path.name}")
    registry = read_object(registry_path, "portable source registry")
    staged = {item["assembled_path"]: item for item in profile["mounted_inputs"].values()}
    staged.update(profile["mounted_files"])
    for logical, item in staged.items():
        source = registry.get("sources", {}).get(logical, {})
        original = source.get("original_identity", {})
        if source.get("sha256") != item["sha256"] or original.get("bytes") != item["bytes"]:
            raise ValueError(f"fit packet disagrees with the portable registry for {logical}")
    mounted = {
        split: assemble(split, item, data_root, discovered)
        for split, item in profile["mounted_inputs"].items()
    }
    indexes = {
        name: copy_mounted(name, item, data_root, discovered)
        for name, item in profile["mounted_files"].items()
    }
    disk_after = shutil.disk_usage(work.parent).free
    if disk_after < reserve:
        raise OSError("post-staging free disk fell below the sealed reserve")
    staging: dict[str, Any] = {
        "mounted_inputs": mounted,
        "mounted_files": indexes,
        "mount_scans": mount_scans,
        "free_bytes_before_assembly": disk_before,
        "free_bytes_after_staging": disk_after,
    }
    return source_root, data_root, manifest_path, runtime, staging
