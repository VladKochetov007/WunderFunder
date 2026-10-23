"""Validate the fixed first-cell export bundle and its prediction arrays."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from first_cell_score_core import (
    ARM_MODES,
    EXPECTED_BINDING_BYTES,
    EXPECTED_BINDING_SHA256,
    EXPECTED_DATA_ADMISSION,
    EXPECTED_PARENT_RESOLUTIONS,
    EXPECTED_PORTABLE_MANIFEST,
    EXPECTED_PORTABLE_SOURCE_NAMES,
    EXPECTED_REGISTERED_SOURCES,
    EXPECTED_SOURCE_MANIFEST_BYTES,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    EXPECTED_SOURCE_SET_SHA256,
    EXPECTED_SPLIT_SHA256,
    EXPECTED_VALID_PARQUET,
    MAX_EXPORT_MANIFEST_BYTES,
    MAX_SMALL_JSON_BYTES,
    N_GROUPS,
    N_OUTPUTS,
    PREDICTION_DATA_BYTES,
    PREDICTION_SHAPE,
    SOURCE_COMMIT,
    SOURCE_PINS,
    _read_json,
    _safe_child,
)


def _validate_data_identity(binding: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Bind producer metadata to the six admitted sources and portable registry."""
    registered = binding.get("registered_data")
    if not isinstance(registered, dict) or set(registered) != {
        "admission",
        "portable_manifest",
        "source_identities",
        "valid_parquet",
    }:
        raise ValueError("export binding registered-data section is malformed")
    admission = registered.get("admission")
    valid = registered.get("valid_parquet")
    source_identities = registered.get("source_identities")
    portable = registered.get("portable_manifest")
    if (
        admission != EXPECTED_DATA_ADMISSION
        or valid != EXPECTED_VALID_PARQUET
        or source_identities != EXPECTED_REGISTERED_SOURCES
        or portable != EXPECTED_PORTABLE_MANIFEST
    ):
        raise ValueError("export binding registered-data identity differs from admission")
    if any(type(valid[key]) is not type(value) for key, value in EXPECTED_VALID_PARQUET.items()):
        raise ValueError("registered validation-data identity has unexpected field types")
    if any(
        type(admission[key]) is not type(value) for key, value in EXPECTED_DATA_ADMISSION.items()
    ):
        raise ValueError("registered-data admission pin has unexpected field types")

    if set(source_identities) != set(EXPECTED_PORTABLE_SOURCE_NAMES):
        raise ValueError("registered-data source membership differs from the portable registry")
    for relative, entry in source_identities.items():
        expected_entry = EXPECTED_REGISTERED_SOURCES[relative]
        if (
            set(entry)
            != {
                "bytes",
                "device",
                "inode",
                "islink",
                "mtime_ns",
                "path",
                "registered_sha256",
                "resolved",
            }
            or entry != expected_entry
            or any(type(entry[key]) is not type(value) for key, value in expected_entry.items())
        ):
            raise ValueError(
                f"registered source identity has unexpected fields or types: {relative}"
            )
        if entry["path"] != str(Path(binding["data_root"]) / relative):
            raise ValueError(f"registered source path differs from the fixed data root: {relative}")
    if source_identities["datasets/valid.parquet"] != valid:
        raise ValueError("duplicated valid.parquet pin differs from the six-source identity map")

    identity = manifest.get("data_identity")
    if not isinstance(identity, dict) or set(identity) != {
        "schema",
        "admission",
        "portable_manifest",
        "registered_source_identities",
        "registered_valid_parquet",
        "observed_sources_before_split",
        "observed_sources_after_split",
        "observed_before_sequence_open",
        "observed_after_export",
    }:
        raise ValueError("export manifest registered-data observations are malformed")
    if (
        identity.get("schema") != "fg19.registered-valid-data-identity.v1"
        or identity.get("admission") != admission
        or identity.get("registered_source_identities") != source_identities
        or identity.get("registered_valid_parquet") != valid
    ):
        raise ValueError("export manifest data identity differs from its fixed binding")
    expected_portable_observation = {
        **EXPECTED_PORTABLE_MANIFEST,
        "source_names": EXPECTED_PORTABLE_SOURCE_NAMES,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "version": 1,
    }
    manifest_portable = identity.get("portable_manifest")
    if manifest_portable != expected_portable_observation or any(
        type(manifest_portable[key]) is not type(value)
        for key, value in expected_portable_observation.items()
    ):
        raise ValueError("export manifest portable registry identity differs")

    source_observation_fields = {
        "path",
        "resolved",
        "bytes",
        "device",
        "inode",
        "islink",
        "mtime_ns",
    }
    for name in ("observed_sources_before_split", "observed_sources_after_split"):
        observations = identity[name]
        if not isinstance(observations, dict) or set(observations) != set(
            EXPECTED_PORTABLE_SOURCE_NAMES
        ):
            raise ValueError(f"export manifest {name} membership differs")
        for relative in EXPECTED_PORTABLE_SOURCE_NAMES:
            observation = observations[relative]
            registered_source = source_identities[relative]
            expected_source_observation = {
                key: registered_source[key] for key in source_observation_fields
            }
            if (
                not isinstance(observation, dict)
                or set(observation) != source_observation_fields
                or observation != expected_source_observation
                or any(
                    type(observation[key]) is not type(value)
                    for key, value in expected_source_observation.items()
                )
            ):
                raise ValueError(f"export manifest {name} differs from admission: {relative}")
    if identity["observed_sources_before_split"] != identity["observed_sources_after_split"]:
        raise ValueError("registered source identities changed during split validation")

    observation_keys = {
        "path",
        "resolved",
        "bytes",
        "device",
        "inode",
        "islink",
        "mtime_ns",
        "parent",
    }
    parent_keys = {"path", "resolved", "device", "inode"}
    for name in ("observed_before_sequence_open", "observed_after_export"):
        observation = identity[name]
        if not isinstance(observation, dict) or set(observation) != observation_keys:
            raise ValueError(f"export manifest {name} fields differ")
        expected_observation = {
            key: valid[key] for key in ("path", "resolved", "bytes", "device", "inode", "mtime_ns")
        }
        if (
            any(observation.get(key) != value for key, value in expected_observation.items())
            or observation.get("islink") is not False
        ):
            raise ValueError(f"export manifest {name} differs from admitted leaf stats")
        if any(
            type(observation[key]) is not type(value) for key, value in expected_observation.items()
        ):
            raise ValueError(f"export manifest {name} has unexpected leaf stat types")
        parent = observation.get("parent")
        if (
            not isinstance(parent, dict)
            or set(parent) != parent_keys
            or parent.get("path") != str(Path(valid["path"]).parent)
            or parent.get("resolved") != EXPECTED_PARENT_RESOLUTIONS["datasets"]
            or type(parent.get("device")) is not int
            or type(parent.get("inode")) is not int
            or parent["device"] < 0
            or parent["inode"] <= 0
        ):
            raise ValueError(f"export manifest {name} parent alias observation differs")
    if identity["observed_before_sequence_open"] != identity["observed_after_export"]:
        raise ValueError("registered data identity changed during feature-only export")


def _read_export_bundle(
    evidence_root: Path,
    export_root: Path,
    binding_path: Path,
    source_manifest_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[int, int, int, int],
]:
    """Read and verify the completed exporter manifest, binding, and source pins."""
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise ValueError("private evidence root must be a real directory")
    if evidence_root.resolve(strict=True) != evidence_root:
        raise ValueError("private evidence root cannot traverse a symlink")
    if (
        export_root != evidence_root / "exports"
        or binding_path != evidence_root / "EXPORT_BINDING_V4.json"
        or source_manifest_path != evidence_root / "EXPORT_SOURCE_MANIFEST.json"
    ):
        raise ValueError(
            "export metadata and predictions must use the fixed private evidence layout"
        )
    if export_root.is_symlink() or not export_root.is_dir():
        raise ValueError("export root must be a real directory")
    if export_root.resolve(strict=True) != export_root:
        raise ValueError("export root cannot traverse a symlink")
    if binding_path.is_symlink() or source_manifest_path.is_symlink():
        raise ValueError("binding and source-manifest files cannot be symlinks")
    binding, binding_pin, _ = _read_json(binding_path, MAX_SMALL_JSON_BYTES)
    if (binding_pin["bytes"], binding_pin["sha256"]) != (
        EXPECTED_BINDING_BYTES,
        EXPECTED_BINDING_SHA256,
    ):
        raise ValueError("first-cell export binding bytes differ from the reviewed contract")
    if binding.get("schema") != "fg19.first-cell-evaluation-export-binding.v1":
        raise ValueError("unsupported first-cell export binding schema")
    source_manifest, source_pin, _ = _read_json(source_manifest_path, MAX_EXPORT_MANIFEST_BYTES)
    if (source_pin["bytes"], source_pin["sha256"]) != (
        EXPECTED_SOURCE_MANIFEST_BYTES,
        EXPECTED_SOURCE_MANIFEST_SHA256,
    ):
        raise ValueError("source manifest bytes differ from the reviewed contract")
    for key, actual in (
        ("evidence_root", evidence_root),
        ("output_dir", export_root),
    ):
        expected = binding.get(key)
        if not isinstance(expected, str) or expected != str(actual):
            raise ValueError(f"export binding {key} differs from the explicit private root")
    for key in ("data_root", "source_root"):
        value = binding.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"export binding {key} is not an explicit absolute B path")
    source = binding.get("source")
    if not isinstance(source, dict):
        raise ValueError("export binding source section is malformed")
    source_manifest_files = source_manifest.get("files")
    if (
        source.get("commit") != SOURCE_COMMIT
        or source.get("manifest_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or source.get("source_set_sha256") != EXPECTED_SOURCE_SET_SHA256
        or source.get("manifest_relative_path") != "EXPORT_SOURCE_MANIFEST.json"
        or source.get("manifest_bytes") != EXPECTED_SOURCE_MANIFEST_BYTES
        or type(source.get("file_count")) is not int
        or source.get("file_count") != 167
        or source_manifest.get("source_commit") != SOURCE_COMMIT
        or source_manifest.get("schema") != "fg19.teacher-cache-source-manifest.v1"
        or source_manifest.get("source_root") != binding.get("source_root")
        or set(source_manifest) != {"files", "schema", "source_commit", "source_root"}
        or not isinstance(source_manifest_files, list)
        or len(source_manifest_files) != source.get("file_count")
        or source.get("source_root") != binding.get("source_root")
    ):
        raise ValueError("export binding and source manifest do not identify source68")
    source_tree_digest = hashlib.sha256()
    source_paths: set[str] = set()
    for item in source_manifest_files:
        if (
            not isinstance(item, dict)
            or set(item) != {"bytes", "path", "sha256"}
            or type(item.get("bytes")) is not int
            or item["bytes"] <= 0
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["sha256"])
            or item["path"] in source_paths
        ):
            raise ValueError("source manifest contains a malformed or duplicate source pin")
        source_paths.add(item["path"])
    for item in sorted(source_manifest_files, key=lambda row: row["path"]):
        source_tree_digest.update(
            item["path"].encode("utf-8") + b"\0" + item["sha256"].encode("ascii") + b"\n"
        )
    if source_tree_digest.hexdigest() != EXPECTED_SOURCE_SET_SHA256:
        raise ValueError("source manifest aggregate pin differs from source68")
    for relative, (byte_count, digest) in SOURCE_PINS.items():
        expected_pin = {"bytes": byte_count, "sha256": digest}
        if source_manifest.get("files") is None or not any(
            item.get("path") == relative and item == {"path": relative, **expected_pin}
            for item in source_manifest["files"]
            if isinstance(item, dict)
        ):
            raise ValueError(
                f"source manifest does not contain the exact canonical pin: {relative}"
            )
    source_file_pins = {
        item.get("path"): item for item in source_manifest["files"] if isinstance(item, dict)
    }
    api_files = source.get("api_files")
    if not isinstance(api_files, dict):
        raise ValueError("export binding canonical API pins differ from the source manifest")
    for relative, item in api_files.items():
        source_pin = source_file_pins.get(relative)
        if (
            not isinstance(relative, str)
            or not isinstance(source_pin, dict)
            or not isinstance(item, dict)
            or set(item) != {"bytes", "sha256"}
            or type(item.get("bytes")) is not int
            or item["bytes"] <= 0
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["sha256"])
            or item["bytes"] != source_pin.get("bytes")
            or item["sha256"] != source_pin.get("sha256")
        ):
            raise ValueError("export binding canonical API pins differ from the source manifest")
    split = binding.get("split")
    expected_split = {
        "absolute_path": str(Path(binding["project_base"]) / "splits" / "fold0.json"),
        "chunk_rows": 2000,
        "dtype": "<f4",
        "evaluation_group_count": N_GROUPS,
        "feature_count": 112,
        "fit_valid_group_count": 1405,
        "fold": 0,
        "n_folds": 4,
        "output_count": N_OUTPUTS,
        "sequence_rows": 20_000,
        "sha256": EXPECTED_SPLIT_SHA256,
        "valid_row_group_count": 1873,
    }
    if not isinstance(split, dict) or split != expected_split:
        raise ValueError("export binding split contract differs from the fixed c09f panel")
    outputs = binding.get("outputs")
    if outputs != {
        "completion_marker": "EXPORT_MANIFEST_V1.json",
        "data_bytes_each": PREDICTION_DATA_BYTES,
        "dtype": "float32-little-endian",
        "files_per_arm": ["predictions.npy", "PREDICTION_SIDECAR.json"],
        "score_or_metric": False,
        "scored_mask_read": False,
        "shape": list(PREDICTION_SHAPE),
        "targets_read": False,
    }:
        raise ValueError("export binding output contract differs from the fixed feature-only panel")
    arms = binding.get("arms")
    if not isinstance(arms, list) or [arm.get("arm") for arm in arms] != list(ARM_MODES):
        raise ValueError("export binding does not list the five fixed arms in canonical order")
    checkpoint = binding.get("expected_checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("epoch") != 5
        or checkpoint.get("updates") != 44760
    ):
        raise ValueError("export binding does not fix the selected epoch-5/44,760-update endpoint")
    for arm in arms:
        name = arm["arm"]
        model = arm.get("model")
        if (
            arm.get("representation_mode") != ARM_MODES[name]
            or arm.get("selected_epoch", 5) != 5
            or not isinstance(model, dict)
            or model.get("bytes") != 4_279_970
            or not isinstance(model.get("sha256"), str)
            or len(model["sha256"]) != 64
        ):
            raise ValueError(f"{name} binding identity differs from the fixed epoch-5 panel")
    manifest, manifest_pin, manifest_identity = _read_json(
        _safe_child(export_root, "EXPORT_MANIFEST_V1.json"), MAX_EXPORT_MANIFEST_BYTES
    )
    if (
        set(manifest)
        != {
            "schema",
            "status",
            "created_utc",
            "binding_sha256",
            "source",
            "split",
            "data_identity",
            "ordered_groups",
            "membership_sha256",
            "arms",
            "quality_or_metric_claim",
        }
        or manifest.get("schema") != "fg19.first-cell-evaluation-export-manifest.v1"
        or manifest.get("status") != "COMPLETE_PREDICTION_EXPORTS_NOT_SCORED"
        or manifest.get("binding_sha256") != binding_pin["sha256"]
        or manifest.get("source") != source
        or manifest.get("quality_or_metric_claim") is not False
        or manifest.get("split")
        != binding.get("split") | {"role": "fit-excluded-exposed-development-evaluation"}
    ):
        raise ValueError("export completion manifest differs from the fixed binding")
    _validate_data_identity(binding, manifest)
    return binding, source_manifest, manifest, manifest_pin, manifest_identity


def _validate_bound_paths(args: argparse.Namespace, binding: dict[str, Any]) -> None:
    """Require every runtime root to match the exact B layout before source imports."""
    paths = {
        "project_base": args.evidence_root.parent.parent,
        "data_root": args.data_root,
        "source_root": args.source_root,
        "evidence_root": args.evidence_root,
    }
    for key, path in paths.items():
        if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
            raise ValueError(f"{key} must be an explicit real directory without symlink traversal")
        if binding.get(key) != str(path):
            raise ValueError(f"{key} differs from the exact export binding")
    base = paths["project_base"]
    if (
        paths["data_root"] != base / "data"
        or paths["source_root"] != base / "source-fg19-cache-68ff6829"
        or paths["evidence_root"]
        != base / "research_tmp" / "fg19_first_cell_f0_s1001_evaluation_v3"
        or args.export_root != paths["evidence_root"] / "exports"
        or args.binding != paths["evidence_root"] / "EXPORT_BINDING_V4.json"
        or args.source_manifest != paths["evidence_root"] / "EXPORT_SOURCE_MANIFEST.json"
    ):
        raise ValueError("runtime paths do not match the fixed project/data/source/evidence layout")
    split_path = Path(str(binding.get("split", {}).get("absolute_path", "")))
    if split_path != base / "splits" / "fold0.json" or split_path.is_symlink():
        raise ValueError("campaign split path differs from the fixed current F0 manifest")
    if args.out.is_symlink() or not args.out.is_absolute() or ".." in args.out.parts:
        raise ValueError("score report must be a direct absolute create-only evidence file")
    if args.out.parent != paths["evidence_root"] or args.out.suffix != ".json":
        raise ValueError("score report must be a JSON child of the private evidence root")
    if args.out.exists():
        raise FileExistsError(f"score report already exists: {args.out}")
