"""Create-only, feature-only prediction exporter for the fixed F0 first cell."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import operator
import os
import shutil
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import ModuleType

EXPORT_SOURCE_MANIFEST_NAME = "EXPORT_SOURCE_MANIFEST.json"
BINDING_BYTES = 15_798
BINDING_SHA256 = "6cd8c46f31214623d97e54c32bac9395608bfc918b9b44e879e44ad2548f0eb9"
SOURCE_MANIFEST_BYTES = 27_569
SOURCE_MANIFEST_SHA256 = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SOURCE_TREE_SHA256 = "9c461bcc5883572a071d41a86e18b9a91136099cf8fa87371ce7c3ee41445bd8"
PORTABLE_MANIFEST_BYTES = 2_157
PORTABLE_MANIFEST_SHA256 = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
PORTABLE_MANIFEST_PATH = (
    "/home/grinchenkoeg/computer-vision-homework-20261018/"
    "support-source/terminal-v1/PORTABLE_SOURCES.json"
)
PORTABLE_SOURCE_NAMES = (
    "cache/chain_folds.npz",
    "cache/chains.npz",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
    "datasets/train.parquet",
    "datasets/valid.parquet",
)
MAX_METADATA_BYTES = 16_000_000


def _digest_file(path: Path, max_bytes: int) -> tuple[int, str]:
    """Hash one regular file through a stable no-follow descriptor."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError(f"file is not regular or exceeds its byte cap: {path}")
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"file changed while hashing: {path}")
        if (linked.st_dev, linked.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError(f"file path changed while hashing: {path}")
        if total != before.st_size:
            raise ValueError(f"short file read: {path}")
        return total, digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_pinned(path: Path, byte_count: int, sha256: str, max_bytes: int) -> bytes:
    """Read a bounded metadata file once and verify its exact bytes and identity."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size != byte_count
            or byte_count > max_bytes
        ):
            raise ValueError(f"pinned metadata size differs: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"metadata changed while reading: {path}")
        if (linked.st_dev, linked.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError(f"metadata path changed while reading: {path}")
        if total != byte_count or digest.hexdigest() != sha256:
            raise ValueError(f"metadata bytes or digest differ: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _safe_child(root: Path, relative: str) -> Path:
    """Resolve a relative evidence path without accepting traversal or symlink parents."""
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in (".", "..") for part in pure.parts):
        raise ValueError(f"unsafe relative path: {relative}")
    candidate = root
    for part in pure.parts[:-1]:
        candidate = candidate / part
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(f"unsafe evidence directory component: {candidate}")
    return candidate / pure.parts[-1]


def _load_binding(directory: Path) -> dict[str, object]:
    """Load the exact local export binding and bundled source manifest."""
    raw = _read_pinned(
        directory / "EXPORT_BINDING_V1.json", BINDING_BYTES, BINDING_SHA256, MAX_METADATA_BYTES
    )
    binding = json.loads(raw)
    if (
        not isinstance(binding, dict)
        or binding.get("schema") != "fg19.first-cell-evaluation-export-binding.v1"
    ):
        raise ValueError("unsupported first-cell export binding")
    expected_arms = [
        ("control", "base"),
        ("ema", "ema"),
        ("sparse", "sparse"),
        ("nll", "base"),
        ("allthree", "ema_sparse"),
    ]
    arms = binding.get("arms")
    if (
        not isinstance(arms, list)
        or [
            (arm.get("arm"), arm.get("representation_mode"))
            for arm in arms
            if isinstance(arm, dict)
        ]
        != expected_arms
        or len(arms) != len(expected_arms)
    ):
        raise ValueError("binding arm and representation panel differs")
    if binding.get("data_root") != str(Path(binding.get("project_base", "")) / "data"):
        raise ValueError("binding data root does not match its project base")
    source = binding.get("source")
    split = binding.get("split")
    if not isinstance(source, dict) or not isinstance(split, dict):
        raise ValueError("binding source or split section is malformed")
    if binding.get("source_root") != source.get("source_root"):
        raise ValueError("binding source-root fields disagree")
    registered_data = binding.get("registered_data")
    if not isinstance(registered_data, dict) or set(registered_data) != {
        "admission",
        "portable_manifest",
        "source_identities",
        "valid_parquet",
    }:
        raise ValueError("binding registered-data identity is malformed")
    registered_valid = registered_data.get("valid_parquet")
    if (
        not isinstance(registered_valid, dict)
        or registered_valid.get("path")
        != str(Path(str(binding["data_root"])) / "datasets" / "valid.parquet")
        or registered_valid.get("resolved")
        != "/home/grinchenkoeg/WunderFunder/datasets/valid.parquet"
    ):
        raise ValueError("binding registered validation-data path differs")
    admission_pin = registered_data.get("admission")
    if (
        not isinstance(admission_pin, dict)
        or admission_pin.get("root_key") != "fit_receipts_root"
        or admission_pin.get("relative_path") != "ADMISSION.json"
        or admission_pin.get("absolute_path")
        != str(Path(str(binding["fit_receipts_root"])) / "ADMISSION.json")
    ):
        raise ValueError("binding registered-data admission path differs")
    registry_pin = registered_data.get("portable_manifest")
    if (
        not isinstance(registry_pin, dict)
        or set(registry_pin) != {"absolute_path", "bytes", "sha256"}
        or registry_pin.get("absolute_path") != PORTABLE_MANIFEST_PATH
        or registry_pin.get("bytes") != PORTABLE_MANIFEST_BYTES
        or registry_pin.get("sha256") != PORTABLE_MANIFEST_SHA256
    ):
        raise ValueError("binding portable source registry pin differs")
    source_identities = registered_data.get("source_identities")
    if not isinstance(source_identities, dict) or set(source_identities) != set(
        PORTABLE_SOURCE_NAMES
    ):
        raise ValueError("binding portable source identity set differs")
    if registered_valid != source_identities.get("datasets/valid.parquet"):
        raise ValueError("binding valid parquet identity differs from registered sources")
    expected_shape = [
        split.get("evaluation_group_count"),
        split.get("sequence_rows"),
        split.get("output_count"),
    ]
    outputs = binding.get("outputs")
    if (
        not isinstance(outputs, dict)
        or outputs.get("shape") != expected_shape
        or outputs.get("dtype") != "float32-little-endian"
        or outputs.get("data_bytes_each")
        != expected_shape[0] * expected_shape[1] * expected_shape[2] * 4
        or outputs.get("targets_read") is not False
        or outputs.get("scored_mask_read") is not False
        or outputs.get("score_or_metric") is not False
    ):
        raise ValueError("binding output contract is inconsistent")
    manifest_bytes = _read_pinned(
        directory / EXPORT_SOURCE_MANIFEST_NAME,
        SOURCE_MANIFEST_BYTES,
        SOURCE_MANIFEST_SHA256,
        MAX_METADATA_BYTES,
    )
    manifest = json.loads(manifest_bytes)
    if source.get("manifest_relative_path") != EXPORT_SOURCE_MANIFEST_NAME:
        raise ValueError("binding source-manifest path differs")
    if (
        manifest.get("schema") != "fg19.teacher-cache-source-manifest.v1"
        or manifest.get("source_commit") != source.get("commit")
        or manifest.get("source_root") != source.get("source_root")
        or source.get("source_root") != binding.get("source_root")
        or len(manifest.get("files", [])) != source.get("file_count")
        or source.get("manifest_bytes") != len(manifest_bytes)
        or source.get("manifest_sha256") != SOURCE_MANIFEST_SHA256
        or source.get("source_set_sha256") != SOURCE_TREE_SHA256
    ):
        raise ValueError("bundled source manifest identity differs")
    return binding


def _verify_source_tree(
    source_root: Path, binding: dict[str, object], directory: Path
) -> dict[str, dict[str, object]]:
    """Verify all 167 source files before importing any project module."""
    source = binding["source"]
    manifest_bytes = _read_pinned(
        directory / EXPORT_SOURCE_MANIFEST_NAME,
        SOURCE_MANIFEST_BYTES,
        SOURCE_MANIFEST_SHA256,
        MAX_METADATA_BYTES,
    )
    manifest = json.loads(manifest_bytes)
    if str(source_root.resolve(strict=True)) != source["source_root"]:
        raise ValueError("source root differs from the accepted source manifest")
    files = manifest["files"]
    if not isinstance(files, list) or len(files) != 167:
        raise ValueError("source manifest must contain exactly 167 files")
    pins: dict[str, dict[str, object]] = {}
    combined = hashlib.sha256()
    for item in sorted(files, key=lambda entry: entry["path"]):
        relative = item.get("path")
        byte_count = item.get("bytes")
        expected = item.get("sha256")
        if (
            not isinstance(relative, str)
            or type(byte_count) is not int
            or not isinstance(expected, str)
        ):
            raise ValueError("source manifest entry is malformed")
        if relative in pins:
            raise ValueError(f"duplicate source manifest entry: {relative}")
        path = _safe_child(source_root, relative)
        size, digest = _digest_file(path, max_bytes=50_000_000)
        if (size, digest) != (byte_count, expected):
            raise ValueError(f"source file differs from manifest: {relative}")
        pins[relative] = item
        combined.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    if combined.hexdigest() != SOURCE_TREE_SHA256:
        raise ValueError("source tree aggregate digest differs")
    api_files = source.get("api_files")
    if not isinstance(api_files, dict):
        raise ValueError("canonical API pin map is malformed")
    for relative, expected in api_files.items():
        if pins.get(relative) != expected:
            raise ValueError(f"canonical API pin differs from the full source manifest: {relative}")
    return pins


def _reject_bytecode(root: Path) -> None:
    """Reject stale bytecode so imports use only source bytes verified here."""
    if any(root.rglob("*.pyc")):
        raise RuntimeError(f"Python bytecode is present under pinned source path: {root}")


def _validate_portable_manifest_payload(
    payload: object, source_identities: dict[str, object], split_sha256: str
) -> None:
    """Join the exact portable registry to all six admitted source identities."""
    if not isinstance(payload, dict) or set(payload) != {
        "created_utc",
        "sources",
        "split_manifest_sha256",
        "version",
    }:
        raise ValueError("portable source registry schema differs")
    if (
        type(payload.get("version")) is not int
        or payload.get("version") != 1
        or payload.get("split_manifest_sha256") != split_sha256
    ):
        raise ValueError("portable source registry split identity differs")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(PORTABLE_SOURCE_NAMES):
        raise ValueError("portable source registry membership differs")
    if set(source_identities) != set(PORTABLE_SOURCE_NAMES):
        raise ValueError("admitted portable-source identity membership differs")
    for name in PORTABLE_SOURCE_NAMES:
        admitted = source_identities[name]
        entry = sources[name]
        if (
            not isinstance(admitted, dict)
            or not isinstance(entry, dict)
            or set(entry) != {"original_identity", "sha256"}
        ):
            raise ValueError(f"portable source entry is malformed: {name}")
        original_identity = entry["original_identity"]
        digest = entry["sha256"]
        if (
            not isinstance(original_identity, dict)
            or type(admitted.get("bytes")) is not int
            or type(admitted.get("mtime_ns")) is not int
            or original_identity.get("bytes") != admitted["bytes"]
            or original_identity.get("mtime_ns") != admitted["mtime_ns"]
            or type(digest) is not str
            or digest != admitted.get("registered_sha256")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"portable source identity differs from admission: {name}")
        if "sha256" in original_identity and original_identity["sha256"] != digest:
            raise ValueError(f"portable original digest differs from admission: {name}")


def _verify_portable_manifest(binding: dict[str, object]) -> dict[str, object]:
    """Read and pin the small approved portability registry without source payloads."""
    registered = binding["registered_data"]
    pin = registered["portable_manifest"]
    path = Path(str(pin["absolute_path"]))
    if path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError("portable source registry path traverses a symbolic link")
    raw = _read_pinned(path, PORTABLE_MANIFEST_BYTES, PORTABLE_MANIFEST_SHA256, MAX_METADATA_BYTES)
    payload = json.loads(raw)
    _validate_portable_manifest_payload(
        payload, registered["source_identities"], binding["split"]["sha256"]
    )
    return {
        **pin,
        "source_names": list(PORTABLE_SOURCE_NAMES),
        "split_manifest_sha256": binding["split"]["sha256"],
        "version": 1,
    }


def _configure_portable_manifest(binding: dict[str, object]) -> None:
    """Set only the approved registry path and reject conflicting inherited values."""
    expected = str(binding["registered_data"]["portable_manifest"]["absolute_path"])
    current = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    if current not in (None, "", expected):
        raise RuntimeError(
            "inherited WNN_PORTABLE_DATA_MANIFEST differs from the approved registry"
        )
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = expected
    if os.environ.get("WNN_PORTABLE_DATA_MANIFEST") != expected:
        raise RuntimeError("approved WNN portable registry environment was not set")


def _read_json_pin(pin: dict[str, object], binding: dict[str, object]) -> dict[str, object]:
    """Read a receipt from its exact fixed B root and relative path."""
    root_key = pin.get("root_key")
    if root_key not in ("fit_receipts_root", "evidence_root"):
        raise ValueError("receipt pin names an unsupported fixed root")
    root = Path(str(binding[root_key]))
    relative_path = pin.get("relative_path")
    absolute_path = pin.get("absolute_path")
    if not isinstance(relative_path, str) or not isinstance(absolute_path, str):
        raise ValueError("receipt path pin is malformed")
    path = _safe_child(root, relative_path)
    if str(path) != absolute_path or root.resolve(strict=True) != root:
        raise ValueError("receipt path is not the exact pinned absolute path")
    if path.resolve(strict=True) != path:
        raise ValueError("receipt path traverses a symbolic link")
    raw = _read_pinned(path, int(pin["bytes"]), str(pin["sha256"]), MAX_METADATA_BYTES)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not a JSON object: {path}")
    return value


def _verify_registered_data_receipt(binding: dict[str, object]) -> dict[str, object]:
    """Bind all six portability sources to the fixed AllThree admission."""
    registered = binding["registered_data"]
    admission_pin = registered["admission"]
    admission = _read_json_pin(admission_pin, binding)
    allthree = next(arm for arm in binding["arms"] if arm["arm"] == "allthree")
    if (
        admission.get("schema") != "fg19-child-allthree-fit-admission.v1"
        or admission.get("run_name") != allthree["run_name"]
        or admission.get("unit") != allthree["unit"]
        or admission.get("source_commit") != binding["source"]["commit"]
        or admission.get("split_sha256") != binding["split"]["sha256"]
    ):
        raise ValueError("registered-data admission identity differs from the AllThree fit")
    admitted_stats = admission.get("data_identity_stats_only")
    source_identities = registered.get("source_identities")
    if (
        not isinstance(admitted_stats, dict)
        or set(admitted_stats) != set(PORTABLE_SOURCE_NAMES)
        or not isinstance(source_identities, dict)
        or admitted_stats != source_identities
    ):
        raise ValueError("AllThree admission registered-data stats are malformed")
    expected_identity = source_identities["datasets/valid.parquet"]
    if registered["valid_parquet"] != expected_identity:
        raise ValueError("registered validation-data identity differs from its source map")
    expected_types = {
        "bytes": int,
        "device": int,
        "inode": int,
        "islink": bool,
        "mtime_ns": int,
        "path": str,
        "registered_sha256": str,
        "resolved": str,
    }
    for name in PORTABLE_SOURCE_NAMES:
        identity = source_identities[name]
        expected_path = str(Path(str(binding["data_root"])) / name)
        if (
            not isinstance(identity, dict)
            or set(identity) != set(expected_types)
            or any(type(identity[key]) is not expected_types[key] for key in expected_types)
            or identity["path"] != expected_path
            or len(identity["registered_sha256"]) != 64
            or any(
                character not in "0123456789abcdef" for character in identity["registered_sha256"]
            )
            or identity["bytes"] <= 0
            or identity["device"] < 0
            or identity["inode"] <= 0
            or identity["mtime_ns"] < 0
            or not Path(identity["resolved"]).is_absolute()
        ):
            raise ValueError(f"registered portable source stat fields differ: {name}")
    if expected_identity["islink"] is not False:
        raise ValueError("registered validation-data leaf must not be a symlink")
    return expected_identity


def _observe_registered_source_identities(
    data_root: Path, source_identities: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Stat the six admitted portable sources without reading their contents."""
    if not isinstance(source_identities, dict) or set(source_identities) != set(
        PORTABLE_SOURCE_NAMES
    ):
        raise ValueError("registered portable source identity membership differs")
    observations: dict[str, dict[str, object]] = {}
    for name in PORTABLE_SOURCE_NAMES:
        expected = source_identities[name]
        if not isinstance(expected, dict):
            raise ValueError(f"registered portable source identity is malformed: {name}")
        path = Path(str(expected["path"]))
        if path != data_root / name:
            raise ValueError(f"registered portable source path differs: {name}")
        link_before = path.lstat()
        stat_before = path.stat()
        resolved_before = path.resolve(strict=True)
        if not stat.S_ISREG(stat_before.st_mode):
            raise ValueError(f"registered portable source is not a regular file: {name}")
        link_after = path.lstat()
        stat_after = path.stat()
        resolved_after = path.resolve(strict=True)
        link_signature_before = (
            link_before.st_dev,
            link_before.st_ino,
            link_before.st_mode,
            link_before.st_size,
            link_before.st_mtime_ns,
        )
        link_signature_after = (
            link_after.st_dev,
            link_after.st_ino,
            link_after.st_mode,
            link_after.st_size,
            link_after.st_mtime_ns,
        )
        stat_signature_before = (
            stat_before.st_dev,
            stat_before.st_ino,
            stat_before.st_size,
            stat_before.st_mtime_ns,
        )
        stat_signature_after = (
            stat_after.st_dev,
            stat_after.st_ino,
            stat_after.st_size,
            stat_after.st_mtime_ns,
        )
        observed = {
            "path": str(path),
            "resolved": str(resolved_after),
            "bytes": stat_after.st_size,
            "device": stat_after.st_dev,
            "inode": stat_after.st_ino,
            "islink": stat.S_ISLNK(link_after.st_mode),
            "mtime_ns": stat_after.st_mtime_ns,
        }
        if (
            link_signature_before != link_signature_after
            or stat_signature_before != stat_signature_after
            or resolved_before != resolved_after
            or any(observed[field] != expected[field] for field in observed)
        ):
            raise ValueError(f"registered portable source stat identity changed: {name}")
        observations[name] = observed
    return observations


def _observe_registered_data_identity(
    data_root: Path, expected_identity: dict[str, object]
) -> dict[str, object]:
    """Stat only the exact registered validation-data leaf and its resolved parent."""
    path = Path(str(expected_identity["path"]))
    expected_resolved = Path(str(expected_identity["resolved"]))
    if path != data_root / "datasets" / "valid.parquet":
        raise ValueError("registered validation-data path differs from the fixed data root")
    linked_before = path.lstat()
    if stat.S_ISLNK(linked_before.st_mode) or not stat.S_ISREG(linked_before.st_mode):
        raise ValueError("registered validation-data leaf is not a regular non-symlink file")
    resolved = path.resolve(strict=True)
    parent = path.parent
    resolved_parent = parent.resolve(strict=True)
    if resolved != expected_resolved or resolved_parent != expected_resolved.parent:
        raise ValueError("validation-data path no longer resolves to its registered target")
    parent_before = parent.stat()
    if not stat.S_ISDIR(parent_before.st_mode):
        raise ValueError("validation-data parent is not a directory")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        linked_after = path.lstat()
        current_path_stat = path.stat()
        resolved_after = path.resolve(strict=True)
        resolved_parent_after = parent.resolve(strict=True)
        parent_after = parent.stat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(linked_after.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked_after.st_dev, linked_after.st_ino)
            or (opened.st_dev, opened.st_ino)
            != (current_path_stat.st_dev, current_path_stat.st_ino)
            or (opened.st_dev, opened.st_ino) != (linked_before.st_dev, linked_before.st_ino)
            or resolved_after != resolved
            or resolved_parent_after != resolved_parent
            or (parent_after.st_dev, parent_after.st_ino)
            != (parent_before.st_dev, parent_before.st_ino)
        ):
            raise ValueError("validation-data file identity changed during metadata check")
        observed_fields = {
            "bytes": opened.st_size,
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "islink": False,
            "mtime_ns": opened.st_mtime_ns,
        }
        if any(observed_fields[key] != expected_identity[key] for key in observed_fields):
            raise ValueError("validation-data file stat differs from its registered identity")
    finally:
        os.close(descriptor)
    parent_stat = parent_after
    return {
        "path": str(path),
        "resolved": str(resolved),
        **observed_fields,
        "parent": {
            "path": str(parent),
            "resolved": str(resolved_parent),
            "device": parent_stat.st_dev,
            "inode": parent_stat.st_ino,
        },
    }


def _check_acceptance_representation_mode(
    input_representation: dict[str, object], arm: dict[str, object]
) -> None:
    """Match each accepted checkpoint's representation metadata to its fixed arm."""
    if arm["arm"] == "control":
        if (
            arm["representation_mode"] != "base"
            or "mode" not in input_representation
            or input_representation["mode"] is not None
        ):
            raise ValueError("control acceptance legacy representation marker differs")
    elif input_representation.get("mode") != arm["representation_mode"]:
        raise ValueError(f"{arm['arm']} accepted representation mode differs")


def _verify_receipts_and_models(data_root: Path, binding: dict[str, object]) -> None:
    """Check every accepted epoch-5 receipt, exact stop, and model byte pin."""
    source = binding["source"]
    representation = binding["representation"]
    split = binding["split"]
    for arm in binding["arms"]:
        acceptance = _read_json_pin(arm["terminal_acceptance"], binding)
        expected = {
            "run_name": arm["run_name"],
            "unit": arm["unit"],
            "invocation_id": arm["invocation_id"],
            "status": "ACCEPTED",
            "selected_epoch": 5,
            "successful_updates": 44760,
            "source_commit": source["commit"],
            "source_file_count": source["file_count"],
            "source_manifest_sha256": source["manifest_sha256"],
            "strict_fp32_finite_model_optimizer_verified": True,
            "selected_model_epoch5_and_training_state_tensor_identity_verified": True,
            "normalization_identity": representation["normalization"],
        }
        for key, value in expected.items():
            if acceptance.get(key) != value:
                raise ValueError(f"{arm['arm']} acceptance field differs: {key}")
        if (
            acceptance.get("split_sha256") is not None
            and acceptance.get("split_sha256") != split["sha256"]
        ):
            raise ValueError(f"{arm['arm']} acceptance split identity differs")
        if (
            acceptance.get("exec_main_status") != 0
            or acceptance.get("main_pid") != 0
            or acceptance.get("exec_main_code") != 1
        ):
            raise ValueError(f"{arm['arm']} accepted native exit status differs")
        input_representation = acceptance.get("input_representation")
        if not isinstance(input_representation, dict) or (
            input_representation.get("artifact_sha256"),
            input_representation.get("state_sha256"),
        ) != (representation["artifact_sha256"], representation["state_sha256"]):
            raise ValueError(f"{arm['arm']} accepted representation identity differs")
        if input_representation.get("artifact_binding") != "shared_fit_cache":
            raise ValueError(f"{arm['arm']} representation cache binding differs")
        _check_acceptance_representation_mode(input_representation, arm)
        model_pin = acceptance.get("files", {}).get("model_ep5.pt")
        if model_pin != {"bytes": arm["model"]["bytes"], "sha256": arm["model"]["sha256"]}:
            raise ValueError(f"{arm['arm']} selected model receipt pin differs")
        stop = _read_json_pin(arm["exact_stop"], binding)
        if (stop.get("acceptance_sha256"), stop.get("unit"), stop.get("invocation_id")) != (
            arm["terminal_acceptance"]["sha256"],
            arm["unit"],
            arm["invocation_id"],
        ):
            raise ValueError(f"{arm['arm']} exact-stop receipt identity differs")
        before, after = stop.get("before"), stop.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError(f"{arm['arm']} exact-stop states are malformed")
        if (
            before.get("ActiveState"),
            before.get("SubState"),
            before.get("InvocationID"),
            before.get("ExecMainCode"),
            before.get("ExecMainStatus"),
            before.get("Result"),
            before.get("MainPID"),
        ) != ("active", "exited", arm["invocation_id"], "1", "0", "success", "0"):
            raise ValueError(f"{arm['arm']} native fit was not captured exited")
        if (
            after.get("ActiveState"),
            after.get("SubState"),
            after.get("ExecMainStatus"),
            after.get("ExecMainCode"),
            after.get("Result"),
            after.get("MainPID"),
            after.get("ControlGroup"),
        ) != (
            "inactive",
            "dead",
            "0",
            "0",
            "success",
            "0",
            "",
        ):
            raise ValueError(f"{arm['arm']} native fit exact stop is incomplete")
        if stop.get("stop", {}).get("returncode") != 0:
            raise ValueError(f"{arm['arm']} exact-stop command failed")
        verifier = stop.get("verifier")
        if (
            not isinstance(verifier, dict)
            or not isinstance(verifier.get("before"), dict)
            or not isinstance(verifier.get("after"), dict)
        ):
            raise ValueError(f"{arm['arm']} verifier stop receipt is malformed")
        if (
            verifier.get("unit"),
            verifier.get("invocation_id"),
        ) != (
            acceptance.get("verification_unit"),
            acceptance.get("verification_invocation_id"),
        ):
            raise ValueError(f"{arm['arm']} verifier stop identity differs from acceptance")
        verifier_before, verifier_after = verifier["before"], verifier["after"]
        if (
            verifier_before.get("ActiveState"),
            verifier_before.get("SubState"),
            verifier_before.get("InvocationID"),
            verifier_before.get("ExecMainCode"),
            verifier_before.get("ExecMainStatus"),
            verifier_before.get("Result"),
            verifier_before.get("MainPID"),
        ) != (
            "active",
            "exited",
            acceptance.get("verification_invocation_id"),
            "1",
            "0",
            "success",
            "0",
        ) or (
            verifier_after.get("ActiveState"),
            verifier_after.get("SubState"),
            verifier_after.get("ExecMainStatus"),
            verifier_after.get("ExecMainCode"),
            verifier_after.get("Result"),
            verifier_after.get("MainPID"),
            verifier_after.get("ControlGroup"),
        ) != ("inactive", "dead", "0", "0", "success", "0", ""):
            raise ValueError(f"{arm['arm']} integrity verifier was not exactly stopped")
        model_path = _safe_child(data_root, str(arm["model"]["relative_path"]))
        model_size, model_sha = _digest_file(
            model_path, int(binding["expected_checkpoint"]["model_byte_cap"])
        )
        if (model_size, model_sha) != (arm["model"]["bytes"], arm["model"]["sha256"]):
            raise ValueError(f"{arm['arm']} selected model bytes differ")
    split_path = Path(str(split["absolute_path"]))
    project_base = Path(str(binding["project_base"]))
    if split_path != project_base / "splits" / "fold0.json":
        raise ValueError("split path is outside the fixed B project base")
    if split_path.resolve(strict=True) != split_path:
        raise ValueError("split path traverses a symbolic link")
    size, digest = _digest_file(split_path, 2_000_000)
    if digest != split["sha256"] or size <= 0:
        raise ValueError("campaign split bytes differ")


def _import_source(
    source_root: Path, support_root: Path, pins: dict[str, dict[str, object]]
) -> dict[str, ModuleType]:
    """Import only pinned loader and canonical WNN source modules after source closure."""
    forbidden = ("wnn", "parent_fit_terminal_check_v3", "propensity_terminal_support")
    if any(
        name in sys.modules or any(key.startswith(name + ".") for key in sys.modules)
        for name in forbidden
    ):
        raise RuntimeError("a private loader or WNN module was already imported")
    if sys.dont_write_bytecode is not True:
        sys.dont_write_bytecode = True
    _reject_bytecode(source_root)
    _reject_bytecode(support_root)
    support_pins = (
        (
            "parent_fit_terminal_check_v3.py",
            64_550,
            "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
        ),
        (
            "propensity_terminal_support.py",
            13_107,
            "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
        ),
    )
    for filename, byte_count, digest in support_pins:
        if _digest_file(support_root / filename, byte_count) != (byte_count, digest):
            raise ValueError(f"restricted loader support differs: {filename}")
    sys.path[:0] = [str(support_root), str(source_root / "src")]
    modules = {
        "numpy": importlib.import_module("numpy"),
        "torch": importlib.import_module("torch"),
        "loader": importlib.import_module("parent_fit_terminal_check_v3"),
        "data": importlib.import_module("wnn.data"),
        "splits": importlib.import_module("wnn.splits"),
        "environment": importlib.import_module("wnn.training.environment"),
        "evaluation": importlib.import_module("wnn.training.evaluation"),
        "models": importlib.import_module("wnn.training.models"),
        "representation": importlib.import_module("wnn.training.input_representation"),
        "resources": importlib.import_module("wnn.resources"),
    }
    if Path(modules["loader"].__file__).resolve(strict=True) != (
        support_root / "parent_fit_terminal_check_v3.py"
    ).resolve(strict=True):
        raise ValueError("restricted loader import origin differs")
    support_module = sys.modules.get("propensity_terminal_support")
    for name, module, filename, byte_count, digest in (
        (
            "restricted loader",
            modules["loader"],
            "parent_fit_terminal_check_v3.py",
            64_550,
            "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
        ),
        (
            "restricted loader support",
            support_module,
            "propensity_terminal_support.py",
            13_107,
            "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
        ),
    ):
        origin = getattr(module, "__file__", None)
        expected_path = (support_root / filename).resolve(strict=True)
        if origin is None or Path(origin).resolve(strict=True) != expected_path:
            raise ValueError(f"{name} import origin differs")
        if _digest_file(Path(origin), byte_count) != (byte_count, digest):
            raise ValueError(f"{name} imported bytes differ")
    expected_origins = {
        "data": "src/wnn/data.py",
        "splits": "src/wnn/splits.py",
        "environment": "src/wnn/training/environment.py",
        "evaluation": "src/wnn/training/evaluation.py",
        "models": "src/wnn/training/models.py",
        "representation": "src/wnn/training/input_representation.py",
        "resources": "src/wnn/resources.py",
    }
    for key, relative in expected_origins.items():
        if Path(modules[key].__file__).resolve(strict=True) != (source_root / relative).resolve(
            strict=True
        ):
            raise ValueError(f"canonical API module origin differs: {relative}")
    for name, module in tuple(sys.modules.items()):
        if name == "wnn" or name.startswith("wnn."):
            origin = getattr(module, "__file__", None)
            if origin is None:
                raise ValueError(f"WNN module has no source origin: {name}")
            relative = (
                Path(origin)
                .resolve(strict=True)
                .relative_to(source_root.resolve(strict=True))
                .as_posix()
            )
            if relative not in pins:
                raise ValueError(f"WNN module came from outside the pinned source tree: {name}")
            if _digest_file(Path(origin), int(pins[relative]["bytes"])) != (
                int(pins[relative]["bytes"]),
                str(pins[relative]["sha256"]),
            ):
                raise ValueError(f"WNN import source changed: {relative}")
    for relative, item in pins.items():
        path = source_root / relative
        if _digest_file(path, int(item["bytes"])) != (int(item["bytes"]), str(item["sha256"])):
            raise ValueError(f"pinned source changed during imports: {relative}")
    return modules


def _check_output_location(output_dir: Path, binding: dict[str, object]) -> None:
    """Require the fresh exports leaf under the fixed private evaluation evidence root."""
    evidence_root = Path(str(binding["evidence_root"])).resolve(strict=True)
    parent = output_dir.parent.resolve(strict=True)
    if (
        parent != evidence_root
        or str(output_dir) != binding["output_dir"]
        or output_dir.exists()
        or output_dir.is_symlink()
    ):
        raise ValueError(
            "output must be the absent fixed exports child of the approved evidence root"
        )
    resource = binding["resource"]
    required_free = max(
        int(resource["minimum_free_bytes"]),
        int(resource["outputs_bytes"]) + int(resource["free_reserve_bytes"]),
    )
    if shutil.disk_usage(evidence_root).free < required_free:
        raise RuntimeError("approved scratch filesystem lacks output space and free reserve")


def _check_runtime(modules: dict[str, ModuleType], binding: dict[str, object]) -> object:
    """Apply the pinned single-GPU, strict-FP32 inference runtime policy."""
    torch = modules["torch"]
    resource = binding["resource"]
    if (
        sys.byteorder != "little"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != resource["selected_gpu_uuid"]
    ):
        raise RuntimeError("runtime byte order or selected GPU identity differs")
    thread_vars = (
        "WNN_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    if any(os.environ.get(name) != "1" for name in thread_vars):
        raise RuntimeError("worker and BLAS thread limits must all be one")
    if os.environ.get("WNN_GPU_MEMORY_BYTES") != str(resource["allocator_bytes"]):
        raise RuntimeError("GPU allocator cap differs from the approved bound")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one approved CUDA device must be visible")
    if torch.__version__ != "2.14.0+cu130":
        raise RuntimeError("worker PyTorch version differs from accepted checkpoint runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    modules["resources"].configure_torch()
    modules["resources"].configure_strict_fp32()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must remain disabled")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device


def _validate_checkpoint(
    checkpoint: object,
    arm: dict[str, object],
    binding: dict[str, object],
    modules: dict[str, ModuleType],
    device: object,
) -> tuple[object, object, dict[str, object]]:
    """Restrict one safely loaded checkpoint to the fixed architecture and input path."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a dictionary")
    expected = binding["expected_checkpoint"]
    args = checkpoint.get("args")
    model_state = checkpoint.get("model")
    if not isinstance(args, dict) or not isinstance(model_state, dict):
        raise ValueError("checkpoint arguments or state dictionary are malformed")
    required = {
        "arch": expected["arch"],
        "hidden": expected["hidden"],
        "layers": expected["layers"],
        "mean_variance_head": expected["mean_variance_head"],
        "precision": expected["precision"],
        "split_manifest_sha256": binding["split"]["sha256"],
        "representation_mode": arm["representation_mode"],
        "input_lead": expected["input_lead"],
        "online_norm": expected["online_norm"],
        "heuristics": expected["heuristics"],
        "raw_only": expected["raw_only"],
        "research_features": expected["research_features"],
        "research_feature_control": expected["research_feature_control"],
        "privileged_mode": expected["privileged_mode"],
        "prop_input": expected["prop_input"],
    }
    defaults = {"cell": "gru", "enc": "linear", "dropout": expected["dropout"]}
    for key, value in required.items():
        if args.get(key) != value:
            raise ValueError(f"checkpoint recipe argument differs: {key}")
    for key, value in defaults.items():
        if args.get(key, value) != value:
            raise ValueError(f"checkpoint architecture option differs: {key}")
    if type(checkpoint.get("n_in")) is not int or checkpoint["n_in"] != expected["n_in"]:
        raise ValueError("checkpoint input width differs")
    head = model_state.get("head.2.weight")
    torch = modules["torch"]
    if not isinstance(head, torch.Tensor) or tuple(head.shape) != (expected["n_out"], 256):
        raise ValueError("checkpoint output head shape differs")
    for unsupported in ("sentinel_preprocessing", "readout", "quantization", "factorization"):
        if checkpoint.get(unsupported) is not None:
            raise ValueError(f"unsupported checkpoint feature is present: {unsupported}")
    if checkpoint.get("research_feature_prior") is not None:
        raise ValueError("unexpected research-feature prior")
    for key in ("mean", "std"):
        tensor = model_state.get(key)
        expected_norm = binding["representation"]["normalization"][key]
        if (
            not isinstance(tensor, torch.Tensor)
            or str(tensor.dtype) != expected_norm["dtype"]
            or list(tensor.shape) != expected_norm["shape"]
        ):
            raise ValueError(f"checkpoint {key} tensor metadata differs")
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != expected_norm["bytes_sha256"]:
            raise ValueError(f"checkpoint {key} tensor bytes differ")
    representation = modules["representation"]
    input_representation = representation.deserialize_input_representation(
        checkpoint.get("input_representation"), device
    )
    state_sha = representation.input_representation_sha256(input_representation)
    if state_sha != binding["representation"]["state_sha256"]:
        raise ValueError("checkpoint representation state digest differs")
    model = modules["models"].make_model(args, expected["n_in"], expected["n_out"]).to(device)
    model.load_state_dict(model_state, strict=True)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("model parameters are not FP32")
    model._inference_fp32 = True
    input_configuration = modules["environment"].InputConfiguration(
        online_norm=False,
        heuristics=False,
        raw_only=False,
        research_features="none",
        research_feature_control="none",
        privileged_mode="none",
        representation_mode=str(arm["representation_mode"]),
        input_representation=input_representation,
    )
    environment = modules["environment"].TrainingEnvironment(
        root=Path(str(binding["data_root"])),
        source_root=Path(str(binding["source"]["source_root"])),
        argv=[],
        inputs=input_configuration,
    )
    return model, environment, args


def _membership_sha256(groups: list[int]) -> str:
    """Hash ordered row-group membership as ASCII decimal IDs with one LF each."""
    raw = b"".join(str(group).encode("ascii") + b"\n" for group in groups)
    return hashlib.sha256(raw).hexdigest()


def _validate_evaluation_groups(values: object, expected_count: int) -> list[int]:
    """Normalize canonical split IDs without accepting coercion or duplicate rows."""
    try:
        raw_values = list(values)
        groups = [operator.index(value) for value in raw_values]
    except (TypeError, ValueError) as error:
        raise ValueError("evaluation membership must contain integer row-group IDs") from error
    if any(isinstance(value, bool) for value in raw_values):
        raise ValueError("boolean evaluation IDs are not valid row-group IDs")
    if len(groups) != expected_count or len(set(groups)) != expected_count:
        raise ValueError("canonical split membership has the wrong count or duplicate groups")
    if any(group < 0 for group in groups):
        raise ValueError("evaluation membership contains a negative row-group ID")
    return groups


def _validate_split_inventory(
    split: object, split_binding: dict[str, object]
) -> tuple[list[int], int]:
    """Derive valid-file row-group coverage from canonical split components."""
    fit_valid = _validate_evaluation_groups(
        split.valid_groups, int(split_binding["fit_valid_group_count"])
    )
    evaluation = _validate_evaluation_groups(
        split.evaluation_groups, int(split_binding["evaluation_group_count"])
    )
    if set(fit_valid) & set(evaluation):
        raise ValueError("fit-valid and evaluation row groups overlap")
    components = getattr(split, "components", None)
    if not isinstance(components, (tuple, list)):
        raise ValueError("canonical split component inventory is malformed")
    valid_components: list[int] = []
    for item in components:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError("canonical split component entry is malformed")
        source_split, group, component = item
        if type(source_split) is not str or type(group) is not int or type(component) is not int:
            raise ValueError("canonical split component entry has unexpected types")
        if source_split == "valid":
            valid_components.append(group)
    valid_count = len(valid_components)
    expected_count = split_binding["valid_row_group_count"]
    if type(expected_count) is not int or valid_count != expected_count:
        raise ValueError("canonical valid component count differs from its binding")
    if len(set(valid_components)) != valid_count:
        raise ValueError("canonical valid component inventory contains duplicate row groups")
    if set(valid_components) != set(fit_valid) | set(evaluation):
        raise ValueError("canonical valid component inventory differs from fit/evaluation groups")
    if set(valid_components) != set(range(valid_count)):
        raise ValueError("canonical valid row-group inventory is not contiguous from zero")
    return evaluation, valid_count


def _write_json_exclusive(path: Path, value: dict[str, object]) -> tuple[int, str]:
    """Create and fsync one deterministic JSON file without replacing anything."""
    raw = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True).encode("ascii") + b"\n"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return len(raw), hashlib.sha256(raw).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist a completed directory entry set before declaring it complete."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_prediction_memmap(
    path: Path, shape: tuple[int, ...], numpy: ModuleType
) -> tuple[object, object, int, int]:
    """Create an exclusive FP32 NPY file and return stream, map, header, data bytes."""
    dtype = numpy.dtype("<f4")
    data_bytes = int(numpy.prod(shape)) * dtype.itemsize
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    stream = os.fdopen(descriptor, "w+b")
    try:
        header = {
            "descr": numpy.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        }
        numpy.lib.format.write_array_header_2_0(stream, header)
        header_bytes = stream.tell()
        stream.truncate(header_bytes + data_bytes)
        stream.flush()
        os.fsync(stream.fileno())
        array = numpy.memmap(stream, dtype=dtype, mode="r+", offset=header_bytes, shape=shape)
        return stream, array, header_bytes, data_bytes
    except BaseException:
        stream.close()
        raise


def _read_feature_only(
    sequences: object, group: int, expected_shape: tuple[int, int], numpy: ModuleType
) -> object:
    """Use the canonical reader without loading targets or the scored mask."""
    sequence = sequences.read(group, targets=False, mask=False)
    if (
        type(sequence.seq_ix) is not int
        or sequence.seq_ix < 0
        or sequence.targets is not None
        or sequence.scored is not None
        or sequence.features.shape != expected_shape
        or sequence.features.dtype != numpy.float32
    ):
        raise ValueError(f"feature-only sequence schema differs at group {group}")
    if not numpy.isfinite(sequence.features).all():
        raise FloatingPointError(f"nonfinite features in evaluation group {group}")
    return sequence


def _export_arm(
    arm: dict[str, object],
    checkpoint_path: Path,
    groups: list[int],
    sequences: object,
    binding: dict[str, object],
    modules: dict[str, ModuleType],
    device: object,
    output_dir: Path,
) -> dict[str, object]:
    """Export one accepted arm in group order, using canonical source inference."""
    numpy = modules["numpy"]
    torch = modules["torch"]
    loader = modules["loader"]
    checkpoint, digest, _stat = loader._load_checkpoint(
        checkpoint_path, int(binding["expected_checkpoint"]["model_byte_cap"])
    )
    if digest != arm["model"]["sha256"]:
        raise ValueError(f"{arm['arm']} model changed after preflight")
    model, environment, _args = _validate_checkpoint(checkpoint, arm, binding, modules, device)
    prediction_path = output_dir / str(arm["arm"]) / "predictions.npy"
    os.mkdir(prediction_path.parent, mode=0o700)
    shape = tuple(binding["outputs"]["shape"])
    data_bytes = int(binding["outputs"]["data_bytes_each"])
    if data_bytes != int(numpy.prod(shape)) * numpy.dtype("<f4").itemsize:
        raise ValueError("prediction data-byte contract differs from shape and dtype")
    stream, array, header_bytes, mapped_data_bytes = _create_prediction_memmap(
        prediction_path, shape, numpy
    )
    predictor = modules["evaluation"].predict
    try:
        if data_bytes != mapped_data_bytes:
            raise ValueError("prediction array allocation differs from its byte contract")
        for row, group in enumerate(groups):
            sequence = _read_feature_only(
                sequences,
                group,
                (
                    int(binding["split"]["sequence_rows"]),
                    int(binding["split"]["feature_count"]),
                ),
                numpy,
            )
            batch = torch.from_numpy(numpy.ascontiguousarray(sequence.features)[None, ...]).to(
                device=device, dtype=torch.float32
            )
            prediction = predictor(
                model, batch, int(binding["split"]["chunk_rows"]), device, environment=environment
            )
            if (
                prediction.shape
                != (
                    1,
                    int(binding["split"]["sequence_rows"]),
                    int(binding["split"]["output_count"]),
                )
                or prediction.dtype != numpy.float32
                or not numpy.isfinite(prediction).all()
            ):
                raise FloatingPointError(f"invalid FP32 prediction for evaluation group {group}")
            array[row] = prediction[0]
            del batch, prediction, sequence
        array.flush()
    finally:
        if array is not None:
            del array
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
    del model, environment, checkpoint
    prediction_size, prediction_sha = _digest_file(
        prediction_path, int(binding["outputs"]["data_bytes_each"]) + 4096
    )
    if prediction_size != data_bytes + header_bytes:
        raise ValueError("prediction NPY file size differs from its header and array")
    sidecar = {
        "schema": "fg19.first-cell-feature-only-prediction.v1",
        "status": "PREDICTION_EXPORT_ONLY_NOT_SCORED",
        "binding_sha256": BINDING_SHA256,
        "arm": arm["arm"],
        "run_name": arm["run_name"],
        "unit": arm["unit"],
        "invocation_id": arm["invocation_id"],
        "selected_epoch": 5,
        "successful_updates": 44760,
        "terminal_acceptance": arm["terminal_acceptance"],
        "exact_stop": arm["exact_stop"],
        "checkpoint": {**arm["model"], "loaded_sha256": digest},
        "representation_mode": arm["representation_mode"],
        "source": binding["source"],
        "split": {**binding["split"], "role": "fit-excluded-exposed-development-evaluation"},
        "ordered_groups": groups,
        "membership_encoding": "ASCII decimal row-group IDs, one per line, final LF",
        "membership_sha256": _membership_sha256(groups),
        "feature_only": {"targets_read": False, "scored_mask_read": False},
        "representation": {
            "mode": arm["representation_mode"],
            "artifact_sha256": binding["representation"]["artifact_sha256"],
            "state_sha256": binding["representation"]["state_sha256"],
            "normalization": binding["representation"]["normalization"],
        },
        "inference": {
            "precision": "strict_fp32",
            "autocast": False,
            "tf32": False,
            "chunk_rows": binding["split"]["chunk_rows"],
            "sequence_batch_size": 1,
        },
        "prediction": {
            "path": "predictions.npy",
            "bytes": prediction_size,
            "npy_header_bytes": header_bytes,
            "data_bytes": data_bytes,
            "sha256": prediction_sha,
            "dtype": "float32-little-endian",
            "shape": list(binding["outputs"]["shape"]),
        },
        "scoring_performed": False,
    }
    sidecar_size, sidecar_sha = _write_json_exclusive(
        prediction_path.parent / "PREDICTION_SIDECAR.json", sidecar
    )
    _fsync_directory(prediction_path.parent)
    return {
        "arm": arm["arm"],
        "directory": str(arm["arm"]),
        "run_name": arm["run_name"],
        "unit": arm["unit"],
        "invocation_id": arm["invocation_id"],
        "representation_mode": arm["representation_mode"],
        "selected_epoch": 5,
        "successful_updates": 44760,
        "checkpoint": {**arm["model"], "loaded_sha256": digest},
        "membership_sha256": _membership_sha256(groups),
        "prediction_path": f"{arm['arm']}/predictions.npy",
        "sidecar_path": f"{arm['arm']}/PREDICTION_SIDECAR.json",
        "prediction_bytes": prediction_size,
        "prediction_sha256": prediction_sha,
        "sidecar_bytes": sidecar_size,
        "sidecar_sha256": sidecar_sha,
    }


def main(argv: list[str] | None = None) -> int:
    """Run one fixed-panel feature-only export after all preflight gates pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-base", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    script_dir = Path(__file__).resolve().parent
    binding = _load_binding(script_dir)
    project_base = args.project_base.resolve(strict=True)
    data_root = args.data_root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    support_root = args.support_root.resolve(strict=True)
    evidence_root = args.evidence_root.resolve(strict=True)
    output_dir = args.output_dir.absolute()
    if (
        args.project_base.absolute() != project_base
        or args.data_root.absolute() != data_root
        or args.source_root.absolute() != source_root
        or args.support_root.absolute() != support_root
        or args.evidence_root.absolute() != evidence_root
    ):
        raise ValueError("root arguments must not traverse symbolic links")
    for actual, key in (
        (project_base, "project_base"),
        (data_root, "data_root"),
        (source_root, "source_root"),
        (support_root, "support_root"),
        (evidence_root, "evidence_root"),
    ):
        if str(actual) != binding[key]:
            raise ValueError(f"argument root differs from fixed {key}")
    if (
        data_root != project_base / "data"
        or source_root != project_base / "source-fg19-cache-68ff6829"
        or Path(binding["scratch_root"]) != project_base / "research_tmp"
        or Path(binding["fit_receipts_root"])
        != project_base / "research_tmp" / "fg19_child_allthree_fit_v1"
        or support_root != project_base / "research_tmp" / "fg19_child_allthree_verify_v2"
        or evidence_root != project_base / "research_tmp" / "fg19_first_cell_f0_s1001_evaluation_v1"
    ):
        raise ValueError("B path layout differs from the fixed campaign contract")
    if str(output_dir) != binding["output_dir"]:
        raise ValueError("output directory differs from its fixed binding")
    if script_dir != evidence_root:
        raise ValueError(
            "worker source and fixed binding must be staged at the private evidence root"
        )
    registered_data = _verify_registered_data_receipt(binding)
    portable_manifest_identity = _verify_portable_manifest(binding)
    _configure_portable_manifest(binding)
    source_observations_before_split = _observe_registered_source_identities(
        data_root, binding["registered_data"]["source_identities"]
    )
    initial_data_observation = _observe_registered_data_identity(data_root, registered_data)
    pins = _verify_source_tree(source_root, binding, script_dir)
    _verify_receipts_and_models(data_root, binding)
    modules = _import_source(source_root, support_root, pins)
    device = _check_runtime(modules, binding)
    if _verify_portable_manifest(binding) != portable_manifest_identity:
        raise ValueError("approved portable source registry changed during preflight")
    _configure_portable_manifest(binding)
    split = modules["splits"].load_campaign_split(
        Path(str(binding["split"]["absolute_path"])), data_root
    )
    if split.sha256 != binding["split"]["sha256"] or split.fold != 0 or split.n_folds != 4:
        raise ValueError("canonical split loader returned a different F0 identity")
    groups, valid_row_group_count = _validate_split_inventory(split, binding["split"])
    source_observations_after_split = _observe_registered_source_identities(
        data_root, binding["registered_data"]["source_identities"]
    )
    if source_observations_after_split != source_observations_before_split:
        raise ValueError("registered portable source stats changed while loading the split")
    _check_output_location(output_dir, binding)
    data_path = Path(str(registered_data["path"]))
    before_sequence_open = _observe_registered_data_identity(data_root, registered_data)
    if before_sequence_open != initial_data_observation:
        raise ValueError("registered validation-data identity changed during preflight")
    sequence_file = modules["data"].SequenceFile(data_path)
    try:
        if len(sequence_file) != valid_row_group_count:
            raise ValueError("valid parquet row-group count differs from canonical components")
        os.mkdir(output_dir, mode=0o700)
        _fsync_directory(evidence_root)
        arms_result = []
        for arm in binding["arms"]:
            checkpoint_path = data_root / str(arm["model"]["relative_path"])
            arms_result.append(
                _export_arm(
                    arm,
                    checkpoint_path,
                    groups,
                    sequence_file,
                    binding,
                    modules,
                    device,
                    output_dir,
                )
            )
            modules["torch"].cuda.synchronize(device)
            modules["torch"].cuda.empty_cache()
    finally:
        sequence_file._pf.close()
    after_export_observation = _observe_registered_data_identity(data_root, registered_data)
    if after_export_observation != before_sequence_open:
        raise ValueError("registered validation-data identity changed during export")
    manifest = {
        "schema": "fg19.first-cell-evaluation-export-manifest.v1",
        "status": "COMPLETE_PREDICTION_EXPORTS_NOT_SCORED",
        "created_utc": datetime.now(UTC).isoformat(),
        "binding_sha256": BINDING_SHA256,
        "source": binding["source"],
        "split": {
            **binding["split"],
            "role": "fit-excluded-exposed-development-evaluation",
        },
        "data_identity": {
            "schema": "fg19.registered-valid-data-identity.v1",
            "admission": binding["registered_data"]["admission"],
            "portable_manifest": portable_manifest_identity,
            "registered_source_identities": binding["registered_data"]["source_identities"],
            "observed_sources_before_split": source_observations_before_split,
            "observed_sources_after_split": source_observations_after_split,
            "registered_valid_parquet": registered_data,
            "observed_before_sequence_open": before_sequence_open,
            "observed_after_export": after_export_observation,
        },
        "ordered_groups": groups,
        "membership_sha256": _membership_sha256(groups),
        "arms": arms_result,
        "quality_or_metric_claim": False,
    }
    _write_json_exclusive(output_dir / "EXPORT_MANIFEST_V1.json", manifest)
    _fsync_directory(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
