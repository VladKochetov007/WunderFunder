from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from wnn.training.sequence_cache_normalization_lineage import validate_normalization_ancestry
from wnn.training.sequence_cache_precision import validate_precision_contract

_PARQUET_FOOTER_FIELDS = frozenset(("bytes", "mtime_ns", "footer_bytes", "footer_sha256"))


def canonical_sha256(value: object) -> str:
    """Hash JSON-compatible provenance with stable encoding."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("teacher provenance is not finite JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: object) -> bool:
    """Return whether value is a lowercase SHA-256 digest."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _verify_portable_parquet_source(
    name: str, record: dict[str, Any], primary_split_sha256: object
) -> str:
    """Bind footer-only run metadata to the registry verified by the data loader."""
    registry_value = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    if not registry_value:
        raise ValueError("footer-only teacher sources require the portable data registry")
    registry = Path(registry_value)
    if registry.is_symlink() or not registry.is_file() or registry.stat().st_size > 4 * 1024**2:
        raise ValueError("portable data registry must be a small regular file")
    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("portable data registry is unreadable") from error
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or not _valid_sha256(primary_split_sha256)
        or payload.get("split_manifest_sha256") != primary_split_sha256
        or not isinstance(payload.get("sources"), dict)
    ):
        raise ValueError("portable data registry does not bind the primary split")
    original_identity = {key: value for key, value in record.items() if key != "sha256"}
    entry = payload["sources"].get(name)
    if (
        not isinstance(entry, dict)
        or entry.get("original_identity") != original_identity
        or not _valid_sha256(entry.get("sha256"))
        or ("sha256" in record and record["sha256"] != entry["sha256"])
    ):
        raise ValueError("portable registry source identity differs from teacher provenance")
    return entry["sha256"]


def common_provenance(producer: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """Normalize seed-specific checkpoint provenance to the shared fit contract."""
    config = producer.get("config")
    identity = provenance.get("identity")
    if (
        not isinstance(config, dict)
        or provenance.get("schema") != "wnn.sequence-provenance.v1"
        or not isinstance(identity, dict)
        or identity.get("schema") != "wnn.sequence-run-identity.v1"
        or identity.get("config_sha256") != canonical_sha256(config)
    ):
        raise ValueError("teacher run provenance is incomplete or detached from its recipe")
    data = identity.get("data")
    normalization = provenance.get("normalization")
    training_mask = data.get("training_mask") if isinstance(data, dict) else None
    expected_mask = {
        "is_scored": "is_scored AND row_index>=99",
        "all_after_warmup": "all_after_warmup AND row_index>=99",
    }.get(training_mask)
    sources = data.get("sources") if isinstance(data, dict) else None
    members = producer.get("fit_members")
    if (
        not isinstance(data, dict)
        or data.get("split_sha256") != producer.get("split_sha256")
        or data.get("fit_members") != members
        or data.get("input_schema") != "raw112-v1"
        or expected_mask is None
        or data.get("mask") != expected_mask
        or not isinstance(sources, dict)
        or not isinstance(members, list)
        or not isinstance(data.get("normalization"), dict)
        or not isinstance(data["normalization"].get("source_fingerprints"), dict)
        or identity.get("data_sha256") != canonical_sha256(data)
        or not isinstance(normalization, dict)
        or not isinstance(identity.get("source_files"), dict)
        or identity.get("source_sha256") != canonical_sha256(identity["source_files"])
    ):
        raise ValueError("teacher data, source or normalizer provenance is malformed")
    member_sources: set[str] = set()
    for member in members:
        if (
            not isinstance(member, list)
            or len(member) != 2
            or member[0] not in ("train", "valid")
            or type(member[1]) is not int
            or member[1] < 0
        ):
            raise ValueError("teacher fit membership is malformed")
        key = f"datasets/{member[0]}.parquet"
        member_sources.add(key)
        record = sources.get(key)
        if not isinstance(record, dict):
            raise ValueError("teacher fit source fingerprints are incomplete")
        if "footer_sha256" in record:
            if (
                not _PARQUET_FOOTER_FIELDS.issubset(record)
                or set(record) - (_PARQUET_FOOTER_FIELDS | {"sha256"})
                or type(record.get("bytes")) is not int
                or record["bytes"] <= 8
                or type(record.get("mtime_ns")) is not int
                or record["mtime_ns"] < 0
                or type(record.get("footer_bytes")) is not int
                or not 0 <= record["footer_bytes"] <= record["bytes"] - 8
                or not _valid_sha256(record.get("footer_sha256"))
                or ("sha256" in record and not _valid_sha256(record["sha256"]))
            ):
                raise ValueError("teacher parquet footer fingerprint is malformed")
            _verify_portable_parquet_source(key, record, data.get("primary_split_sha256"))
        elif set(record) != {"sha256"} or not _valid_sha256(record.get("sha256")):
            raise ValueError("teacher fit source fingerprints are incomplete")
    for key, fingerprint in data["normalization"]["source_fingerprints"].items():
        if (
            key not in member_sources
            or not isinstance(fingerprint, dict)
            or fingerprint != sources[key]
        ):
            raise ValueError("teacher normalizer source fingerprints are inconsistent")
    if not data["normalization"]["source_fingerprints"]:
        raise ValueError("teacher normalizer source fingerprints are incomplete")
    means, scales = normalization.get("mean"), normalization.get("std")
    if (
        not isinstance(means, list)
        or not isinstance(scales, list)
        or len(means) != 112
        or len(scales) != 112
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in means)
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in scales
        )
    ):
        raise ValueError("teacher actual normalizer must contain finite raw112 moments")
    if (
        type(config.get("epochs")) is not int
        or provenance.get("completed_epochs") != config["epochs"]
        or type(provenance.get("global_updates")) is not int
        or provenance["global_updates"] < 1
        or provenance.get("fixed_checkpoint_policy") != "configured-final-epoch"
    ):
        raise ValueError("teacher provenance does not attest a complete fixed fit")
    runtime = identity.get("runtime")
    devices = runtime.get("devices") if isinstance(runtime, dict) else None
    if not isinstance(runtime, dict) or not isinstance(devices, list) or not devices:
        raise ValueError("teacher runtime provenance is incomplete")
    world_size = identity.get("world_size")
    if (
        type(world_size) is not int
        or world_size < 1
        or runtime.get("world_size") != world_size
        or len(devices) != world_size
    ):
        raise ValueError("teacher runtime world size is inconsistent")
    if (
        runtime.get("matmul_precision") != "highest"
        or runtime.get("matmul_tf32") is not False
        or runtime.get("cudnn_tf32") is not False
        or runtime.get("autocast") is not False
        or runtime.get("amp_scaler") is not False
    ):
        raise ValueError("teacher runtime has incompatible FP32 process controls")
    precision = validate_precision_contract(config, provenance, runtime)
    # Historical scalar precision receipts predate membership-bound normalizer hashes.
    if isinstance(precision, dict):
        validate_normalization_ancestry(config, data, normalization, members)
    semantic_devices = []
    for device in devices:
        if (
            not isinstance(device, dict)
            or not isinstance(device.get("name"), str)
            or not device["name"]
            or not isinstance(device.get("capability"), list)
            or any(type(value) is not int or value < 0 for value in device["capability"])
            or type(device.get("total_memory_bytes")) is not int
            or device["total_memory_bytes"] <= 0
            or type(device.get("rank")) is not int
            or type(device.get("local_rank")) is not int
        ):
            raise ValueError("teacher device provenance is malformed")
        semantic_devices.append(
            {
                "name": device["name"],
                "capability": device["capability"],
                "total_memory_bytes": device["total_memory_bytes"],
            }
        )
    semantic_devices.sort(key=canonical_sha256)
    common_identity = dict(identity)
    common_identity.pop("config_sha256")
    common_runtime = dict(runtime)
    common_runtime["devices"] = semantic_devices
    common_identity["runtime"] = common_runtime
    return {
        "schema": provenance["schema"],
        "identity": common_identity,
        "normalization": normalization,
        "completed_epochs": provenance.get("completed_epochs"),
        "global_updates": provenance.get("global_updates"),
        "precision": precision,
        "fixed_checkpoint_policy": provenance.get("fixed_checkpoint_policy"),
    }
