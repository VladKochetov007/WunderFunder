from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

LEGACY_STRICT_FP32 = "strict_fp32_no_autocast_no_tf32_math_sdpa"
BF16_BF16X9 = "bf16_bf16x9"
SUPPORTED_ATTENTION_BACKENDS = frozenset(("math", "efficient", "flex"))
COMPILER_RECOMPILE_LIMIT = 64


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


def _bf16_bf16x9_identity(attention_backend: str) -> dict[str, object]:
    """Mirror PR40's precision identity for the one accepted nonlegacy mode."""
    if attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError("teacher precision names an unsupported attention backend")
    return {
        "mode": BF16_BF16X9,
        "attention_backend": attention_backend,
        "cublas_emulate_single_precision": "1",
        "parameter_dtype": "float32",
        "gradient_dtype": "float32",
        "optimizer_state_dtype": "float32",
        "activation_cache_dtype": "float32",
        "amp_scaler": False,
        "model_compute": "BF16 autocast plus process-wide cuBLAS single-precision emulation",
        "model_forward": "FP32 tensor interface with BF16 autocast and cuBLAS emulation",
        "loss": "FP32 after prediction and hint casts",
        "optimizer": "process-wide cuBLAS emulation may affect Muon operations",
    }


def _compiler_identity(enabled: bool) -> dict[str, object]:
    """Return the PR40 compiler receipt dictated by the producer config."""
    return {
        "enabled": enabled,
        "backend": "inductor" if enabled else None,
        "mode": "default" if enabled else None,
        "fullgraph": False,
        "dynamic": None,
        "recompile_limit": COMPILER_RECOMPILE_LIMIT if enabled else None,
    }


def _validate_precision_contract(
    config: dict[str, Any], provenance: dict[str, Any], runtime: dict[str, Any]
) -> object:
    """Bind legacy strict-FP32 or PR40 BF16x9 attestations to the config."""
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("teacher precision recipe has no model config")
    mode = config.get("precision_mode", "fp32")
    attention_backend = model.get("attention_backend", "math")
    compile_blocks = model.get("compile_blocks", False)
    if (
        not isinstance(mode, str)
        or not isinstance(attention_backend, str)
        or type(compile_blocks) is not bool
    ):
        raise ValueError("teacher precision recipe is malformed")

    if mode == BF16_BF16X9:
        if model.get("precision_mode") != mode:
            raise ValueError("teacher runtime model precision differs from its config")
        expected = _bf16_bf16x9_identity(attention_backend)
        if (
            provenance.get("precision") != expected
            or runtime.get("precision") != expected
            or runtime.get("attention_backend") != attention_backend
        ):
            raise ValueError("teacher BF16x9 precision provenance does not match its config")
        if runtime.get("compiler") != _compiler_identity(compile_blocks):
            raise ValueError("teacher compiler provenance does not match its config")
        return expected

    if mode == "fp32" and provenance.get("precision") == LEGACY_STRICT_FP32:
        if (
            attention_backend != "math"
            or compile_blocks
            or "precision" in runtime
            or "compiler" in runtime
            or runtime.get("attention_backend") != "math"
        ):
            raise ValueError("legacy strict-FP32 provenance does not match its config")
        return LEGACY_STRICT_FP32

    raise ValueError("teacher precision provenance does not match a supported config")


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
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise ValueError("teacher fit source fingerprints are incomplete")
    for key, fingerprint in data["normalization"]["source_fingerprints"].items():
        if (
            key not in member_sources
            or not isinstance(fingerprint, dict)
            or fingerprint.get("sha256") != sources[key].get("sha256")
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
    precision = _validate_precision_contract(config, provenance, runtime)
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
