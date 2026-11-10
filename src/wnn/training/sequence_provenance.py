from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist

from wnn.research_reporting import RunMetadata, RunRecorder
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_precision import runtime_precision_identity


def json_value(value: object) -> object:
    """Normalize supported provenance values into deterministic JSON data."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("provenance object keys must be strings")
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        return json_value(scalar())
    raise TypeError(f"unsupported provenance value: {type(value).__name__}")


def canonical_sha256(value: object) -> str:
    """Hash a normalized JSON value using a stable key and whitespace policy."""
    encoded = json.dumps(
        json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file in bounded blocks without loading its payload into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(source_root: Path, entrypoint: Path) -> dict[str, object]:
    """Hash every source file under src plus the public entrypoint for exact resume."""
    root = source_root.resolve()
    paths = sorted((root / "src").rglob("*.py"))
    resolved_entrypoint = entrypoint.resolve()
    if resolved_entrypoint not in paths:
        paths.append(resolved_entrypoint)
    files = {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}
    return {"sha256": canonical_sha256(files), "files": files}


def _local_runtime_identity(context: DistributedContext) -> dict[str, object]:
    """Capture strict-FP32 software, device, and process identity before fitting."""
    properties = torch.cuda.get_device_properties(context.local_rank)
    driver_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    driver_versions = {line.strip() for line in driver_result.stdout.splitlines() if line.strip()}
    if len(driver_versions) != 1:
        raise RuntimeError("could not establish one NVIDIA driver identity")
    raw_uuid = getattr(properties, "uuid", None)
    device_uuid = None
    if raw_uuid is not None:
        device_uuid = raw_uuid.decode("utf-8") if isinstance(raw_uuid, bytes) else str(raw_uuid)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": driver_versions.pop(),
        "device": torch.cuda.get_device_name(context.local_rank),
        "capability": [properties.major, properties.minor],
        "device_uuid": device_uuid,
        "device_total_memory_bytes": int(properties.total_memory),
        "visible_devices": torch.cuda.device_count(),
        "world_size": context.world_size,
        "rank": context.rank,
        "local_rank": context.local_rank,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "autocast": False,
        "amp_scaler": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def runtime_identity(context: DistributedContext) -> dict[str, object]:
    """Collect identical software and per-rank device metadata on every worker."""
    local: dict[str, object] | None = None
    local_error: str | None = None
    try:
        local = _local_runtime_identity(context)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    if context.process_group:
        gathered: list[tuple[dict[str, object] | None, str | None]] = [
            (None, None)
        ] * context.world_size
        dist.all_gather_object(gathered, (local, local_error))
        failures = [f"rank {rank}: {error}" for rank, (_, error) in enumerate(gathered) if error]
        if failures:
            raise RuntimeError("could not collect runtime identity: " + "; ".join(failures))
        workers = [identity for identity, _ in gathered if identity is not None]
    else:
        if local is None or local_error is not None:
            raise RuntimeError(
                f"could not collect runtime identity: {local_error or 'empty result'}"
            )
        workers = [local]
    if len(workers) != context.world_size:
        raise RuntimeError("could not collect runtime identity from every rank")
    common_fields = (
        ("python", "platform", "torch", "cuda_runtime", "cudnn", "nvidia_driver")
        + ("visible_devices", "world_size", "matmul_precision", "matmul_tf32", "cudnn_tf32")
        + ("autocast", "amp_scaler", "cublas_workspace_config")
    )
    baseline = workers[0]
    if any(
        any(worker.get(field) != baseline.get(field) for field in common_fields)
        for worker in workers
    ):
        raise RuntimeError("torchrun workers disagree on their runtime precision identity")
    devices = [
        {
            "rank": worker["rank"],
            "local_rank": worker["local_rank"],
            "name": worker["device"],
            "capability": worker["capability"],
            "uuid": worker["device_uuid"],
            "total_memory_bytes": worker["device_total_memory_bytes"],
        }
        for worker in sorted(workers, key=lambda item: int(item["rank"]))
    ]
    return {field: baseline[field] for field in common_fields} | {"devices": devices}


def run_identity(
    config: dict[str, object],
    data_provenance: Mapping[str, object],
    source: dict[str, object],
    runtime: dict[str, object],
    world_size: int,
) -> dict[str, object]:
    """Bind configuration, supplied data/teacher, source, runtime, and exact rank count."""
    data_value = json_value(data_provenance)
    identity = {
        "schema": "wnn.sequence-run-identity.v1",
        "config_sha256": canonical_sha256(config),
        "data_sha256": canonical_sha256(data_value),
        "data": data_value,
        "source_sha256": source["sha256"],
        "source_files": source["files"],
        "runtime": runtime_precision_identity(config, runtime),
        "world_size": world_size,
    }
    if config.get("profile_reload", False):
        identity["profile_reload_required"] = True
    return identity


def assert_same_identity(identity: dict[str, object], context: DistributedContext) -> None:
    """Reject rank-local config, data, or source differences before any optimization."""
    digest = canonical_sha256(identity)
    if not context.process_group:
        return
    hashes: list[str | None] = [None] * context.world_size
    dist.all_gather_object(hashes, digest)
    if any(value != digest for value in hashes):
        raise RuntimeError("torchrun ranks disagree on the source-qualified run identity")


def recorder_for_run(
    output: Path,
    config: dict[str, object],
    identity: dict[str, object],
) -> RunRecorder:
    """Initialize the repository's immutable run metadata and append-only metrics."""
    metadata = RunMetadata(
        source={"sha256": identity["source_sha256"], "files": identity["source_files"]},
        config=json_value(config),
        provenance={
            "data_sha256": identity["data_sha256"],
            "data": identity["data"],
            "runtime": identity["runtime"],
            "world_size": identity["world_size"],
            "model_parameters": identity.get("model_parameters"),
        },
    )
    return RunRecorder(output, metadata)


def validate_profile_receipt(path: Path, identity: dict[str, object]) -> str:
    """Validate a profile receipt through the focused receipt component."""
    from wnn.training.sequence_profile_receipt import validate_profile_receipt as validate

    return validate(path, identity)
