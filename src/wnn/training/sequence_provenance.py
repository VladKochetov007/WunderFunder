from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist

from wnn.research_reporting import RunMetadata, RunRecorder
from wnn.training.sequence_distributed import DistributedContext


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
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(context.local_rank),
        "capability": [properties.major, properties.minor],
        "visible_devices": torch.cuda.device_count(),
        "world_size": context.world_size,
        "rank": context.rank,
        "local_rank": context.local_rank,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "autocast": False,
        "amp_scaler": False,
        "attention_backend": "math",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def runtime_identity(context: DistributedContext) -> dict[str, object]:
    """Collect identical software and per-rank device metadata on every worker."""
    local = _local_runtime_identity(context)
    if context.process_group:
        gathered: list[dict[str, object] | None] = [None] * context.world_size
        dist.all_gather_object(gathered, local)
        workers = [item for item in gathered if item is not None]
    else:
        workers = [local]
    if len(workers) != context.world_size:
        raise RuntimeError("could not collect runtime identity from every rank")
    common_fields = (
        "python",
        "platform",
        "torch",
        "cuda_runtime",
        "cudnn",
        "visible_devices",
        "world_size",
        "matmul_precision",
        "matmul_tf32",
        "cudnn_tf32",
        "autocast",
        "amp_scaler",
        "attention_backend",
        "cublas_workspace_config",
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
    return {
        "schema": "wnn.sequence-run-identity.v1",
        "config_sha256": canonical_sha256(config),
        "data_sha256": canonical_sha256(data_value),
        "data": data_value,
        "source_sha256": source["sha256"],
        "source_files": source["files"],
        "runtime": runtime,
        "world_size": world_size,
    }


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
    """Require a manually approved engineering profile with exact fit identities."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("profile receipt must contain a JSON object")
    if value.get("receipt_schema") != "wnn.sequence-profile.v1":
        raise ValueError("unsupported engineering profile receipt")
    if value.get("status") != "profile_complete" or value.get("approved") is not True:
        raise ValueError("full fitting requires an explicitly approved completed profile receipt")
    if value.get("identity") != identity:
        raise ValueError("profile receipt config/data/source/runtime identity does not match")
    for name in ("updates", "forwarded_rows", "real_rows", "valid_rows", "epochs_traversed"):
        if not _positive_integer(value.get(name)):
            raise ValueError(f"profile receipt {name} must be a positive integer")
    if value["forwarded_rows"] < value["real_rows"] or value["valid_rows"] > value["real_rows"]:
        raise ValueError("profile work counts are internally inconsistent")
    if not _finite_nonnegative(value.get("elapsed_seconds")) or value["elapsed_seconds"] <= 0:
        raise ValueError("profile receipt elapsed time must be finite and positive")
    if value.get("scientific_checkpoint_written") is not False:
        raise ValueError("engineering profiles must not write a scientific checkpoint")
    errors = value.get("errors")
    if errors != [] or value.get("error_count") != 0:
        raise ValueError("profile receipt must have no runtime errors")
    losses = value.get("losses")
    if not isinstance(losses, dict) or set(losses) != {"label", "distillation", "hint", "total"}:
        raise ValueError("profile receipt is missing its objective summary")
    if not all(_finite_nonnegative(loss) for loss in losses.values()):
        raise ValueError("profile losses must be finite and nonnegative")
    if value.get("model_parameters") != identity.get("model_parameters"):
        raise ValueError("profile model parameter count does not match this fit")
    memory = value.get("memory_by_rank")
    world_size = identity.get("world_size")
    if not isinstance(world_size, int) or isinstance(world_size, bool):
        raise ValueError("run identity has an invalid world size")
    if not isinstance(memory, list) or len(memory) != world_size:
        raise ValueError("profile receipt must record memory metrics for every rank")
    if [item.get("rank") for item in memory if isinstance(item, dict)] != list(range(world_size)):
        raise ValueError("profile resource records are missing or misordered by rank")
    memory_fields = (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "device_total_memory_bytes",
        "host_peak_rss_bytes",
    )
    for item in memory:
        if not isinstance(item, dict) or not all(
            _positive_integer(item.get(field)) for field in memory_fields
        ):
            raise ValueError("profile resource values must be positive integer byte counts")
        if item["peak_allocated_bytes"] > item["peak_reserved_bytes"]:
            raise ValueError("profile allocated memory cannot exceed reserved memory")
    if value.get("peak_memory_bytes") != max(item["peak_allocated_bytes"] for item in memory):
        raise ValueError("profile aggregate allocated memory does not match per-rank values")
    if value.get("peak_reserved_bytes") != max(item["peak_reserved_bytes"] for item in memory):
        raise ValueError("profile aggregate reserved memory does not match per-rank values")
    return file_sha256(path)


def _positive_integer(value: object) -> bool:
    """Accept only positive Python integer counters, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_nonnegative(value: object) -> bool:
    """Accept finite nonnegative numeric receipt values, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )
