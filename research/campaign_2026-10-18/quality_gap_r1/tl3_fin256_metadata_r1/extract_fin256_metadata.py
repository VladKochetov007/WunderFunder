#!/usr/bin/env python3
"""Extract metadata from the two pinned fin256 initializer checkpoints without exporting weights."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Any is required for heterogeneous checkpoint mappings returned by Torch's restricted loader.
SCHEMA = "quality_gap.tl3_fin256_parent_metadata.v1"
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
MAX_DEPTH = 8
MAX_ITEMS = 4096
TIME_LIMIT_SECONDS = 120
RESOURCE_CONTRACT = {
    "cpu_quota_cores": 0.5,
    "memory_max_bytes": 1_500_000_000,
    "swap_max_bytes": 0,
    "runtime_max_seconds": 120,
}

# Exact resolved parent endpoints observed through the fin256_s3/s4 symlink paths.
# Root must freshly bind each SHA-256 and admission receipt before execution.
ALLOWLIST: dict[str, tuple[str, int]] = {
    "fin256_s3_ep3": (
        "/home/grinchenkoeg/WunderFunder/artifacts/runs/fin256_s3/model_ep3.pt",
        3_933_599,
    ),
    "fin256_s4_ep3": (
        "/home/grinchenkoeg/WunderFunder/artifacts/runs/fin256_s4/model_ep3.pt",
        3_933_599,
    ),
}
SOURCE_OBSERVATION = {
    "probe_receipt_sha256": "93c7bda0dc3cad159d456022661ec2ef670bb4b6776257da55c8f662a0f18a18",
    "logical_root": "/home/grinchenkoeg/WunderFunder-creative",
    "resolved_root": "/home/grinchenkoeg/WunderFunder",
    "symlink_targets": {
        "fin256_s3": "/home/grinchenkoeg/WunderFunder/artifacts/runs/fin256_s3",
        "fin256_s4": "/home/grinchenkoeg/WunderFunder/artifacts/runs/fin256_s4",
    },
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_RE = re.compile(r"secret|password|token|credential|api.?key|access.?key", re.IGNORECASE)
_OMIT_KEYS = {
    "model",
    "optimizer",
    "optimizers",
    "schedulers",
    "scaler",
    "torch_rng",
    "cuda_rng",
    "numpy_rng",
    "training_state",
}


class MetadataError(ValueError):
    """Raised when a source, checkpoint or execution binding is not exact."""


def require(condition: bool, message: str) -> None:
    """Raise a concise fail-closed error when a contract condition is false."""
    if not condition:
        raise MetadataError(message)


def _signature(path: Path) -> tuple[int, int, int, int, int]:
    """Return the stable identity tuple for a regular, non-symlink input."""
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")
    require(not path.is_symlink(), f"symlink input rejected: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode)


def sha256_file(path: Path) -> str:
    """Hash one bounded file using a fixed-size buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: Any) -> list[dict[str, Any]]:
    """Validate a root-bound manifest against the exact two-path allowlist."""
    require(isinstance(manifest, dict), "manifest must be an object")
    require(manifest.get("schema") == SCHEMA, "manifest schema mismatch")
    require(manifest.get("status") == "ROOT_HASH_BOUND", "fresh root-bound manifest required")
    require(manifest.get("resource_contract") == RESOURCE_CONTRACT, "resource contract mismatch")
    require(
        manifest.get("source_observation") == SOURCE_OBSERVATION,
        "source observation binding differs",
    )
    require(
        _SHA_RE.fullmatch(str(manifest.get("root_admission_receipt_sha256", ""))) is not None,
        "root admission receipt SHA-256 missing",
    )
    inputs = manifest.get("inputs")
    require(isinstance(inputs, list) and len(inputs) == len(ALLOWLIST), "exact two inputs required")
    seen: set[str] = set()
    total_bytes = 0
    for item in inputs:
        require(isinstance(item, dict), "input entry must be an object")
        require(
            set(item) == {"id", "path", "expected_bytes", "expected_sha256"},
            "input entry fields differ",
        )
        logical_id = item.get("id")
        require(isinstance(logical_id, str) and logical_id in ALLOWLIST, "input id not allowlisted")
        require(logical_id not in seen, "duplicate input id")
        seen.add(logical_id)
        expected_path, expected_bytes = ALLOWLIST[logical_id]
        require(item.get("path") == expected_path, f"unexpected path for {logical_id}")
        require(
            item.get("expected_bytes") == expected_bytes, f"unexpected size pin for {logical_id}"
        )
        require(0 < expected_bytes <= MAX_FILE_BYTES, f"file bound for {logical_id}")
        require(
            _SHA_RE.fullmatch(str(item.get("expected_sha256", ""))) is not None,
            f"fresh SHA-256 missing for {logical_id}",
        )
        total_bytes += expected_bytes
    require(seen == set(ALLOWLIST), "input inventory differs from allowlist")
    require(total_bytes <= MAX_TOTAL_BYTES, "aggregate file bound")
    return inputs


def safe_metadata(value: Any, *, depth: int = 0, seen_items: list[int] | None = None) -> Any:
    """Keep bounded primitive config metadata while redacting secret-shaped keys."""
    if seen_items is None:
        seen_items = [0]
    require(depth <= MAX_DEPTH, "metadata nesting exceeds limit")
    seen_items[0] += 1
    require(seen_items[0] <= MAX_ITEMS, "metadata item count exceeds limit")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:4096]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if _SECRET_RE.search(key_text):
                continue
            result[key_text[:256]] = safe_metadata(child, depth=depth + 1, seen_items=seen_items)
        return result
    if isinstance(value, (list, tuple)):
        require(len(value) <= MAX_ITEMS, "metadata sequence exceeds limit")
        return [safe_metadata(child, depth=depth + 1, seen_items=seen_items) for child in value]
    return {"unsupported_type": type(value).__name__}


def tensor_metadata(value: Any, tensor_check: Any, path: str = "") -> list[dict[str, Any]]:
    """Describe tensor names, shapes and dtypes without reading tensor values."""
    if tensor_check(value):
        return [
            {
                "path": path,
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
            }
        ]
    found: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(
                tensor_metadata(child, tensor_check, f"{path}.{key}" if path else str(key))
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(tensor_metadata(child, tensor_check, f"{path}[{index}]"))
    return found


def summarize_payload(payload: Any, tensor_check: Any) -> dict[str, Any]:
    """Return scalar checkpoint metadata and a tensor schema, never tensor values."""
    require(isinstance(payload, Mapping), "checkpoint root must be a mapping")
    top_keys = sorted(str(key) for key in payload)
    metadata: dict[str, Any] = {}
    tensor_index: list[dict[str, Any]] = []
    omitted: list[str] = []
    for key, value in payload.items():
        name = str(key)
        if _SECRET_RE.search(name):
            omitted.append(name)
            continue
        if name.lower() in _OMIT_KEYS or tensor_check(value):
            omitted.append(name)
            tensor_index.extend(tensor_metadata(value, tensor_check, name))
        else:
            metadata[name] = safe_metadata(value)
            tensor_index.extend(tensor_metadata(value, tensor_check, name))
    encoded = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()
    require(len(encoded) <= MAX_METADATA_BYTES, "safe metadata output exceeds limit")
    require(len(tensor_index) <= MAX_ITEMS, "tensor schema count exceeds limit")
    return {
        "top_level_keys": top_keys,
        "metadata": metadata,
        "tensor_schema": tensor_index,
        "redacted_or_weight_payload_keys": sorted(set(omitted)),
    }


def _read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Read a small regular manifest and pin the exact bytes used."""
    before = _signature(path)
    require(path.stat().st_size <= MAX_MANIFEST_BYTES, "manifest size bound")
    digest_before = sha256_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(_signature(path) == before, "manifest changed during read")
    require(isinstance(value, dict), "manifest must be an object")
    require(sha256_file(path) == digest_before, "manifest changed after read")
    return value, digest_before


def _load_one(item: dict[str, Any], torch: Any, deadline: float) -> dict[str, Any]:
    """Hash, CPU-load and summarize one exact checkpoint, then discard it."""
    require(time.monotonic() < deadline, "120-second execution deadline reached")
    path = Path(item["path"])
    before = _signature(path)
    require(before[2] == item["expected_bytes"], f"size mismatch: {item['id']}")
    digest_before = sha256_file(path)
    require(_signature(path) == before, f"source changed during initial hash: {item['id']}")
    require(digest_before == item["expected_sha256"], f"source SHA mismatch: {item['id']}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(not torch.cuda.is_initialized(), "CUDA initialized during CPU metadata extraction")
    summary = summarize_payload(payload, torch.is_tensor)
    del payload
    gc.collect()
    require(time.monotonic() < deadline, "120-second execution deadline reached")
    digest_after = sha256_file(path)
    after = _signature(path)
    require(
        after == before and digest_after == digest_before,
        f"source generation changed: {item['id']}",
    )
    return {
        "id": item["id"],
        "path": item["path"],
        "bytes": before[2],
        "sha256_before": digest_before,
        "sha256_after": digest_after,
        "source_generation": {
            "device": before[0],
            "inode": before[1],
            "size": before[2],
            "mtime_ns": before[3],
        },
        **summary,
    }


def _resource_observation() -> dict[str, Any]:
    """Read the enforced Linux cgroup resource limits for this process."""
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    entry = next((line for line in lines if line.startswith("0::")), None)
    require(entry is not None, "unified cgroup path unavailable")
    relative = entry.split("::", maxsplit=1)[1]
    cgroup = Path("/sys/fs/cgroup") / relative.lstrip("/")
    quota, period = (int(value) for value in (cgroup / "cpu.max").read_text().split())
    memory_max = int((cgroup / "memory.max").read_text())
    swap_max = int((cgroup / "memory.swap.max").read_text())
    cpu_cores = quota / period
    require(0 < cpu_cores <= RESOURCE_CONTRACT["cpu_quota_cores"], "CPU cgroup ceiling mismatch")
    require(
        0 < memory_max <= RESOURCE_CONTRACT["memory_max_bytes"], "memory cgroup ceiling mismatch"
    )
    require(swap_max == 0, "swap must be disabled")
    return {
        "cgroup_path": relative,
        "cpu_max": f"{quota} {period}",
        "cpu_quota_cores": cpu_cores,
        "memory_max_bytes": memory_max,
        "memory_swap_max_bytes": swap_max,
        "runtime_seconds": RESOURCE_CONTRACT["runtime_max_seconds"],
        "runtime_enforced_by": "external root-admitted systemd RuntimeMaxSec=120; cgroup verification alone does not enforce elapsed time",
    }


def extract(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Extract metadata from exactly two root-hash-bound fin256 parent checkpoints."""
    require(
        not output_path.exists() and not output_path.is_symlink(), "output receipt already exists"
    )
    manifest, manifest_sha = _read_manifest(manifest_path)
    inputs = validate_manifest(manifest)
    require(output_path.parent.is_dir(), "output parent directory must already exist")
    started = time.monotonic()
    deadline = started + TIME_LIMIT_SECONDS
    resources = _resource_observation()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "CUDA was initialized before extraction")
    records: list[dict[str, Any]] = []
    for item in inputs:
        records.append(_load_one(item, torch, deadline))
    require(time.monotonic() < deadline, "120-second execution deadline reached")
    receipt = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "status": "METADATA_ONLY_NO_WEIGHT_VALUES_EXPORTED",
        "resources": resources,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "torch_version": str(torch.__version__),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "inputs": records,
        "excluded": [
            "tensor values",
            "prediction caches",
            "optimizer state values",
            "training",
            "inference",
        ],
    }
    encoded = json.dumps(receipt, indent=2, allow_nan=False).encode() + b"\n"
    require(len(encoded) <= MAX_OUTPUT_BYTES, "receipt size bound")
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main() -> None:
    """Run one admitted metadata-only extraction; never modify source checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.manifest, args.output)


if __name__ == "__main__":
    main()
