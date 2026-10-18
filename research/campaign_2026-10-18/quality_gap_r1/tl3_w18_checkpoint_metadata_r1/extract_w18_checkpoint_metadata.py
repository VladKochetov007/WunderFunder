#!/usr/bin/env python3
"""Read safe metadata from eight pinned W18 checkpoints without exporting weights."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import stat
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from w18_metadata_core import (
    MAX_MANIFEST_BYTES,
    MAX_OUTPUT_BYTES,
    RESOURCE_CONTRACT,
    SCHEMA,
    SOURCE_OBSERVATION,
    TIME_LIMIT_SECONDS,
    _fd_sha256,
    _path_resolution,
    _read_manifest,
    _sha256_file,
    _stat_signature,
    _verify_admission,
    require,
    validate_manifest,
)
from w18_metadata_summary import summarize_payload


# Any is necessary for values returned by Torch's restricted checkpoint loader.
def _resource_observation() -> dict[str, Any]:
    """Verify this process runs under the enforced CPU, memory and swap limits."""
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
    require(swap_max == RESOURCE_CONTRACT["swap_max_bytes"], "swap must be disabled")
    return {
        "cgroup_path": relative,
        "cpu_quota_cores": cpu_cores,
        "memory_max_bytes": memory_max,
        "swap_max_bytes": swap_max,
        "runtime_max_seconds": TIME_LIMIT_SECONDS,
        "runtime_enforced_by": "root-admitted external systemd RuntimeMaxSec=120",
    }


def _load_one(item: dict[str, Any], torch: Any, deadline: float) -> dict[str, Any]:
    """Hash, restricted-load and summarize one exact checkpoint through a pinned file descriptor."""
    require(time.monotonic() < deadline, "120-second execution deadline reached")
    logical = Path(item["logical_path"])
    expected_resolution = item["source_resolution"]
    current_resolution = _path_resolution(logical)
    require(
        current_resolution == expected_resolution, f"path/symlink resolution changed: {item['id']}"
    )
    resolved = Path(current_resolution["resolved_path"])
    path_info = resolved.lstat()
    require(stat.S_ISREG(path_info.st_mode), f"resolved target is not a regular file: {resolved}")
    expected_generation = item["expected_generation"]
    require(
        _stat_signature(path_info) == expected_generation,
        f"resolved target generation mismatch: {item['id']}",
    )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    require(nofollow != 0, "O_NOFOLLOW is required for checkpoint inputs")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    fd = os.open(resolved, flags)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), f"opened target is not regular: {item['id']}")
        require(
            _stat_signature(before) == expected_generation,
            f"opened generation mismatch: {item['id']}",
        )
        digest_before = _fd_sha256(fd)
        require(
            _stat_signature(os.fstat(fd)) == expected_generation,
            f"source changed during hash: {item['id']}",
        )
        require(digest_before == item["expected_sha256"], f"source SHA mismatch: {item['id']}")
        require(time.monotonic() < deadline, "120-second execution deadline reached")
        with os.fdopen(os.dup(fd), "rb") as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        require(not torch.cuda.is_initialized(), "CUDA initialized during CPU metadata extraction")
        summary = summarize_payload(payload, torch.is_tensor)
        del payload
        gc.collect()
        require(time.monotonic() < deadline, "120-second execution deadline reached")
        digest_after = _fd_sha256(fd)
        after = os.fstat(fd)
        require(
            _stat_signature(after) == expected_generation,
            f"source generation changed: {item['id']}",
        )
        require(digest_after == digest_before, f"source hash changed after load: {item['id']}")
        require(
            _path_resolution(logical) == expected_resolution,
            f"path/symlink resolution changed after load: {item['id']}",
        )
        require(
            _stat_signature(resolved.lstat()) == expected_generation,
            f"resolved path generation changed: {item['id']}",
        )
        return {
            "id": item["id"],
            "logical_path": str(logical),
            "source_resolution": current_resolution,
            "bytes": before.st_size,
            "sha256_before": digest_before,
            "sha256_after": digest_after,
            "source_generation": expected_generation,
            **summary,
        }
    finally:
        os.close(fd)


def extract(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Extract metadata from exactly eight root-hash-bound W18 checkpoints."""
    require(
        not output_path.exists() and not output_path.is_symlink(), "output receipt already exists"
    )
    require(output_path.parent.is_dir(), "output parent directory must already exist")
    started = time.monotonic()
    deadline = started + TIME_LIMIT_SECONDS
    manifest, manifest_sha = _read_manifest(manifest_path)
    inputs, admission = validate_manifest(manifest)
    _verify_admission(admission)
    resources = _resource_observation()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "CUDA was initialized before extraction")
    records = [_load_one(item, torch, deadline) for item in inputs]
    manifest_sha_after, _ = _sha256_file(manifest_path, MAX_MANIFEST_BYTES)
    require(manifest_sha_after == manifest_sha, "input manifest changed during extraction")
    _verify_admission(admission)
    require(time.monotonic() < deadline, "120-second execution deadline reached")
    receipt = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "METADATA_ONLY_NO_WEIGHT_VALUES_EXPORTED",
        "input_manifest_path": str(manifest_path),
        "input_manifest_sha256": manifest_sha,
        "root_admission_receipt_path": admission["path"],
        "root_admission_receipt_sha256": admission["sha256"],
        "source_observation": SOURCE_OBSERVATION,
        "resources": resources,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "torch_version": str(torch.__version__),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "inputs": records,
        "excluded": ["tensor values", "prediction or propensity arrays", "training", "inference"],
    }
    encoded = json.dumps(receipt, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    require(len(encoded) <= MAX_OUTPUT_BYTES, "receipt size bound")
    descriptor = os.open(
        output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main() -> None:
    """Run one admitted metadata-only extraction and emit a compact capture binding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = extract(args.manifest, args.output)
    output_bytes = args.output.read_bytes()
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output_path": str(args.output),
                "output_bytes": len(output_bytes),
                "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
                "input_manifest_sha256": receipt["input_manifest_sha256"],
                "root_admission_receipt_sha256": receipt["root_admission_receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
