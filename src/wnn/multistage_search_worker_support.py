from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_gpu_identity import normalize_gpu_uuid
from wnn.multistage_search import atomic_write_json, file_sha256, read_json
from wnn.multistage_search_budget import utc_now


def _device_spec(admission: Mapping[str, object], gpu_key: str) -> Mapping[str, object]:
    (policy,) = (admission.get("profile_policy"),)
    if not isinstance(policy, Mapping) or not isinstance(policy.get("devices"), Mapping):
        raise ValueError("admission does not map an owned worker to a measured GPU")
    devices = policy["devices"]
    if gpu_key not in {str(key) for key in devices}:
        raise ValueError("requested GPU is outside the independently profiled device set")
    spec = next(value for key, value in devices.items() if str(key) == gpu_key)
    if not isinstance(spec, Mapping) or not isinstance(spec.get("cuda_visible_devices"), str):
        raise ValueError("admitted GPU device specification is incomplete")
    return spec


def _nvml_memory_snapshot(device_selector: str) -> dict[str, object]:
    """Read available device memory before a profile without touching MPS or processes."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                device_selector,
                "--query-gpu=uuid,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 3:
        return {"status": "invalid_response"}
    try:
        free_mib, total_mib = int(rows[0][1].strip()), int(rows[0][2].strip())
    except ValueError:
        return {"status": "invalid_response"}
    return {
        "status": "observed",
        "device_uuid": rows[0][0].strip(),
        "free_memory_bytes": free_mib * (1 << 20),
        "total_memory_bytes": total_mib * (1 << 20),
        "observed_at_utc": utc_now(),
    }


def _validate_profile_output(
    profile_path: Path,
    output_dir: Path,
    requested_updates: int,
    request: Mapping[str, object],
    source_sha256: object,
) -> bool:
    """Check the exact-profile native receipt and one per-update rank log before review."""
    if not profile_path.is_file():
        return False
    profile = read_json(profile_path)
    identity = profile.get("identity")
    profile_data = identity.get("data") if isinstance(identity, Mapping) else None
    if (
        profile.get("receipt_schema") != "wnn.sequence-profile.v1"
        or profile.get("status") != "profile_complete"
        or profile.get("approved") is not False
        or not isinstance(identity, Mapping)
        or identity.get("source_sha256") != source_sha256
        or not isinstance(profile_data, Mapping)
        or profile_data.get("split_sha256") != request.get("auxiliary_split_sha256")
        or profile.get("updates") != requested_updates
    ):
        return False
    runtime = identity.get("runtime")
    devices = runtime.get("devices") if isinstance(runtime, Mapping) else None
    if not isinstance(devices, list) or len(devices) != 1 or not isinstance(devices[0], Mapping):
        return False
    expected_uuids = request.get("expected_device_uuid_set")
    if (
        not isinstance(expected_uuids, list)
        or len(expected_uuids) != 1
        or normalize_gpu_uuid(devices[0].get("uuid")) != normalize_gpu_uuid(expected_uuids[0])
    ):
        return False
    torch_total = request.get("expected_device_torch_total_memory_bytes")
    if type(torch_total) is not int or devices[0].get("total_memory_bytes") != torch_total:
        return False
    refs = _rank_log_refs(output_dir)
    return len(refs) == request.get("requested_world_size") == 1 and bool(refs)


def _device_uuid(device: Mapping[str, object]) -> str:
    """Return a measured immutable hardware UUID from the admission inventory."""
    value = device.get("uuid", device.get("device_uuid"))
    if not isinstance(value, str) or not value:
        raise ValueError("admitted GPU needs its measured UUID")
    return value


def _find_anchor(anchors: object, anchor_id: str) -> Mapping[str, object]:
    """Find one exact reviewed anchor by its stable ID."""
    if not isinstance(anchors, list):
        raise ValueError("stage1 anchor inventory is malformed")
    matches = [
        item
        for item in anchors
        if isinstance(item, Mapping) and item.get("id", item.get("anchor_id")) == anchor_id
    ]
    if len(matches) != 1:
        raise ValueError("queued anchor id is absent or duplicated in the admission")
    return matches[0]


def _device_total_memory(device: Mapping[str, object]) -> int:
    """Return measured total device memory in bytes."""
    value = device.get("total_memory_bytes", device.get("device_total_memory_bytes"))
    if type(value) is not int or value <= 0:
        raise ValueError("admitted GPU needs measured total memory in bytes")
    return value


def _device_torch_total_memory(device: Mapping[str, object]) -> int:
    """Return the separately measured CUDA runtime memory total in bytes."""
    value = device.get("torch_total_memory_bytes", device.get("torch_runtime_total_memory_bytes"))
    if type(value) is not int or value <= 0:
        raise ValueError("admitted GPU needs a measured Torch runtime memory total")
    return value


def _process_tuple(value: object) -> dict[str, object]:
    """Extract a complete PID/birth/PGID tuple from an owned-process receipt."""
    if not isinstance(value, Mapping):
        raise ValueError("owned profile process receipt is missing")
    pid, birth, pgid = value.get("pid"), value.get("birth"), value.get("pgid")
    if type(pid) is not int or not isinstance(birth, str) or type(pgid) is not int:
        raise ValueError("owned profile process identity is incomplete")
    return {"pid": pid, "birth": birth, "pgid": pgid}


def _rank_log_refs(output_dir: Path) -> list[dict[str, object]]:
    refs: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("rank*.jsonl")):
        name = path.stem.removeprefix("rank")
        try:
            rank = int(name)
        except ValueError:
            continue
        refs.append({"rank": rank, "path": str(path.resolve()), "sha256": file_sha256(path)})
    return refs


def _failed_launch_receipt(
    path: Path, owner: Mapping[str, object], command: list[str], output: Path, error: Exception
) -> dict[str, object]:
    receipt = {
        "schema": "wnn.multistage-owned-process.v1",
        "owner": dict(owner),
        "child": None,
        "command": command,
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "output_dir": str(output.resolve()),
        "status": "launch_failed",
        "error": f"{type(error).__name__}: {error}",
        "started_at_utc": utc_now(),
        "started_at_epoch": __import__("time").time(),
        "finished_at_utc": utc_now(),
        "finished_at_epoch": __import__("time").time(),
    }
    atomic_write_json(path, receipt, replace=False)
    return receipt
