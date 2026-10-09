"""Fail-closed, profile-only E520 Kaggle worker.

This module never has a fit or prediction-export mode. It performs identity and
fold preflight, invokes the pinned trainer's bounded early-return profile, and
measures one real-input inference chunk without persisting model outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import shutil
import signal
import sys
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

PACKET_ID = "E520_PROFILE_V1"
PROFILE_WARMUP = 3
PROFILE_STEPS = 20
PROFILE_WALL_CAP_S = 1_800
SPLIT_RELATIVE = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
PORTABLE_RELATIVE = Path("research/campaign_2026-10-04/portable_inputs.json")
EXPECTED = {
    "train": {
        "sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
        "bytes": 29_240_219_011,
        "footer_sha256": "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb",
        "footer_bytes": 60_437_285,
    },
    "valid": {
        "sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
        "bytes": 5_615_509_128,
        "footer_sha256": "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b",
        "footer_bytes": 10_686_926,
    },
}
ASSETS_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
ASSETS_BYTES = 6_268
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PORTABLE_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
TARGET_HASHES = {
    "train": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
    "valid": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
}
ASSET_HASHES = {
    "cache/chain_folds.npz": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
    "cache/chains.npz": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766",
    "cache/train_targets.npz": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
    "cache/valid_targets.npz": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
    "research/campaign_2026-09-29/artifacts/splits/fold0.json": SPLIT_SHA256,
    "research/campaign_2026-10-04/portable_inputs.json": PORTABLE_SHA256,
}
TARGET_COUNTS = {"train": 10_606, "valid": 1_873}
FIT_COUNTS = {"train": 10_535, "valid": 1_405, "evaluation": 468}
CORRUPT_TRAIN_GROUP = 528
CAPS = {
    "rss_bytes": 8_000_000_000,
    "cuda_allocated_bytes": 2_000_000_000,
    "cuda_reserved_bytes": 2_500_000_000,
    "gpu_process_bytes": 3_000_000_000,
    "headroom_bytes": 4_294_967_296,
    "headroom_fraction": 0.20,
    "disk_free_bytes": 8_000_000_000,
    "visible_gpus": 1,
}


class ProfileAbort(RuntimeError):
    """Raised when a preflight, external-review, or live resource gate fails."""


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a small JSON receipt after writing its staging file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def require_profile_only(argv: list[str]) -> None:
    """Reject any command that could fit, save weights, or export predictions."""
    forbidden = {
        "--save-training-state",
        "--resume-state",
        "--parity-dump",
        "--checkpoint-path",
        "--prediction-output",
        "--export-predictions",
    }
    if any(item in forbidden for item in argv):
        raise ProfileAbort("profile-only command contains a fit/save/export option")
    pairs = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    if pairs.get("--profile-steps") != str(PROFILE_STEPS):
        raise ProfileAbort("profile must request exactly 20 timed outer batches")
    if pairs.get("--profile-warmup") != str(PROFILE_WARMUP):
        raise ProfileAbort("profile must request exactly 3 warmup batches")


def validate_external_review(
    receipt_bytes: bytes, expected_sha256: str, actor_id: str
) -> dict[str, Any]:
    """Require a controller-supplied, externally hashed approval receipt."""
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ProfileAbort("independent review hash was not supplied by the controller")
    observed = hashlib.sha256(receipt_bytes).hexdigest()
    if observed != expected_sha256:
        raise ProfileAbort("external review receipt hash mismatch")
    receipt = json.loads(receipt_bytes)
    reviewer = receipt.get("reviewer", {})
    if (
        receipt.get("experiment_id") != PACKET_ID
        or receipt.get("scope") != "profile-only"
        or receipt.get("status") != "approved"
        or not reviewer.get("id")
        or reviewer.get("id") == actor_id
        or reviewer.get("role") != "independent-controller"
    ):
        raise ProfileAbort("external review receipt is missing independent profile-only approval")
    hashes = receipt.get("reviewed_hashes", {})
    required = {
        "profile_worker.py",
        "profile_kernel.py",
        "prepare_profile_packet.py",
        "export_guards.py",
        "tests/test_profile_guards.py",
        "E520_PACKET_PLAN.json",
        "source_archive.tar",
    }
    if not required.issubset(hashes):
        raise ProfileAbort("external review does not bind all executable/source inputs")
    return receipt


def percentile_summary(values: list[float]) -> dict[str, float | int]:
    """Return deterministic nearest-rank p50/p90 and count for finite samples."""
    cleaned = sorted(value for value in values if math.isfinite(value))
    if not cleaned:
        raise ProfileAbort("no finite timing samples were recorded")

    def nearest_rank(q: float) -> float:
        return cleaned[max(0, math.ceil(q * len(cleaned)) - 1)]

    return {"count": len(cleaned), "p50": nearest_rank(0.50), "p90": nearest_rank(0.90)}


def validate_export_shape(batch: int, rows: int, features: int, outputs: int) -> dict[str, int]:
    """Validate the fixed probe and account for input/output bytes without allocating them."""
    if (batch, rows, outputs) != (16, 4_000, 2) or features <= 0:
        raise ProfileAbort("exporter probe must use batch=16, chunk_rows=4000, outputs=2")
    return {
        "input_float32_bytes": batch * rows * features * 4,
        "output_float32_bytes_if_written": batch * rows * outputs * 4,
    }


def validate_export_tensors(
    raw_shape: tuple[int, ...],
    model_input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    *,
    raw_features: int,
    model_features: int,
    raw_dtype: str,
    model_input_dtype: str,
    output_dtype: str,
    outputs_finite: bool,
) -> None:
    """Fail unless the real export probe matches the exact shape/dtype/value contract."""
    if raw_shape != (16, 4_000, raw_features):
        raise ProfileAbort(f"raw exporter input shape mismatch: {raw_shape}")
    if model_input_shape != (16, 4_000, model_features):
        raise ProfileAbort(f"model exporter input shape mismatch: {model_input_shape}")
    if output_shape != (16, 4_000, 2):
        raise ProfileAbort(f"exporter output shape mismatch: {output_shape}")
    if raw_dtype != "torch.float32" or model_input_dtype != "torch.float32":
        raise ProfileAbort("exporter raw/model input dtype must remain float32")
    if output_dtype != "torch.float32":
        raise ProfileAbort("exporter two-target output dtype must be float32")
    if not outputs_finite:
        raise ProfileAbort("exporter output contains a non-finite value")


def _read_meminfo() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, tail = line.partition(":")
        field = tail.strip().split()
        if field and field[0].isdigit():
            values[key] = int(field[0]) * (1024 if len(field) > 1 and field[1] == "kB" else 1)
    return values.get("MemAvailable", 0), values.get("MemTotal", 0)


def _cgroup_headroom() -> tuple[int, int]:
    mount = Path("/sys/fs/cgroup")
    candidates = [mount]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, _, relative = line.partition(":")
            if hierarchy == "0" and relative:
                current = (mount / relative.lstrip("/")).resolve()
                while current == mount or mount in current.parents:
                    candidates.append(current)
                    if current == mount:
                        break
                    current = current.parent
                break
    except OSError:
        pass
    observations: list[tuple[int, int]] = []
    for root in dict.fromkeys(candidates):
        current_path, max_path = root / "memory.current", root / "memory.max"
        if current_path.exists() and max_path.exists():
            current = int(current_path.read_text().strip())
            raw_max = max_path.read_text().strip()
            if raw_max != "max":
                maximum = int(raw_max)
                observations.append((max(0, maximum - current), maximum))
    if observations:
        return min(observations, key=lambda item: item[0])
    return 0, 0


def _rss_bytes() -> tuple[int, int]:
    current = 0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            current = int(line.split()[1]) * 1024
            break
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    return current, peak


def resource_snapshot(torch_module: Any | None, working: Path) -> dict[str, Any]:
    """Collect current RSS, host/cgroup/disk and process-local CUDA allocator facts."""
    rss_current, rss_peak = _rss_bytes()
    host_available, host_total = _read_meminfo()
    cgroup_available, cgroup_limit = _cgroup_headroom()
    usage = shutil.disk_usage(working)
    result: dict[str, Any] = {
        "rss_current_bytes": rss_current,
        "rss_peak_bytes": rss_peak,
        "host_available_bytes": host_available,
        "host_total_bytes": host_total,
        "host_available_fraction": host_available / host_total if host_total else 0.0,
        "cgroup_available_bytes": cgroup_available,
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_available_fraction": cgroup_available / cgroup_limit if cgroup_limit else 0.0,
        "disk_free_bytes": usage.free,
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_allocated_peak_bytes": None,
        "cuda_reserved_peak_bytes": None,
        "gpu_process_bytes": None,
        "gpu_process_query_status": "pre_cuda_idle",
        "gpu_process_query_raw": None,
        "gpu_free_bytes": None,
    }
    if torch_module is not None and torch_module.cuda.is_available():
        process_measurement = _nvidia_process_measurement(os.getpid())
        result.update(
            {
                "cuda_allocated_bytes": int(torch_module.cuda.memory_allocated(0)),
                "cuda_reserved_bytes": int(torch_module.cuda.memory_reserved(0)),
                "cuda_allocated_peak_bytes": int(torch_module.cuda.max_memory_allocated(0)),
                "cuda_reserved_peak_bytes": int(torch_module.cuda.max_memory_reserved(0)),
                "gpu_process_bytes": process_measurement["bytes"],
                "gpu_process_query_status": process_measurement["status"],
                "gpu_process_query_raw": process_measurement["raw"],
                "gpu_free_bytes": int(torch_module.cuda.mem_get_info(0)[0]),
            }
        )
    return result


def parse_nvidia_process_bytes(pid: int, output: str) -> int | None:
    """Parse a process allocation; no matching row means unavailable, never zero."""
    import csv
    import io

    matches = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if (
            len(row) >= 2
            and row[0].strip().isdigit()
            and int(row[0]) == pid
            and row[1].strip().isdigit()
        ):
            matches.append(int(row[1]) * 1024 * 1024)
    return sum(matches) if matches else None


def _nvidia_process_measurement(pid: int) -> dict[str, Any]:
    """Read and retain the raw nvidia-smi process query for diagnostics."""
    import subprocess

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"bytes": None, "status": "query_error", "raw": f"{type(error).__name__}: {error}"}
    used = parse_nvidia_process_bytes(pid, completed.stdout)
    return {
        "bytes": used,
        "status": "matched_pid" if used is not None else "pid_not_reported",
        "raw": completed.stdout[:4_096],
    }


def runtime_identity(torch_module: Any) -> dict[str, Any]:
    """Capture actual selected-GPU, Torch, CUDA, cuDNN, driver, and Python identity."""
    import subprocess

    properties = torch_module.cuda.get_device_properties(0)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        nvidia_smi = {"status": "ok", "rows": completed.stdout[:4_096]}
    except (OSError, subprocess.SubprocessError) as error:
        nvidia_smi = {"status": "query_error", "diagnostic": f"{type(error).__name__}: {error}"}
    return {
        "visible_gpu_count": int(torch_module.cuda.device_count()),
        "selected_gpu_name": str(torch_module.cuda.get_device_name(0)),
        "selected_gpu_total_bytes": int(properties.total_memory),
        "selected_gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "torch_version": str(torch_module.__version__),
        "cuda_runtime_version": torch_module.version.cuda,
        "cudnn_version": torch_module.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "nvidia_smi_identity": nvidia_smi,
    }


def enforce_resource_caps(snapshot: dict[str, Any], elapsed_s: float) -> None:
    """Raise immediately on a hard worker limit or missing resource observation."""
    if elapsed_s > PROFILE_WALL_CAP_S:
        raise ProfileAbort("profile worker exceeded its 1,800-second engineering wall cap")
    if snapshot["rss_peak_bytes"] > CAPS["rss_bytes"]:
        raise ProfileAbort("worker RSS high-water exceeded 8,000,000,000 bytes")
    for field, cap in (
        ("cuda_allocated_bytes", CAPS["cuda_allocated_bytes"]),
        ("cuda_reserved_bytes", CAPS["cuda_reserved_bytes"]),
        ("cuda_allocated_peak_bytes", CAPS["cuda_allocated_bytes"]),
        ("cuda_reserved_peak_bytes", CAPS["cuda_reserved_bytes"]),
        ("gpu_process_bytes", CAPS["gpu_process_bytes"]),
    ):
        value = snapshot[field]
        if value is not None and value > cap:
            raise ProfileAbort(f"{field} exceeded its E520 profile cap")
    for prefix in ("host", "cgroup"):
        available = snapshot[f"{prefix}_available_bytes"]
        fraction = snapshot[f"{prefix}_available_fraction"]
        if available < CAPS["headroom_bytes"] or fraction < CAPS["headroom_fraction"]:
            raise ProfileAbort(f"{prefix} available memory fell below both E520 headroom gates")
    if snapshot["disk_free_bytes"] < CAPS["disk_free_bytes"]:
        raise ProfileAbort("working disk fell below 8,000,000,000 free bytes")
    if (
        snapshot.get("gpu_process_bytes") is None
        and snapshot.get("gpu_process_query_status") != "pre_cuda_idle"
    ):
        raise ProfileAbort("GPU process allocation is unavailable outside explicit pre-CUDA idle")
    if snapshot.get("cuda_allocated_bytes") is not None and any(
        snapshot.get(field) is None
        for field in ("cuda_allocated_peak_bytes", "cuda_reserved_peak_bytes")
    ):
        raise ProfileAbort("CUDA allocator peak APIs are unavailable during active profiling")
    if snapshot.get("cuda_allocated_bytes") is not None:
        if snapshot.get("gpu_process_bytes") is None:
            raise ProfileAbort("could not measure this process's actual GPU allocation")
        if snapshot.get("gpu_free_bytes", 0) < 6 * 1024**3:
            raise ProfileAbort("selected GPU free memory fell below 6 GiB")


def parquet_footer(path: Path) -> dict[str, Any]:
    """Stream-hash only a Parquet footer and trailer, then report size and digest."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        footer_bytes = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or footer_bytes > size - 8:
            raise ProfileAbort(f"invalid Parquet trailer: {path.name}")
        count = footer_bytes + 8
        stream.seek(-count, os.SEEK_END)
        digest = hashlib.sha256()
        remaining = count
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ProfileAbort(f"truncated Parquet footer: {path.name}")
            digest.update(block)
            remaining -= len(block)
    return {"bytes": size, "footer_bytes": footer_bytes, "footer_sha256": digest.hexdigest()}


def _unique(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise ProfileAbort(f"expected one mounted {name}, found {len(matches)}")
    return matches[0]


def persist_identity_check(
    receipt_path: Path,
    identity: dict[str, Any],
    logical_name: str,
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> None:
    """Atomically persist both sides of an identity check before raising mismatch."""
    identity.setdefault("expected", {})[logical_name] = expected
    identity.setdefault("observed", {})[logical_name] = observed
    atomic_json(receipt_path, identity)
    mismatches = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatches:
        raise ProfileAbort(f"identity mismatch for {logical_name}: fields={','.join(mismatches)}")


def _index_digest(groups: Any, seq_ix: Any) -> str:
    digest = hashlib.sha256()
    digest.update(groups.astype("<i8", copy=False).tobytes())
    digest.update(seq_ix.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def verify_and_stage_inputs(
    input_root: Path, source_root: Path, receipt_path: Path
) -> dict[str, Any]:
    """Record expected/observed mount identities, stage symlinks, and validate fold membership."""
    identity: dict[str, Any] = {
        "phase": "input_identity",
        "status": "in_progress",
        "expected": {
            "datasets/train.parquet": EXPECTED["train"],
            "datasets/valid.parquet": EXPECTED["valid"],
            "assets-manifest.json": {"sha256": ASSETS_SHA256, "bytes": ASSETS_BYTES},
            **{relative: {"sha256": digest} for relative, digest in ASSET_HASHES.items()},
        },
        "observed": {},
        "mount_version": None,
    }
    atomic_json(receipt_path, identity)
    try:
        assets_path = _unique(input_root, "assets-manifest.json")
    except ProfileAbort as error:
        persist_identity_check(
            receipt_path,
            identity,
            "assets-manifest.json",
            {"sha256": ASSETS_SHA256, "bytes": ASSETS_BYTES},
            {"status": "unavailable", "diagnostic": str(error)},
        )
        raise
    manifest_observed = {"bytes": assets_path.stat().st_size, "sha256": sha256_file(assets_path)}
    persist_identity_check(
        receipt_path,
        identity,
        "assets-manifest.json",
        {"sha256": ASSETS_SHA256, "bytes": ASSETS_BYTES},
        manifest_observed,
    )
    assets = json.loads(assets_path.read_text())
    if assets.get("dataset") != "vladkochetov07/wnn-c20261004-fold0-assets":
        persist_identity_check(
            receipt_path,
            identity,
            "assets-manifest.json.dataset",
            {"dataset": "vladkochetov07/wnn-c20261004-fold0-assets"},
            {"dataset": assets.get("dataset")},
        )
        raise ProfileAbort("fold assets manifest dataset identity mismatch")
    by_path = {entry["path"]: entry for entry in assets.get("files", [])}
    required = [
        "cache/chain_folds.npz",
        "cache/chains.npz",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
        SPLIT_RELATIVE.as_posix(),
        PORTABLE_RELATIVE.as_posix(),
    ]
    staged: dict[str, Path] = {}
    for relative in required:
        entry = by_path.get(relative)
        if not isinstance(entry, dict):
            persist_identity_check(
                receipt_path,
                identity,
                relative,
                {"sha256": ASSET_HASHES[relative]},
                {"status": "unavailable", "diagnostic": "asset missing from verified manifest"},
            )
        if "filename" not in entry or "size" not in entry:
            persist_identity_check(
                receipt_path,
                identity,
                relative,
                {"sha256": ASSET_HASHES[relative]},
                {"status": "malformed", "manifest_entry": entry},
            )
        expected_asset = {
            "sha256": ASSET_HASHES[relative],
            "bytes": int(entry["size"]),
        }
        try:
            path = _unique(input_root, str(entry["filename"]))
        except ProfileAbort as error:
            persist_identity_check(
                receipt_path,
                identity,
                relative,
                expected_asset,
                {"status": "unavailable", "diagnostic": str(error)},
            )
            raise
        observed_asset = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        persist_identity_check(receipt_path, identity, relative, expected_asset, observed_asset)
        staged[relative] = path
    portable = json.loads(staged[PORTABLE_RELATIVE.as_posix()].read_text())
    raw: dict[str, Path] = {}
    for split_name in ("train", "valid"):
        logical_name = f"datasets/{split_name}.parquet"
        expected = EXPECTED[split_name]
        try:
            path = _unique(input_root, f"{split_name}.parquet")
        except ProfileAbort as error:
            persist_identity_check(
                receipt_path,
                identity,
                logical_name,
                expected,
                {"status": "unavailable", "diagnostic": str(error)},
            )
            raise
        observed: dict[str, Any] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        identity.setdefault("observed", {})[logical_name] = observed
        atomic_json(receipt_path, identity)
        try:
            observed.update(parquet_footer(path))
        except ProfileAbort as error:
            observed["footer_status"] = "invalid"
            observed["footer_diagnostic"] = str(error)
            atomic_json(receipt_path, identity)
            raise
        persist_identity_check(receipt_path, identity, logical_name, expected, observed)
        raw[split_name] = path
    for logical, path in raw.items():
        destination = source_root / "datasets" / f"{logical}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise ProfileAbort(f"refusing to replace staged input {destination}")
        destination.symlink_to(path)
    for relative, path in staged.items():
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise ProfileAbort(f"refusing to replace staged asset {destination}")
        destination.symlink_to(path)
    sys.path.insert(0, str(source_root / "src"))
    from wnn.splits import load_campaign_split

    split = load_campaign_split(source_root / SPLIT_RELATIVE, source_root)
    train, valid, evaluation = (
        set(map(int, split.train_groups)),
        set(map(int, split.valid_groups)),
        set(map(int, split.evaluation_groups)),
    )
    if (len(train), len(valid), len(evaluation)) != (
        FIT_COUNTS["train"],
        FIT_COUNTS["valid"],
        FIT_COUNTS["evaluation"],
    ):
        raise ProfileAbort("fold-0 exact fit/evaluation membership counts changed")
    if CORRUPT_TRAIN_GROUP in train:
        raise ProfileAbort("corrupt train row group 528 entered the fit membership")
    if split.sha256 != SPLIT_SHA256:
        raise ProfileAbort("loaded split fingerprint differs from E508-v5")
    orders: dict[str, list[int]] = {}
    import numpy as np

    for split_name in ("train", "valid"):
        with np.load(staged[f"cache/{split_name}_targets.npz"], allow_pickle=False) as cache:
            groups = np.asarray(cache["row_group"], dtype=np.int64)
            seq_ix = np.asarray(cache["seq_ix"], dtype=np.int64)
        if (
            groups.shape != (TARGET_COUNTS[split_name],)
            or len(set(map(int, groups))) != TARGET_COUNTS[split_name]
        ):
            raise ProfileAbort(f"{split_name} target row-group order dimensions changed")
        expected_index_hash = portable["sources"][f"cache/{split_name}_targets.npz"][
            "original_identity"
        ]["index_sha256"]
        observed_index_hash = _index_digest(groups, seq_ix)
        persist_identity_check(
            receipt_path,
            identity,
            f"cache/{split_name}_targets.npz.index",
            {"index_sha256": expected_index_hash},
            {"index_sha256": observed_index_hash},
        )
        target_hash = sha256_file(staged[f"cache/{split_name}_targets.npz"])
        persist_identity_check(
            receipt_path,
            identity,
            f"cache/{split_name}_targets.npz.v5_hash",
            {"sha256": TARGET_HASHES[split_name]},
            {"sha256": target_hash},
        )
        eligible = train if split_name == "train" else valid
        if not eligible.issubset(set(map(int, groups))):
            raise ProfileAbort(f"{split_name} target order omits an eligible fit group")
        orders[split_name] = list(map(int, groups[:16]))
    if CORRUPT_TRAIN_GROUP in set(
        map(int, np.load(staged["cache/train_targets.npz"], allow_pickle=False)["row_group"])
    ):
        raise ProfileAbort("corrupt train row group entered target-cache order")
    identity.update(
        {
            "status": "matched",
            "split_sha256": split.sha256,
            "fit_membership_counts": {"train": len(train), "valid": len(valid)},
            "evaluation_group_count": len(evaluation),
            "target_counts": TARGET_COUNTS,
            "target_index_hashes": {
                name: json.loads(staged[PORTABLE_RELATIVE.as_posix()].read_text())["sources"][
                    f"cache/{name}_targets.npz"
                ]["original_identity"]["index_sha256"]
                for name in orders
            },
            "dataset_version_from_runtime_mount": None,
        }
    )
    atomic_json(receipt_path, identity)
    return {"identity": identity, "orders": orders, "split": split}


class LiveGuard:
    """Poll hard resource limits while the trainer or chunk probe is active."""

    def __init__(
        self, torch_module: Any | None, working: Path, receipts: Path, start: float
    ) -> None:
        self.torch = torch_module
        self.working = working
        self.receipts = receipts
        self.start = start
        self.stop = threading.Event()
        self.failure: str | None = None
        self.latest: dict[str, Any] = {}
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def check(self) -> dict[str, Any]:
        snap = resource_snapshot(self.torch, self.working)
        self.latest = snap
        enforce_resource_caps(snap, time.monotonic() - self.start)
        return snap

    def _poll(self) -> None:
        while not self.stop.wait(1.0):
            try:
                snap = self.check()
                atomic_json(
                    self.receipts / "resource.json",
                    {"phase": "profile_active", "time_unix": time.time(), "snapshot": snap},
                )
            except (
                Exception
            ) as error:  # the main thread cannot safely continue after a gate failure
                self.failure = str(error)
                atomic_json(
                    self.receipts / "resource_abort.json",
                    {
                        "status": "aborted",
                        "reason": self.failure,
                        "snapshot": self.latest,
                        "time_unix": time.time(),
                    },
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def start_polling(self) -> None:
        self.thread.start()

    def stop_polling(self) -> None:
        self.stop.set()
        self.thread.join(timeout=3)

    def raise_if_failed(self) -> None:
        if self.failure:
            raise ProfileAbort(self.failure)


def run_profile(packet_root: Path) -> dict[str, Any]:
    """Run mounted-data preflight, frozen trainer profile, and real-input 4,000-row probe."""
    work = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
    receipts = work / "e520_profile"
    receipts.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    phase: dict[str, Any] = {
        "packet": PACKET_ID,
        "status": "active",
        "phase": "review_gate",
        "start_unix": time.time(),
    }
    phase_path = receipts / "phase.json"
    atomic_json(phase_path, phase)
    guard: LiveGuard | None = None
    identity_path = receipts / "input_identity.json"

    def handle_termination(signum: int, frame: Any) -> None:
        phase.update(
            {
                "phase": "failed",
                "status": "aborted",
                "reason": "SIGTERM resource/runtime gate",
                "signal": signum,
                "updated_unix": time.time(),
                "wall_seconds": time.monotonic() - start,
            }
        )
        atomic_json(phase_path, phase)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_termination)
    try:
        if not work.is_dir():
            raise ProfileAbort(f"working directory is missing: {work}")
        guard = LiveGuard(None, work, receipts, start)
        guard.start_polling()
        meta = json.loads((packet_root / "packet_meta.json").read_text())
        receipt_bytes = (packet_root / "external_review.json").read_bytes()
        receipt = validate_external_review(
            receipt_bytes, meta.get("external_review_sha256", ""), "weak_diversity_kaggle_actor"
        )
        if receipt.get("reviewed_hashes") != meta.get("reviewed_hashes"):
            raise ProfileAbort("controller receipt and packet review hash map differ")
        for name in (
            "profile_worker.py",
            "profile_kernel.py",
            "prepare_profile_packet.py",
            "export_guards.py",
            "tests/test_profile_guards.py",
            "E520_PACKET_PLAN.json",
        ):
            actual_hash = sha256_file(packet_root / name)
            if actual_hash != meta["reviewed_hashes"].get(name):
                raise ProfileAbort(f"reviewed code hash mismatch: {name}")
        source_tar = packet_root / "source_archive.tar"
        if sha256_file(source_tar) != meta.get("source_archive_sha256"):
            raise ProfileAbort("pinned source archive hash mismatch")
        if sha256_file(source_tar) != meta["reviewed_hashes"].get("source_archive.tar"):
            raise ProfileAbort("source archive is not the reviewed archive")
        source_root = work / "e520_source"
        source_root.mkdir(parents=True, exist_ok=False)
        with tarfile.open(source_tar, "r:") as archive:
            members = archive.getmembers()
            for member in members:
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or member.issym()
                    or member.islnk()
                ):
                    raise ProfileAbort(f"unsafe path in pinned source archive: {member.name}")
            archive.extractall(source_root, members=members)
        trainer = source_root / "scripts/train_rnn.py"
        if sha256_file(trainer) != meta.get("trainer_sha256"):
            raise ProfileAbort("pinned trainer source hash mismatch")
        phase.update({"phase": "mount_identity_and_fold", "updated_unix": time.time()})
        atomic_json(phase_path, phase)
        mounted = verify_and_stage_inputs(
            Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input")), source_root, identity_path
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["WNN_LOADER_WORKERS"] = "0"
        os.environ["WNN_THREADS"] = "2"
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != CAPS["visible_gpus"]:
            raise ProfileAbort("profile requires exactly one visible CUDA GPU")
        guard.torch = torch
        guard.check()
        if torch.cuda.mem_get_info(0)[0] < 6 * 1024**3:
            raise ProfileAbort("less than 6 GiB free GPU memory before profiling")
        device_identity = runtime_identity(torch)
        atomic_json(
            receipts / "environment.json",
            {
                "phase": "pre_profile_environment_identity",
                "identity": device_identity,
                "gpu_name_is_observed_not_hardcoded": True,
                "time_unix": time.time(),
            },
        )
        torch.cuda.reset_peak_memory_stats(0)
        profile_capture: dict[str, Any] = {}
        sys.path.insert(0, str(source_root))
        import importlib.util

        spec = importlib.util.spec_from_file_location("e520_train_rnn", trainer)
        if spec is None or spec.loader is None:
            raise ProfileAbort("could not load pinned train_rnn.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_make_model = module.make_model

        def capture_model(args: dict[str, Any], n_in: int, n_out: int) -> Any:
            model = original_make_model(args, n_in, n_out)
            profile_capture.update(
                {"model": model, "n_in": int(n_in), "n_out": int(n_out), "args": args.copy()}
            )
            return model

        module.make_model = capture_model
        original_stop = module.StageTimer.stop

        def capture_stop(timer: Any) -> None:
            if timer.cur is not None:
                timer.cur["_e520_stop_wall"] = time.perf_counter()
            original_stop(timer)
            guard.check()
            guard.raise_if_failed()

        module.StageTimer.stop = capture_stop
        original_report = module.StageTimer.report

        def report(timer: Any, warmup: int, batch: int) -> dict[str, Any]:
            summary = original_report(timer, warmup, batch)
            training_resources = resource_snapshot(torch, work)
            guard.latest = training_resources
            training_peaks = {
                "allocated_peak_bytes": training_resources["cuda_allocated_peak_bytes"],
                "reserved_peak_bytes": training_resources["cuda_reserved_peak_bytes"],
                "process_gpu_bytes_at_capture": training_resources["gpu_process_bytes"],
                "process_gpu_query_status": training_resources["gpu_process_query_status"],
                "process_gpu_query_raw": training_resources["gpu_process_query_raw"],
            }
            profile_capture["training_cuda_peaks"] = training_peaks
            atomic_json(
                receipts / "training_profile_peak.json",
                {
                    "phase": "training_profile_peak_captured_before_trainer_early_return",
                    "snapshot": training_resources,
                    "peaks": training_peaks,
                    "time_unix": time.time(),
                },
            )
            enforce_resource_caps(training_resources, time.monotonic() - start)
            guard.raise_if_failed()
            timed = timer.steps[warmup:]
            if len(timed) != PROFILE_STEPS:
                raise ProfileAbort(
                    f"trainer recorded {len(timed)} timed outer batches, expected {PROFILE_STEPS}"
                )
            intervals_ms = [
                (timed[i + 1]["_wall"] - timed[i]["_wall"]) * 1000 for i in range(len(timed) - 1)
            ]
            step_ms = [(item["_e520_stop_wall"] - item["_wall"]) * 1000 for item in timed]
            loader_ms = [value * 1000 for value in timer.loader_wait[warmup:]]
            profile_capture["timing"] = {
                "timed_outer_batches": len(timed),
                "end_to_end_inter_start_ms": percentile_summary(intervals_ms),
                "host_batch_span_ms": percentile_summary(step_ms),
                "loader_wait_ms": percentile_summary(loader_ms),
                "cuda_stage_median_ms": summary,
                "timing_note": "p50/p90 of inter-start throughput intervals (19 observations from 20 timed starts); separate host spans and loader waits are not added.",
            }
            print(
                "E520_PROFILE_TIMINGS " + json.dumps(profile_capture["timing"], sort_keys=True),
                flush=True,
            )
            return summary

        module.StageTimer.report = report
        argv = [
            "--name",
            "e520_profile_s3_f0",
            "--device",
            "cuda",
            "--arch",
            "gru",
            "--cell",
            "gru",
            "--hidden",
            "128",
            "--layers",
            "2",
            "--dropout",
            "0",
            "--seed",
            "3",
            "--raw-only",
            "--loss",
            "wmse",
            "--loss-norm",
            "chunk",
            "--train-source",
            "train+valid",
            "--train-seqs",
            "10607",
            "--split-manifest",
            SPLIT_RELATIVE.as_posix(),
            "--valid-fold",
            "0",
            "--epochs",
            "4",
            "--checkpoint-policy",
            "fixed",
            "--checkpoint-epoch",
            "3",
            "--no-eval-during-fit",
            "--batch",
            "16",
            "--chunk",
            "2000",
            "--lr",
            "0.001",
            "--wd",
            "0.01",
            "--optim",
            "adamw",
            "--sched",
            "onecycle",
            "--precision",
            "fp16",
            "--init",
            "",
            "--distill",
            "",
            "--distill-boot-h",
            "0",
            "--propensity",
            "",
            "--prop-input",
            "",
            "--prop-power",
            "1",
            "--prop-eps",
            "0",
            "--valid-true-mask",
            "-1",
            "--ema-teacher",
            "0",
            "--aux-struct",
            "0",
            "--aux-next",
            "0",
            "--aux-tier",
            "0",
            "--aux-mask",
            "0",
            "--aux-sign",
            "0",
            "--aux-quote",
            "0",
            "--aux",
            "",
            "--aux-weight",
            "0",
            "--swa-frac",
            "0",
            "--fast",
            "--profile-warmup",
            "3",
            "--profile-steps",
            "20",
        ]
        require_profile_only(argv)
        phase.update(
            {
                "phase": "trainer_profile",
                "updated_unix": time.time(),
                "profile_warmup": 3,
                "profile_steps": 20,
            }
        )
        atomic_json(phase_path, phase)
        sys.argv = [str(trainer), *argv]
        try:
            module.main()
        finally:
            module.StageTimer.stop = original_stop
            module.StageTimer.report = original_report
        guard.raise_if_failed()
        run_dir = source_root / "artifacts/runs/e520_profile_s3_f0"
        forbidden = [
            path
            for path in run_dir.rglob("*")
            if path.is_file()
            and (
                path.name.endswith(".pt")
                or path.name.endswith(".npy")
                or "pred" in path.name.lower()
            )
        ]
        if forbidden:
            raise ProfileAbort(
                f"profile trainer wrote forbidden checkpoint/prediction artifact: {forbidden[0].name}"
            )
        phase.update({"phase": "export_shape_probe", "updated_unix": time.time()})
        atomic_json(phase_path, phase)
        torch.cuda.empty_cache()
        model = profile_capture.get("model")
        if model is None:
            raise ProfileAbort("trainer profile did not instantiate its registered model")
        model.eval()
        model.zero_grad(set_to_none=True)
        from wnn.data import SequenceFile

        export_metrics = []
        for split_name in ("train", "valid"):
            file_path = source_root / "datasets" / f"{split_name}.parquet"
            reader = SequenceFile(file_path)
            chunks = []
            import numpy as np

            for row_group in mounted["orders"][split_name]:
                seq = reader.read(row_group, targets=False, mask=False)
                feature_chunk = np.asarray(seq.features[:4_000], dtype=np.float32)
                if feature_chunk.shape != (4_000, module.N_FEATURES):
                    raise ProfileAbort(
                        "raw input feature width/shape differs from pinned data schema"
                    )
                chunks.append(torch.from_numpy(feature_chunk.copy()))
                del seq, feature_chunk
            cpu_batch = torch.stack(chunks, dim=0)
            del chunks
            shape_bytes = validate_export_shape(16, 4_000, profile_capture["n_in"], 2)
            torch.cuda.reset_peak_memory_stats(0)
            before = resource_snapshot(torch, work)
            raw_batch = cpu_batch.to("cuda", non_blocking=False)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                model_inputs = module.build_inputs(raw_batch)
                outputs, _ = module.forward_chunk(model, model_inputs, slice(0, 4_000), None)
                outputs = (
                    module.readout(outputs, *module.READOUT, tilted=module.READOUT_TILTED)
                    if module.READOUT is not None
                    else outputs[..., :2].float()
                )
            output_finite = bool(torch.isfinite(outputs).all().item())
            validate_export_tensors(
                tuple(raw_batch.shape),
                tuple(model_inputs.shape),
                tuple(outputs.shape),
                raw_features=module.N_FEATURES,
                model_features=profile_capture["n_in"],
                raw_dtype=str(raw_batch.dtype),
                model_input_dtype=str(model_inputs.dtype),
                output_dtype=str(outputs.dtype),
                outputs_finite=output_finite,
            )
            torch.cuda.synchronize(0)
            after = resource_snapshot(torch, work)
            guard.check()
            export_metrics.append(
                {
                    "split": split_name,
                    "raw_input_shape": list(raw_batch.shape),
                    "model_input_shape": list(model_inputs.shape),
                    "input_dtype": str(model_inputs.dtype),
                    "output_shape": list(outputs.shape),
                    "output_dtype": str(outputs.dtype),
                    "output_finite": output_finite,
                    **shape_bytes,
                    "predictions_written": False,
                    "torch_allocated_peak_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "torch_reserved_peak_bytes": int(torch.cuda.max_memory_reserved(0)),
                    "rss_peak_bytes": after["rss_peak_bytes"],
                    "rss_before_bytes": before["rss_current_bytes"],
                    "rss_after_bytes": after["rss_current_bytes"],
                    "gpu_free_after_bytes": after["gpu_free_bytes"],
                }
            )
            del outputs, model_inputs, raw_batch, cpu_batch, reader
            torch.cuda.empty_cache()
        guard.raise_if_failed()
        final_resource = guard.check()
        guard.stop_polling()
        phase.update(
            {
                "phase": "complete",
                "status": "profile_only_complete",
                "updated_unix": time.time(),
                "end_unix": time.time(),
                "wall_seconds": time.monotonic() - start,
            }
        )
        result = {
            "packet": PACKET_ID,
            "status": "profile_only_complete",
            "fit_started": False,
            "checkpoint_written": False,
            "predictions_written": False,
            "source_archive_sha256": meta["source_archive_sha256"],
            "trainer_sha256": meta["trainer_sha256"],
            "external_review_sha256": meta["external_review_sha256"],
            "input_identity": mounted["identity"],
            "timing": profile_capture.get("timing"),
            "training_cuda_peaks": profile_capture.get("training_cuda_peaks"),
            "environment_identity": device_identity,
            "export_probe": export_metrics,
            "resources_final": final_resource,
            "caps": CAPS,
            "wall_seconds": time.monotonic() - start,
        }
        atomic_json(receipts / "profile_result.json", result)
        atomic_json(phase_path, phase)
        return result
    except BaseException as error:
        if guard is not None:
            guard.stop_polling()
        if identity_path.exists():
            try:
                identity = json.loads(identity_path.read_text())
                if identity.get("status") == "in_progress":
                    identity.update(
                        {"status": "failed", "failure": f"{type(error).__name__}: {error}"}
                    )
                    atomic_json(identity_path, identity)
            except (OSError, json.JSONDecodeError):
                pass
        phase.update(
            {
                "phase": "failed",
                "status": "aborted",
                "reason": f"{type(error).__name__}: {error}",
                "updated_unix": time.time(),
                "wall_seconds": time.monotonic() - start,
            }
        )
        atomic_json(phase_path, phase)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise ProfileAbort(str(error)) from error


if __name__ == "__main__":  # pragma: no cover - runtime-only entry point
    raise SystemExit("Use profile_kernel.py; it validates the external review gate first.")
