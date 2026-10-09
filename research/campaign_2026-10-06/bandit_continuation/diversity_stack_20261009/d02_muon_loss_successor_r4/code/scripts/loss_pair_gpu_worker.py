#!/usr/bin/env python3
"""Run one hash-pinned parsed-cache validation and the matched two-T4 loss pair."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
OUT = WORK / "out"
SOURCE_KERNEL = "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1"
SOURCE_DATASET = "vladkochetov07/wnn-d02-muon-loss-code-20261009-r4"
FOLD0_ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
EXPECTED_FOLD0_ASSET_MANIFEST_SHA256 = (
    "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
)
CPU_SOURCE_COMMIT = "d5daaa0f90b0362877522703c7e17aa78c19cac9"
EXPECTED_SOURCE_STAGE_RECEIPT_SHA256 = (
    "557cd0a61f1129f811dc104398c3c9c7295c3343a1c9a302018fcb588c0c1de7"
)
EXPECTED_PHASE_LOG_SHA256 = "cee62c0cdfa0a8b13998690fa49a7b111b56798a8ff6baf09e0a06b315c008e5"
EXPECTED_CACHE_MANIFEST_SHA256 = "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b"
EXPECTED_CACHE_KEY = "949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6"
EXPECTED_CPU_VALIDATION_SHA256 = "84686dff926036b37362a3ea28edcf2166d0bb44bee11b2fcbd64082e4b5cf35"
EXPECTED_CPU_PROFILE_SHA256 = "c371f3c70e79e69de8b382b442869d2e06e72ea4890a16d19b22923d7c82825e"
EXPECTED_SOURCE_VERIFICATION_SHA256 = (
    "1ccd91ccec85bd157cbd0244ae9032316b88d781a0caf19daf7a027f3a908567"
)
EXPECTED_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_REGISTRY_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
EXPECTED_MEMBERSHIP_SHA256 = "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd"
SPLIT_RELATIVE = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
REGISTRY_RELATIVE = "research/campaign_2026-10-04/portable_inputs.json"
MEMBERSHIP_RELATIVE = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "loss_pair_r1_membership_fold0_f03_s20261009.json"
)
WALL_SECONDS = 3300
POLL_SECONDS = 5
OUTPUT_CAP_BYTES = 12_000_000_000
CGROUP_CURRENT_CAP_BYTES = 24_000_000_000
MIN_MEMORY_AVAILABLE_BYTES = 2_000_000_000
MIN_CGROUP_HEADROOM_BYTES = 2_000_000_000
MIN_START_CGROUP_HEADROOM_BYTES = 4_000_000_000
MIN_START_DISK_FREE_BYTES = 4_000_000_000
_STARTED = time.monotonic()
_SAMPLES: list[dict[str, Any]] = []
_COMMANDS: list[dict[str, Any]] = []
_FOLD0_ASSET_RECEIPT_SHA256: str | None = None


class WorkerRejected(RuntimeError):
    """The mounted producer artifacts or runtime envelope failed a strict gate."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    if not replace and path.exists():
        temporary.unlink()
        raise WorkerRejected(f"refusing to replace an existing artifact: {path}")
    temporary.replace(path)


def _append_phase(phase: str, **fields: Any) -> None:
    row = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.monotonic() - _STARTED,
        "phase": phase,
        **fields,
    }
    with (WORK / "phase.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


def _find_files(root: Path, filename: str, max_depth: int = 8) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"input root is missing: {root}")
    root_depth = len(root.parts)
    hits: list[Path] = []
    for directory, subdirs, files in os.walk(root, followlinks=False):
        current = Path(directory)
        depth = len(current.parts) - root_depth
        subdirs[:] = sorted(subdirs) if depth < max_depth else []
        if filename in files:
            hits.append(current / filename)
    return hits


def _mounted_cache() -> tuple[Path, Path]:
    candidates = []
    for path in _find_files(INPUT, "manifest.json"):
        if (path.parent / "source_verification.json").is_file():
            candidates.append(path)
    matching = [path for path in candidates if _sha256_file(path) == EXPECTED_CACHE_MANIFEST_SHA256]
    if len(matching) != 1:
        raise WorkerRejected(
            f"expected one mounted producer manifest by SHA, found {len(matching)}"
        )
    manifest_path = matching[0]
    output_root = manifest_path.parent.parent
    receipt_path = output_root / "source_stage_receipt.json"
    phase_path = output_root.parent / "phase.jsonl"
    if (
        not receipt_path.is_file()
        or _sha256_file(receipt_path) != EXPECTED_SOURCE_STAGE_RECEIPT_SHA256
    ):
        raise WorkerRejected("mounted CPU source-stage receipt is missing or has changed")
    if not phase_path.is_file() or _sha256_file(phase_path) != EXPECTED_PHASE_LOG_SHA256:
        raise WorkerRejected("mounted CPU source-stage phase log is missing or has changed")
    return manifest_path, receipt_path


def _validate_source_stage(manifest_path: Path, receipt_path: Path) -> dict[str, Any]:
    source_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required_receipt = {
        "schema": "wnn.kaggle-parsed-cache-source-stage",
        "format_version": 1,
        "state": "complete",
        "source_commit": CPU_SOURCE_COMMIT,
        "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "cache_key": EXPECTED_CACHE_KEY,
        "reader_validation_sha256": EXPECTED_CPU_VALIDATION_SHA256,
        "reader_profile_sha256": EXPECTED_CPU_PROFILE_SHA256,
        "raw_source_verification_sha256": EXPECTED_SOURCE_VERIFICATION_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "portable_registry_sha256": EXPECTED_REGISTRY_SHA256,
        "membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "gpu_enabled": False,
        "tpu_enabled": False,
    }
    for key, expected in required_receipt.items():
        if source_receipt.get(key) != expected:
            raise WorkerRejected(f"CPU source-stage receipt differs at {key}")
    cache_root = manifest_path.parent
    pins = {
        "source_verification.json": EXPECTED_SOURCE_VERIFICATION_SHA256,
        "reader_validation.json": EXPECTED_CPU_VALIDATION_SHA256,
        "reader_profile.json": EXPECTED_CPU_PROFILE_SHA256,
    }
    for filename, expected in pins.items():
        path = cache_root / filename
        if not path.is_file() or _sha256_file(path) != expected:
            raise WorkerRejected(f"mounted CPU artifact changed: {filename}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("cache_key") != EXPECTED_CACHE_KEY
        or manifest.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256
        or manifest.get("portable_registry_sha256") != EXPECTED_REGISTRY_SHA256
        or manifest.get("membership_artifact", {}).get("sha256") != EXPECTED_MEMBERSHIP_SHA256
        or manifest.get("source_verification_artifact", {}).get("sha256")
        != EXPECTED_SOURCE_VERIFICATION_SHA256
    ):
        raise WorkerRejected("mounted parsed-cache manifest has a different source identity")
    return source_receipt


def _cgroup_directory() -> Path | None:
    candidates: list[Path] = []
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and relative.startswith("/"):
                parts = Path(relative.lstrip("/")).parts
                if ".." not in parts:
                    candidates.append(Path("/sys/fs/cgroup").joinpath(*parts))
            elif "memory" in controllers.split(",") and relative.startswith("/"):
                parts = Path(relative.lstrip("/")).parts
                if ".." not in parts:
                    candidates.append(Path("/sys/fs/cgroup/memory").joinpath(*parts))
    except (OSError, ValueError):
        pass
    candidates.extend((Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")))
    for directory in dict.fromkeys(candidates):
        if (directory / "memory.stat").is_file():
            return directory
    return None


def _memory_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    directory = _cgroup_directory()
    if directory is not None:
        result["cgroup_path"] = str(directory)
        for field, filenames in (
            ("current_bytes", ("memory.current", "memory.usage_in_bytes")),
            ("max_bytes", ("memory.max", "memory.limit_in_bytes")),
        ):
            for filename in filenames:
                path = directory / filename
                if path.is_file():
                    raw = path.read_text(encoding="utf-8").strip()
                    if raw != "max":
                        value = int(raw)
                        result[f"cgroup_{field}"] = (
                            None if field == "max_bytes" and value >= (1 << 60) else value
                        )
                    break
        try:
            for line in (directory / "memory.stat").read_text(encoding="utf-8").splitlines():
                key, raw = line.split()
                if key in {
                    "anon",
                    "file",
                    "file_dirty",
                    "file_writeback",
                    "inactive_file",
                    "active_file",
                }:
                    result[f"cgroup_{key}_bytes"] = int(raw)
        except OSError:
            pass
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                result["mem_available_bytes"] = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    return result


def _process_snapshot(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": pid}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(":")
            if key in {"Rss", "Pss", "Private_Clean", "Private_Dirty"}:
                result[f"{key.lower()}_bytes"] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        io_values = {}
        for line in Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(":")
            if raw:
                io_values[key] = int(raw.strip())
        result["io"] = io_values
    except (OSError, ValueError):
        result["io"] = None
    return result


def _tree_bytes(root: Path) -> int:
    total = 0
    if root.exists():
        for directory, _, names in os.walk(root, followlinks=False):
            for name in names:
                path = Path(directory) / name
                if not path.is_symlink():
                    try:
                        total += path.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
    return total


def _gpu_snapshot() -> list[dict[str, Any]] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            rows.append(
                {
                    "physical_index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_used_mib": int(fields[3]),
                    "utilization_percent": int(fields[4]),
                }
            )
        except ValueError:
            continue
    return rows


def _sample(phase: str, pids: dict[str, int] | None = None) -> dict[str, Any]:
    row = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.monotonic() - _STARTED,
        "phase": phase,
        "working_output_bytes": _tree_bytes(WORK),
        "working_disk_free_bytes": shutil.disk_usage(WORK).free,
        "memory": _memory_snapshot(),
        "children": {name: _process_snapshot(pid) for name, pid in (pids or {}).items()},
        "gpus": _gpu_snapshot(),
    }
    _SAMPLES.append(row)
    _write_json(WORK / "heartbeat.json", row)
    return row


def _guard(sample: dict[str, Any], deadline: float, *, startup: bool = False) -> str | None:
    if time.monotonic() >= deadline:
        return "bounded worker wall deadline reached"
    if int(sample["working_output_bytes"]) > OUTPUT_CAP_BYTES:
        return f"working outputs exceeded cap {OUTPUT_CAP_BYTES} bytes"
    if int(sample["working_disk_free_bytes"]) < MIN_START_DISK_FREE_BYTES:
        return "working disk free fell below 4 GB reserve"
    memory = sample["memory"]
    available = memory.get("mem_available_bytes")
    if not isinstance(available, int) or available < MIN_MEMORY_AVAILABLE_BYTES:
        return "MemAvailable is unavailable or below 2 GB"
    current = memory.get("cgroup_current_bytes")
    maximum = memory.get("cgroup_max_bytes")
    if not isinstance(current, int):
        return "cgroup current memory is unavailable"
    if current > CGROUP_CURRENT_CAP_BYTES:
        return "cgroup physical memory exceeded 24 GB aggregate cap"
    if isinstance(maximum, int):
        minimum = MIN_START_CGROUP_HEADROOM_BYTES if startup else MIN_CGROUP_HEADROOM_BYTES
        if maximum - current < minimum:
            return f"cgroup headroom fell below {minimum} bytes"
    elif startup:
        return "cgroup memory maximum is unavailable at worker admission"
    return None


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _environment(
    *, gpu: str | None = None, arm_dir: Path | None = None, run_name: str = ""
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "WNN_THREADS": "1",
            "WNN_PORTABLE_DATA_MANIFEST": str(ROOT / REGISTRY_RELATIVE),
            "WNN_LOSS_PAIR_PARITY": "1",
        }
    )
    if _FOLD0_ASSET_RECEIPT_SHA256 is not None:
        env["WNN_FOLD0_ASSET_VALIDATION_RECEIPT"] = str(
            OUT / "shared" / "fold0_assets_validation.json"
        )
        env["WNN_FOLD0_ASSET_VALIDATION_SHA256"] = _FOLD0_ASSET_RECEIPT_SHA256
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    if arm_dir is not None:
        env["WNN_LOSS_PAIR_ARM_DIR"] = str(arm_dir)
        env["WNN_LOSS_PAIR_RUN_NAME"] = run_name
    return env


def _launch(
    command: list[str], label: str, log_path: Path, env: dict[str, str]
) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("xb")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    _COMMANDS.append(
        {
            "label": label,
            "argv": command,
            "cwd": str(ROOT),
            "pid": process.pid,
            "log_path": log_path.relative_to(WORK).as_posix(),
            "started_monotonic_seconds": time.monotonic() - _STARTED,
            "environment": {
                key: env[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "WNN_THREADS",
                    "WNN_LOSS_PAIR_ARM_DIR",
                    "WNN_LOSS_PAIR_RUN_NAME",
                    "WNN_PORTABLE_DATA_MANIFEST",
                )
                if key in env
            },
        }
    )
    return process


def _wait_one(
    command: list[str],
    label: str,
    log_path: Path,
    deadline: float,
    env: dict[str, str],
    poll_seconds: float = POLL_SECONDS,
) -> dict[str, Any]:
    process = _launch(command, label, log_path, env)
    while process.poll() is None:
        sample = _sample(label, {label: process.pid})
        reason = _guard(sample, deadline)
        if reason:
            _append_phase("guard_stop", label=label, reason=reason)
            _terminate(process)
            return {"label": label, "returncode": process.returncode, "guard_stop": reason}
        time.sleep(min(poll_seconds, max(0.1, deadline - time.monotonic())))
    code = process.wait()
    _sample(f"{label}_terminal", {label: process.pid})
    entry = next(row for row in reversed(_COMMANDS) if row["label"] == label)
    entry["returncode"] = code
    entry["finished_monotonic_seconds"] = time.monotonic() - _STARTED
    entry = next(row for row in reversed(_COMMANDS) if row["label"] == label)
    return {
        "label": label,
        "returncode": code,
        "wall_seconds": entry["finished_monotonic_seconds"] - entry["started_monotonic_seconds"],
    }


def _wait_parallel(
    specs: list[tuple[list[str], str, Path, dict[str, str]]], deadline: float
) -> list[dict[str, Any]]:
    active: dict[str, subprocess.Popen[str]] = {}
    results: dict[str, dict[str, Any]] = {}
    try:
        for command, label, log_path, env in specs:
            active[label] = _launch(command, label, log_path, env)
        while active:
            live = {name: proc.pid for name, proc in active.items() if proc.poll() is None}
            sample = _sample("fit_pair", live)
            reason = _guard(sample, deadline)
            if reason:
                _append_phase("guard_stop", labels=sorted(active), reason=reason)
                for process in active.values():
                    _terminate(process)
                for label, process in active.items():
                    results[label] = {
                        "label": label,
                        "returncode": process.returncode,
                        "guard_stop": reason,
                    }
                active.clear()
                break
            for label, process in list(active.items()):
                code = process.poll()
                if code is not None:
                    process.wait()
                    entry = next(row for row in reversed(_COMMANDS) if row["label"] == label)
                    entry["returncode"] = code
                    entry["finished_monotonic_seconds"] = time.monotonic() - _STARTED
                    results[label] = {
                        "label": label,
                        "returncode": code,
                        "wall_seconds": entry["finished_monotonic_seconds"]
                        - entry["started_monotonic_seconds"],
                    }
                    del active[label]
            if active:
                time.sleep(min(POLL_SECONDS, max(0.1, deadline - time.monotonic())))
    except BaseException:
        for process in active.values():
            _terminate(process)
        raise
    return [results.get(label, {"label": label, "returncode": None}) for _, label, _, _ in specs]


def _gpu_inventory() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise WorkerRejected("exactly two visible CUDA devices are required")
    rows = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        name = str(properties.name)
        if "T4" not in name.upper():
            raise WorkerRejected(f"expected T4 at visible device {index}, found {name}")
        rows.append(
            {
                "visible_index": index,
                "name": name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [properties.major, properties.minor],
            }
        )
    if any(row["total_memory_bytes"] < 14_000_000_000 for row in rows):
        raise WorkerRejected("T4 memory inventory is below the expected 14 GB per device")
    return rows


def _common_train_argv(
    run_name: str,
    loss: str,
    manifest_path: Path,
    validation_path: Path,
    validation_sha256: str,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/train_rnn.py"),
        "--name",
        run_name,
        "--device",
        "cuda",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--enc",
        "linear",
        "--hidden",
        "256",
        "--layers",
        "2",
        "--train-source",
        "train",
        "--valid-fold",
        "0",
        "--full-valid",
        "--split-manifest",
        str(ROOT / SPLIT_RELATIVE),
        "--parsed-cache-manifest",
        str(manifest_path),
        "--parsed-cache-validation-receipt",
        str(validation_path),
        "--expected-parsed-cache-manifest-sha256",
        EXPECTED_CACHE_MANIFEST_SHA256,
        "--expected-parsed-cache-validation-sha256",
        validation_sha256,
        "--expected-parsed-cache-key",
        EXPECTED_CACHE_KEY,
        "--seed",
        "20261009",
        "--batch-order",
        "seeded",
        "--batch",
        "32",
        "--chunk",
        "2000",
        "--epochs",
        "2",
        "--optim",
        "muon",
        "--muon-scope",
        "rnn",
        "--muon-ortho-dtype-policy",
        "fp32",
        "--warmup-frac",
        "0.05",
        "--beta2",
        "0.999",
        "--head-lr-mult",
        "1.0",
        "--readout-activation",
        "gelu",
        "--dropout",
        "0",
        "--sched",
        "onecycle",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--clip",
        "1.0",
        "--precision",
        "fp32",
        "--loss-norm",
        "chunk",
        "--huber-delta",
        "1.0",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "1",
        "--no-eval-during-fit",
        "--save-training-state",
        "--grad-stats",
        "--deterministic",
        "--sync-free",
        "--pin-memory",
        "--loss",
        loss,
    ]


def _hash_array(array: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _artifact(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _copy_required(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise WorkerRejected(f"required produced artifact is absent or non-regular: {source}")
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _verify_prefit_pair(arms: dict[str, Path]) -> None:
    import numpy as np

    wmse, huber = arms["wmse"], arms["huber_d1"]
    for filename in ("initial_state.npz", "normalization.npz", "sample_orders.json"):
        left = wmse / "prefit" / filename
        right = huber / "prefit" / filename
        if not left.is_file() or not right.is_file():
            raise WorkerRejected(f"paired pre-fit artifact is missing: {filename}")
        if filename.endswith(".json"):
            if json.loads(left.read_text()) != json.loads(right.read_text()):
                raise WorkerRejected(f"paired pre-fit metadata differs: {filename}")
        else:
            with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
                if set(a.files) != set(b.files):
                    raise WorkerRejected(f"paired pre-fit NPZ keys differ: {filename}")
                for key in a.files:
                    if (
                        a[key].dtype != b[key].dtype
                        or a[key].shape != b[key].shape
                        or not np.array_equal(a[key], b[key])
                    ):
                        raise WorkerRejected(f"paired pre-fit NPZ values differ: {filename}:{key}")
    wmse_schedule = json.loads((wmse / "schedule_trace.json").read_text())
    huber_schedule = json.loads((huber / "schedule_trace.json").read_text())
    if wmse_schedule != huber_schedule or wmse_schedule.get("total_steps") != 220:
        raise WorkerRejected(
            "WMSE and Huber OneCycle pre-update traces do not match exactly at 220 steps"
        )
    if wmse_schedule.get("schema") != "wnn.loss-pair.onecycle-trace.v1":
        raise WorkerRejected("trainer schedule trace schema differs from the TPU consumer contract")


def _copy_arm_outputs(run_name: str, arm_dir: Path) -> None:
    run_dir = ROOT / "artifacts" / "runs" / run_name
    missing = []
    for filename in (
        "model_ep0.pt",
        "model_ep1.pt",
        "model.pt",
        "training_state.pt",
        "provenance.json",
        "history.json",
        "checkpoint_timing.jsonl",
    ):
        source = run_dir / filename
        if source.is_file() and not source.is_symlink():
            _copy_required(source, arm_dir / "model" / filename)
        else:
            missing.append(filename)
    for filename in ("update_trace.jsonl", "schedule_trace.json"):
        source = arm_dir / filename
        if not source.is_file():
            missing.append(filename)
    _write_json(arm_dir / "artifact_collection.json", {"missing": missing})


def _export_predictions(
    name: str,
    run_name: str,
    physical_gpu: str,
    manifest_path: Path,
    validation_path: Path,
    validation_sha256: str,
    arm_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts/predict_run.py"),
        run_name,
        "valid",
        "--parsed-cache-manifest",
        str(manifest_path),
        "--parsed-cache-validation-receipt",
        str(validation_path),
        "--expected-parsed-cache-manifest-sha256",
        EXPECTED_CACHE_MANIFEST_SHA256,
        "--expected-parsed-cache-validation-sha256",
        validation_sha256,
        "--expected-parsed-cache-key",
        EXPECTED_CACHE_KEY,
    ]
    return {
        "command": command,
        "env": _environment(gpu=physical_gpu),
        "label": f"predict_{name}",
        "log_path": OUT / "logs" / f"predict_{name}.log",
        "arm_dir": arm_dir,
    }


def _collect_prediction(prediction: dict[str, Any], run_name: str, arm_dir: Path) -> None:
    import numpy as np

    source = ROOT / "cache" / f"pred_{run_name}_valid.npy"
    sidecar_path = source.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    # The producer manifest stores source-qualified row objects in its sibling membership file.
    membership_meta = prediction["manifest"]["membership_artifact"]
    membership_path = Path(prediction["manifest_path"]).parent / membership_meta["path"]
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    rows = membership["memberships"]["evaluation_valid"]
    expected_groups = [int(row["row_group"]) for row in rows]
    if sidecar.get("row_groups") != expected_groups or len(rows) != 468:
        raise WorkerRejected("prediction row order does not match the fixed evaluation membership")
    mapped = np.load(source, mmap_mode="r", allow_pickle=False)
    if mapped.shape != (468, 20_000, 2) or mapped.dtype != np.float32:
        raise WorkerRejected("prediction array has the wrong shape or dtype")
    for start in range(0, len(mapped), 8):
        if not np.isfinite(mapped[start : start + 8]).all():
            raise WorkerRejected("prediction array contains nonfinite values")
    if sidecar.get("parsed_sequence_cache", {}).get("cache_key") != EXPECTED_CACHE_KEY:
        raise WorkerRejected("prediction sidecar does not bind the parsed cache")
    destination = arm_dir / "predictions_valid.npy"
    _copy_required(source, destination)
    _copy_required(sidecar_path, arm_dir / "predictions_valid.json")
    membership_sidecar = {
        "schema": "wnn.loss-pair.prediction-membership.v1",
        "format_version": 1,
        "prediction_path": "predictions_valid.npy",
        "prediction_shape": list(mapped.shape),
        "prediction_dtype": mapped.dtype.str,
        "cache_key": EXPECTED_CACHE_KEY,
        "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "rows": [
            {
                "position": index,
                "source": row["source"],
                "row_group": int(row["row_group"]),
                "sequence_id": str(row["sequence_id"]),
            }
            for index, row in enumerate(rows)
        ],
    }
    _write_json(arm_dir / "prediction_membership.json", membership_sidecar)


def _collect_shared(
    arms: dict[str, Path],
    arm_run_names: dict[str, str],
    runtime_validation: Path,
    runtime_profile: Path,
    source_receipt: dict[str, Any],
    bootstrap_receipt: dict[str, Any],
    manifest_path: Path,
    fit_results: list[dict[str, Any]],
    prediction_results: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    _verify_prefit_pair(arms)
    shared = OUT / "shared"
    for filename in (
        "initial_state.npz",
        "normalization.npz",
        "normalization.json",
        "sample_orders.json",
        "profile_inputs.npz",
    ):
        _copy_required(arms["wmse"] / "prefit" / filename, shared / filename)
    _copy_required(runtime_validation, shared / "cache_validation_runtime.json")
    _copy_required(runtime_profile, shared / "cache_reader_profile_runtime.json")
    schedule = json.loads((arms["wmse"] / "schedule_trace.json").read_text(encoding="utf-8"))
    _write_json(shared / "schedule_trace.json", schedule)
    with np.load(shared / "profile_inputs.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"inputs", "targets", "row_mask"}:
            raise WorkerRejected("actual profile fixture NPZ keys differ from the TPU contract")
        if arrays["inputs"].shape != (32, 2000, 211) or arrays["inputs"].dtype != np.dtype("<f4"):
            raise WorkerRejected("actual profile feature fixture has the wrong shape or dtype")
        if arrays["targets"].shape != (32, 2000, 2) or arrays["targets"].dtype != np.dtype("<f4"):
            raise WorkerRejected("actual profile target fixture has the wrong shape or dtype")
        if arrays["row_mask"].shape != (32, 2000) or arrays["row_mask"].dtype != np.dtype("|b1"):
            raise WorkerRejected("actual profile row mask has the wrong shape or dtype")
        expected_mask = np.arange(2000, dtype=np.int64)[None, :] >= 99
        if not np.array_equal(arrays["row_mask"], np.broadcast_to(expected_mask, (32, 2000))):
            raise WorkerRejected("actual profile fixture row mask is not the WARMUP=99 mask")
    sample_orders = json.loads((shared / "sample_orders.json").read_text(encoding="utf-8"))
    if sample_orders.get("schema") != "wnn.loss-pair.sample-orders.v1":
        raise WorkerRejected("sample order schema differs from the TPU contract")
    first_ids = sample_orders["epochs"]["0"][:32]
    if len(first_ids) != 32 or len(set(first_ids)) != 32:
        raise WorkerRejected("first actual profile batch lacks 32 distinct source-qualified IDs")
    profile_receipt = {
        "schema": "wnn.loss-pair.profile-inputs-receipt.v1",
        "format_version": 1,
        "source": "WMSE arm, actual first epoch-zero batch, before update attempt zero",
        "feature_semantics": "112 raw float32 features plus 99 causal event features; not normalized; model applies shared normalization",
        "target_semantics": "raw float32 target labels; TPU clips to [-2,2] and weights by absolute clipped labels",
        "row_mask_semantics": "row index >= 99; not the official is_scored mask",
        "sequence_ids": first_ids,
        "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "cache_validation_receipt_sha256": _sha256_file(shared / "cache_validation_runtime.json"),
        "cache_key": EXPECTED_CACHE_KEY,
        "initial_state_sha256": _sha256_file(shared / "initial_state.npz"),
        "normalization_sha256": _sha256_file(shared / "normalization.npz"),
        "profile_inputs_sha256": _sha256_file(shared / "profile_inputs.npz"),
        "profile_inputs_arrays": {
            "inputs": {"shape": [32, 2000, 211], "dtype": "<f4"},
            "targets": {"shape": [32, 2000, 2], "dtype": "<f4"},
            "row_mask": {"shape": [32, 2000], "dtype": "|b1"},
        },
    }
    _write_json(shared / "profile_inputs_receipt.json", profile_receipt)
    role_paths = {
        "device_profile_gate": shared / "device_profile_gate.json",
        "fold0_asset_validation_receipt": shared / "fold0_assets_validation.json",
        "cache_manifest": OUT / "shared" / "cache_manifest.json",
        "cache_validation_receipt": shared / "cache_validation_runtime.json",
        "cache_reader_profile": shared / "cache_reader_profile_runtime.json",
        "source_stage_receipt": OUT / "shared" / "source_stage_receipt.json",
        "source_verification": OUT / "shared" / "source_verification.json",
        "initial_state": shared / "initial_state.npz",
        "normalization": shared / "normalization.npz",
        "normalization_metadata": shared / "normalization.json",
        "sample_orders": shared / "sample_orders.json",
        "onecycle_trace": shared / "schedule_trace.json",
        "profile_inputs": shared / "profile_inputs.npz",
        "profile_inputs_receipt": shared / "profile_inputs_receipt.json",
    }
    cache_root = manifest_path.parent
    # Keep the exact producer metadata and small source receipts beside shared artifacts.
    producer_manifest_path = manifest_path
    for source, destination in (
        (producer_manifest_path, role_paths["cache_manifest"]),
        (
            producer_manifest_path.parent / "source_verification.json",
            role_paths["source_verification"],
        ),
        (
            producer_manifest_path.parent.parent / "source_stage_receipt.json",
            role_paths["source_stage_receipt"],
        ),
    ):
        _copy_required(source, destination)
    artifact_roles = {role: _artifact(path, OUT) for role, path in role_paths.items()}
    arms_contract: dict[str, Any] = {}
    for arm_name, run_name in arm_run_names.items():
        arm_dir = OUT / "arms" / arm_name
        arm_artifacts = []
        for path in sorted(arm_dir.rglob("*")):
            if path.is_file() and not path.is_symlink():
                arm_artifacts.append(_artifact(path, OUT))
        trace_path = arm_dir / "update_trace.jsonl"
        update_seconds = [
            float(json.loads(line)["elapsed_seconds"])
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        checkpoint_rows = [
            json.loads(line)
            for line in (arm_dir / "model" / "checkpoint_timing.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        fit_row = next(row for row in fit_results if row.get("label") == f"fit_{arm_name}")
        prediction_row = prediction_results[arm_name]
        arms_contract[arm_name] = {
            "run_name": run_name,
            "loss": "wmse" if arm_name == "wmse" else "huber",
            "huber_delta": 1.0,
            "updates_attempted": 220,
            "timing": {
                "fit_process_wall_seconds_including_startup_and_checkpoint_export": fit_row.get(
                    "wall_seconds"
                ),
                "prediction_export_wall_seconds": prediction_row.get("wall_seconds"),
                "first_update_cuda_event_seconds_including_lazy_setup": update_seconds[0]
                if update_seconds
                else None,
                "median_update_cuda_event_seconds_after_first_five": (
                    float(np.median(update_seconds[5:])) if len(update_seconds) > 5 else None
                ),
                "checkpoint_and_state_export_seconds_by_epoch": checkpoint_rows,
            },
            "artifacts": arm_artifacts,
        }
    contract = {
        "schema": "wnn.loss-pair.training-contract.v1",
        "format_version": 1,
        "state": "complete",
        "d02_controls": {
            "training_contract_sha256": os.environ["WNN_D02_CONTROL_CONTRACT_SHA256"],
            "root_admission_sha256": bootstrap_receipt["d02_root_admission_sha256"],
        },
        "source": {
            "dataset": SOURCE_DATASET,
            "fold0_asset_dataset": FOLD0_ASSET_DATASET,
            "fold0_asset_manifest_sha256": EXPECTED_FOLD0_ASSET_MANIFEST_SHA256,
            "fold0_asset_validation_receipt_sha256": artifact_roles[
                "fold0_asset_validation_receipt"
            ]["sha256"],
            "source_commit": bootstrap_receipt["source_commit"],
            "trainer_base_commit": CPU_SOURCE_COMMIT,
            "source_stage_state": source_receipt["state"],
            "asset_manifest_sha256": bootstrap_receipt["asset_manifest_sha256"],
            "archive_sha256": bootstrap_receipt["archive_sha256"],
            "code_manifest_sha256": bootstrap_receipt["code_manifest_sha256"],
            "source_stage_kernel": SOURCE_KERNEL,
            "source_stage_receipt_sha256": EXPECTED_SOURCE_STAGE_RECEIPT_SHA256,
        },
        "cache": {
            "scope": "cohort",
            "parse_once": True,
            "parsed_cache_kernel": SOURCE_KERNEL,
            "manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
            "validation_receipt_sha256": artifact_roles["cache_validation_receipt"]["sha256"],
            "cache_key": EXPECTED_CACHE_KEY,
            "source_verification_sha256": EXPECTED_SOURCE_VERIFICATION_SHA256,
            "reader_profile_sha256": artifact_roles["cache_reader_profile"]["sha256"],
            "membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "portable_registry_sha256": EXPECTED_REGISTRY_SHA256,
            "fold": 0,
            "fit_train_count": 350,
            "fit_valid_count_not_used_for_fit": 28,
            "evaluation_valid_count": 468,
            "mount_path": str(cache_root),
            "array_payload_bytes": 7_732_440_000,
        },
        "model": {
            "family": "two-layer GRU",
            "input_width": 211,
            "input_layout": {"raw_features": 112, "causal_event_features": 99},
            "input_projection": "linear",
            "hidden_size": 256,
            "layers": 2,
            "activation": "GELU",
            "outputs": 2,
            "precision": "float32",
            "tf32": False,
        },
        "fit": {
            "seed": 20261009,
            "batch_size": 32,
            "chunk_rows": 2000,
            "epochs": 2,
            "updates_per_arm": 220,
            "optimizer": "MuonAdamW",
            "muon_policy": {
                "scope": "rnn",
                "gatewise": "all",
                "ortho_dtype": "fp32",
                "lr_multiplier": 1.0,
            },
            "optimizer_defaults": {"beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0.01},
            "scheduler": "OneCycleLR",
            "schedule_trace_artifact_role": "onecycle_trace",
            "learning_rate_max": 0.001,
            "global_gradient_clip": 1.0,
            "pre_fit_device_gate_artifact_role": "device_profile_gate",
            "loss_normalization": "per chunk",
            "warmup_row_mask": "row_index >= 99",
            "checkpoint_epoch_zero_indexed": 1,
            "evaluation": "full fixed 468 sequence export after fit; held out during fitting",
        },
        "arms": arms_contract,
        "artifacts": artifact_roles,
        "resource_profile": {
            "sample_count": len(_SAMPLES),
            "sample_interval_seconds": POLL_SECONDS,
            "max_cgroup_current_bytes": max(
                (row["memory"].get("cgroup_current_bytes", 0) for row in _SAMPLES), default=None
            ),
            "min_mem_available_bytes": min(
                (row["memory"].get("mem_available_bytes", 0) for row in _SAMPLES), default=None
            ),
            "max_working_output_bytes": max(
                (int(row["working_output_bytes"]) for row in _SAMPLES), default=0
            ),
        },
        "runtime": {
            "worker_wall_seconds": time.monotonic() - _STARTED,
            "process_records": _COMMANDS,
            "scientific_result_claim": "none until independent grading; exposed fold-0 screen",
        },
    }
    _write_json(OUT / "training_contract.json", contract)
    return contract, profile_receipt


def _device_gate_argv(
    run_name: str, loss: str, manifest_path: Path, validation_path: Path, validation_sha256: str
) -> list[str]:
    """Disposable profiles cannot request epoch continuation state (GPU R2 failure)."""
    command = _common_train_argv(run_name, loss, manifest_path, validation_path, validation_sha256)
    command.remove("--save-training-state")
    return command + ["--profile-steps", "1", "--profile-warmup", "0"]


def _run_device_shape_update_gate(
    manifest_path: Path,
    validation_path: Path,
    validation_sha256: str,
    gpu_inventory: list[dict[str, Any]],
    deadline: float,
    loss: str = "wmse",
) -> dict[str, Any]:
    import numpy as np

    gate_dir = OUT / "profile_gate" / loss
    gate_dir.parent.mkdir(exist_ok=True)
    gate_dir.mkdir(exist_ok=False)
    run_name = f"d02_muon_{loss}_device_shape_profile"
    command = _device_gate_argv(run_name, loss, manifest_path, validation_path, validation_sha256)
    gate_label = f"device_shape_update_gate_{loss}"
    log_path = OUT / "logs" / f"{gate_label}.log"
    before = _sample(f"{gate_label}_before")
    reason = _guard(before, deadline)
    if reason:
        raise WorkerRejected(f"device shape/update gate resource check failed: {reason}")
    _append_phase(
        "device_shape_update_gate_start",
        loss=loss,
        visible_gpu="0",
        recipe="same B32/C2000/H256 two-layer FP32 GRU; one batch, ten chunk updates",
    )
    result = _wait_one(
        command,
        gate_label,
        log_path,
        deadline,
        _environment(gpu="0", arm_dir=gate_dir, run_name=run_name),
        poll_seconds=1.0,
    )
    after = _sample(f"{gate_label}_after")
    if result.get("returncode") != 0 or result.get("guard_stop"):
        raise WorkerRejected(f"actual-T4 shape/update gate failed: {result}")
    evidence_lines = [
        line.partition("PROFILE_UPDATE_EVIDENCE ")[2]
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if "PROFILE_UPDATE_EVIDENCE " in line
    ]
    if len(evidence_lines) != 1:
        raise WorkerRejected("actual-T4 profile did not emit exactly one update-evidence record")
    evidence = json.loads(evidence_lines[0])
    if (
        evidence.get("valid_for_throughput") is not True
        or evidence.get("optimizer_attempts") != 10
        or evidence.get("optimizer_completed_updates") != 10
        or evidence.get("amp_skipped_updates") != 0
        or evidence.get("loss_finite") is not True
        or evidence.get("gradient_finite") is not True
    ):
        raise WorkerRejected(f"actual-T4 profile update evidence is invalid: {evidence}")
    precision_lines = [
        line.partition("MUON_PRECISION ")[2]
        for line in log_path.read_text().splitlines()
        if "MUON_PRECISION " in line
    ]
    if len(precision_lines) != 1:
        raise WorkerRejected("D02 device gate omitted its explicit Muon precision record")
    precision = json.loads(precision_lines[0])
    if (
        precision.get("selected_orthogonalization_dtype") != "float32"
        or precision.get("model_precision") != "fp32"
    ):
        raise WorkerRejected("D02 device gate used different model/optimizer precision")
    reload_receipt = json.loads((gate_dir / "device_gate_reload.json").read_text())
    if reload_receipt.get("status") != "pass" or reload_receipt.get("device_type") != "cuda":
        raise WorkerRejected("D02 actual-device reload/reset/prefix gate failed")
    prefit = gate_dir / "prefit"
    required = (
        "initial_state.npz",
        "normalization.npz",
        "normalization.json",
        "sample_orders.json",
        "profile_inputs.npz",
    )
    missing = [name for name in required if not (prefit / name).is_file()]
    if missing:
        raise WorkerRejected(f"actual-T4 pre-fit producer omitted artifacts: {missing}")
    profile_npz = prefit / "profile_inputs.npz"
    with np.load(profile_npz, allow_pickle=False) as arrays:
        expected = {
            "inputs": ((32, 2000, 211), "<f4"),
            "targets": ((32, 2000, 2), "<f4"),
            "row_mask": ((32, 2000), "|b1"),
        }
        if set(arrays.files) != set(expected):
            raise WorkerRejected("actual-T4 producer fixture has unexpected NPZ keys")
        for name, (shape, dtype) in expected.items():
            if arrays[name].shape != shape or arrays[name].dtype.str != dtype:
                raise WorkerRejected(f"actual-T4 producer fixture differs at {name}")
    gate_samples = [row for row in _SAMPLES if row.get("phase") == gate_label]
    memory_peak = max(
        (int(row["memory"].get("cgroup_current_bytes", 0)) for row in gate_samples),
        default=0,
    )
    child_rows = [
        child
        for row in gate_samples
        for child in row.get("children", {}).values()
        if isinstance(child, dict)
    ]
    gpu_used_mib = [
        int(device["memory_used_mib"])
        for row in gate_samples
        for device in row.get("gpus", []) or []
        if int(device.get("physical_index", -1)) == 0
    ]
    profile_receipt = {
        "schema": "wnn.loss-pair.device-shape-update-gate.v1",
        "format_version": 1,
        "state": "passed",
        "run_name": run_name,
        "loss": loss,
        "muon_precision": precision,
        "reload": reload_receipt,
        "update_trace": _artifact(gate_dir / "update_trace.jsonl", OUT),
        "device": gpu_inventory[0],
        "profile": {
            "batch_size": 32,
            "chunk_rows": 2000,
            "input_width": 211,
            "hidden_size": 256,
            "layers": 2,
            "precision": "float32",
            "tf32": False,
            "measured_batches": 1,
            "warmup_batches": 0,
            "attempted_updates": 10,
            "evidence": evidence,
            "process_wall_seconds": result.get("wall_seconds"),
        },
        "log": _artifact(log_path, OUT),
        "resource": {
            "sample_interval_seconds": 1.0,
            "cgroup_current_before_bytes": before["memory"].get("cgroup_current_bytes"),
            "cgroup_current_after_bytes": after["memory"].get("cgroup_current_bytes"),
            "cgroup_current_peak_sampled_bytes": memory_peak,
            "child_peak_rss_bytes": max(
                (int(row.get("rss_bytes", 0)) for row in child_rows), default=0
            ),
            "child_peak_pss_bytes": max(
                (int(row.get("pss_bytes", 0)) for row in child_rows), default=0
            ),
            "gpu0_memory_used_peak_sampled_mib": max(gpu_used_mib, default=None),
            "gpu_samples_before": before.get("gpus"),
            "gpu_samples_after": after.get("gpus"),
        },
        "producer_fixture": {name: _artifact(prefit / name, OUT) for name in required},
    }
    _write_json(OUT / "shared" / f"device_profile_gate_{loss}.json", profile_receipt)
    _append_phase(
        "device_shape_update_gate_pass",
        loss=loss,
        attempts=10,
        cgroup_current_peak_sampled_bytes=memory_peak,
        receipt_sha256=_sha256_file(OUT / "shared" / f"device_profile_gate_{loss}.json"),
    )
    return profile_receipt


def run_worker(bootstrap_receipt: dict[str, Any]) -> dict[str, Any]:
    global _FOLD0_ASSET_RECEIPT_SHA256
    admission = json.loads(os.environ["WNN_D02_ROOT_ADMISSION_JSON"])
    if (
        admission.get("push_allowed") is not True
        or admission.get("source_code_manifest_sha256") != bootstrap_receipt["code_manifest_sha256"]
        or not admission.get("gpu_control_contract_sha256")
    ):
        raise WorkerRejected("D02 root source/provider/control admission is pending")
    from wnn.d02_controls import digest, load_controls

    control_paths = [
        path
        for path in INPUT.rglob("training_contract.json")
        if digest(path) == admission["gpu_control_contract_sha256"]
    ]
    if len(control_paths) != 1:
        raise WorkerRejected("D02 requires exactly one pinned completed AdamW control contract")
    load_controls(control_paths[0], admission["gpu_control_contract_sha256"])
    os.environ["WNN_D02_CONTROL_CONTRACT"] = str(control_paths[0])
    os.environ["WNN_D02_CONTROL_CONTRACT_SHA256"] = admission["gpu_control_contract_sha256"]
    runtime_seconds = admission.get("runtime_seconds", 0)
    if type(runtime_seconds) is not int or not 0 < runtime_seconds <= WALL_SECONDS:
        raise WorkerRejected("D02 root runtime exceeds the finite worker envelope")
    deadline = _STARTED + runtime_seconds
    if OUT.exists():
        present = {path.name for path in OUT.iterdir()}
        if present != {"bootstrap_receipt.json", "logs"}:
            raise WorkerRejected("output root contains files outside the sealed bootstrap receipt")
    else:
        OUT.mkdir(parents=True)
        (OUT / "logs").mkdir()
    (OUT / "arms").mkdir(exist_ok=False)
    (OUT / "shared").mkdir(exist_ok=False)
    fold0_receipt_path = Path(bootstrap_receipt.get("fold0_asset_validation_receipt", ""))
    fold0_receipt_sha256 = bootstrap_receipt.get("fold0_asset_validation_receipt_sha256")
    if (
        bootstrap_receipt.get("fold0_asset_dataset") != FOLD0_ASSET_DATASET
        or bootstrap_receipt.get("fold0_asset_manifest_sha256")
        != EXPECTED_FOLD0_ASSET_MANIFEST_SHA256
        or not fold0_receipt_path.is_file()
        or not isinstance(fold0_receipt_sha256, str)
        or _sha256_file(fold0_receipt_path) != fold0_receipt_sha256
    ):
        raise WorkerRejected("bootstrap omitted the verified fold-0 auxiliary asset receipt")
    shared_fold0_receipt = OUT / "shared" / "fold0_assets_validation.json"
    _copy_required(fold0_receipt_path, shared_fold0_receipt)
    _FOLD0_ASSET_RECEIPT_SHA256 = _sha256_file(shared_fold0_receipt)
    if fold0_receipt_sha256 != _FOLD0_ASSET_RECEIPT_SHA256:
        raise WorkerRejected("copied fold-0 auxiliary asset receipt differs from bootstrap")
    from wnn.fold0_assets import validate_preverified_fold0_assets

    validate_preverified_fold0_assets(
        ROOT,
        ROOT / SPLIT_RELATIVE,
        ROOT / REGISTRY_RELATIVE,
        shared_fold0_receipt,
        fold0_receipt_sha256,
    )
    _append_phase(
        "fold0_auxiliary_assets_verified",
        asset_manifest_sha256=EXPECTED_FOLD0_ASSET_MANIFEST_SHA256,
        receipt_sha256=fold0_receipt_sha256,
    )
    _append_phase(
        "worker_start", worker_wall_seconds=WALL_SECONDS, output_cap_bytes=OUTPUT_CAP_BYTES
    )
    manifest_path, source_stage_receipt_path = _mounted_cache()
    source_receipt = _validate_source_stage(manifest_path, source_stage_receipt_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_root = manifest_path.parent
    expected_disk = shutil.disk_usage(WORK).free
    if expected_disk < MIN_START_DISK_FREE_BYTES:
        raise WorkerRejected(f"working disk free {expected_disk} < 4 GB start reserve")
    _append_phase(
        "producer_mount_verified",
        cache_manifest_sha256=EXPECTED_CACHE_MANIFEST_SHA256,
        source_stage_receipt_sha256=EXPECTED_SOURCE_STAGE_RECEIPT_SHA256,
        cache_mount=str(cache_root),
        disk_free_bytes=expected_disk,
    )
    runtime_validation = OUT / "shared" / "cache_validation_runtime.json"
    runtime_profile = OUT / "shared" / "cache_reader_profile_runtime.json"
    profile_command = [
        sys.executable,
        str(ROOT / "scripts/profile_parsed_sequence_cache.py"),
        "--manifest",
        str(manifest_path),
        "--expected-manifest-sha256",
        EXPECTED_CACHE_MANIFEST_SHA256,
        "--validation-receipt",
        str(runtime_validation),
        "--profile-output",
        str(runtime_profile),
        "--per-role",
        "2",
        "--seed",
        "20261009",
    ]
    profile_result = _wait_one(
        profile_command,
        "cache_profile_verify",
        OUT / "logs" / "cache_profile_verify.log",
        deadline,
        _environment(),
    )
    if profile_result.get("returncode") != 0:
        raise WorkerRejected("mounted cache verification/profile process failed")
    runtime_validation_sha = _sha256_file(runtime_validation)
    runtime_profile_sha = _sha256_file(runtime_profile)
    profile_value = json.loads(runtime_profile.read_text(encoding="utf-8"))
    if (
        profile_value.get("cache_key") != EXPECTED_CACHE_KEY
        or profile_value.get("validation_receipt_sha256") != runtime_validation_sha
    ):
        raise WorkerRejected("runtime cache profile does not bind its validation receipt")
    inventory = _gpu_inventory()
    sample = _sample("pre_fit_gate")
    reason = _guard(sample, deadline, startup=True)
    if reason:
        raise WorkerRejected(f"pre-fit resource gate failed: {reason}")
    if len(inventory) != 2:
        raise WorkerRejected("two-T4 inventory did not pass")
    _write_json(
        OUT / "gpu_inventory.json",
        {"schema": "wnn.loss-pair.gpu-inventory.v1", "devices": inventory},
    )
    _append_phase("pre_fit_gate_pass", gpu_inventory=inventory, memory=sample["memory"])

    gates = {
        loss: _run_device_shape_update_gate(
            manifest_path, runtime_validation, runtime_validation_sha, inventory, deadline, loss
        )
        for loss in ("wmse", "huber")
    }
    device_profile_gate = {
        "schema": "wnn.d02.two-loss-device-gate.v1",
        "state": "passed",
        "arms": gates,
    }
    _write_json(OUT / "shared" / "device_profile_gate.json", device_profile_gate)
    if admission.get("conditional_fit_allowed") is not True:
        _write_json(OUT / "d02_profile_only_receipt.json", device_profile_gate)
        return {
            "state": "profile_complete_fit_not_admitted",
            "device_profile_gate": device_profile_gate,
        }

    arm_names = {"wmse": "d02_muon_wmse_r1", "huber_d1": "d02_muon_huber_d1_r1"}
    arm_dirs = {name: OUT / "arms" / name for name in arm_names}
    for arm_dir in arm_dirs.values():
        arm_dir.mkdir(parents=True, exist_ok=False)
    fit_specs = []
    for gpu_index, (arm, loss) in enumerate((("wmse", "wmse"), ("huber_d1", "huber"))):
        run_name = arm_names[arm]
        env = _environment(gpu=str(gpu_index), arm_dir=arm_dirs[arm], run_name=run_name)
        command = _common_train_argv(
            run_name,
            loss,
            manifest_path,
            runtime_validation,
            runtime_validation_sha,
        )
        fit_specs.append((command, f"fit_{arm}", OUT / "logs" / f"fit_{arm}.log", env))
    _append_phase(
        "fit_pair_start",
        arms=arm_names,
        precision="float32",
        tf32=False,
        updates_per_arm=220,
        device_visibility={"wmse": "0", "huber_d1": "1"},
    )
    fit_results = _wait_parallel(fit_specs, deadline)
    for arm, run_name in arm_names.items():
        _copy_arm_outputs(run_name, arm_dirs[arm])
    fit_ok = all(
        result.get("returncode") == 0 and not result.get("guard_stop") for result in fit_results
    )
    if not fit_ok:
        _append_phase("fit_terminal_partial", results=fit_results)
        return {
            "state": "failed_fit",
            "fit_results": fit_results,
            "runtime_validation_sha256": runtime_validation_sha,
            "runtime_profile_sha256": runtime_profile_sha,
            "gpu_inventory": inventory,
        }

    for arm_dir in arm_dirs.values():
        missing = json.loads((arm_dir / "artifact_collection.json").read_text(encoding="utf-8"))[
            "missing"
        ]
        if missing:
            raise WorkerRejected(f"successful fit is missing required artifacts: {missing}")

    _verify_prefit_pair(arm_dirs)
    prediction_results: dict[str, dict[str, Any]] = {}
    for arm, run_name in arm_names.items():
        prediction = _export_predictions(
            arm,
            run_name,
            str(0 if arm == "wmse" else 1),
            manifest_path,
            runtime_validation,
            runtime_validation_sha,
            arm_dirs[arm],
        )
        result = _wait_one(
            prediction["command"],
            prediction["label"],
            prediction["log_path"],
            deadline,
            prediction["env"],
        )
        _COMMANDS[-1]["returncode"] = result.get("returncode")
        prediction_results[arm] = result
        if result.get("returncode") != 0:
            raise WorkerRejected(f"prediction export failed for {arm}")
        prediction["manifest_path"] = str(manifest_path)
        prediction["manifest"] = manifest
        _collect_prediction(prediction, run_name, arm_dirs[arm])
        _append_phase("prediction_collected", arm=arm, output_bytes=_tree_bytes(arm_dirs[arm]))

    contract, profile_receipt = _collect_shared(
        arm_dirs,
        arm_names,
        runtime_validation,
        runtime_profile,
        source_receipt,
        bootstrap_receipt,
        manifest_path,
        fit_results,
        prediction_results,
    )
    _write_json(OUT / "commands.json", {"commands": _COMMANDS})
    run_receipt = {
        "schema": "wnn.loss-pair.gpu-operator-run-receipt.v1",
        "format_version": 1,
        "state": "complete",
        "provider_status": "not_observed_by_worker",
        "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "cache_validation_sha256": runtime_validation_sha,
        "cache_profile_sha256": runtime_profile_sha,
        "training_contract_sha256": _sha256_file(OUT / "training_contract.json"),
        "fit_results": fit_results,
        "gpu_inventory": inventory,
        "device_profile_gate": device_profile_gate,
        "device_profile_gate_sha256": _sha256_file(OUT / "shared" / "device_profile_gate.json"),
        "resource_samples": len(_SAMPLES),
        "max_working_output_bytes": contract["resource_profile"]["max_working_output_bytes"],
        "memory_peak": contract["resource_profile"]["max_cgroup_current_bytes"],
        "profile_input_receipt_sha256": _sha256_file(
            OUT / "shared" / "profile_inputs_receipt.json"
        ),
    }
    _write_json(OUT / "gpu_pair_run_receipt.json", run_receipt)
    _append_phase("worker_complete", contract_sha256=run_receipt["training_contract_sha256"])
    return {"state": "complete", "run_receipt": run_receipt}


def main() -> int:
    bootstrap_path = OUT / "bootstrap_receipt.json"
    status_path = WORK / "final_status.json"
    try:
        bootstrap_receipt = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        result = run_worker(bootstrap_receipt)
        _write_json(
            status_path, {"state": result["state"], "elapsed_seconds": time.monotonic() - _STARTED}
        )
        return 0 if result["state"] in {"complete", "profile_complete_fit_not_admitted"} else 2
    except BaseException as error:
        failure = {
            "state": "failed",
            "elapsed_seconds": time.monotonic() - _STARTED,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            _write_json(WORK / "failure.json", failure)
            _write_json(
                status_path, {"state": "failed", "elapsed_seconds": failure["elapsed_seconds"]}
            )
            _append_phase("worker_failed", error_type=failure["error_type"], error=failure["error"])
        except OSError:
            print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
