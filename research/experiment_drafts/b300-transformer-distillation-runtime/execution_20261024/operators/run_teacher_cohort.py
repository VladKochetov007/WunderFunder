#!/usr/bin/env python3
"""Run one frozen four-fit B300 teacher cohort from a hash-bound JSON plan.

Plan schema ``wnn.pr32.teacher-cohort.v1`` has source {root, commit,
manifest_path, manifest_sha256}, data_root, data_files (relative path -> SHA256),
run_root, and four jobs. Each job has name, config, config_file_sha256,
config_sha256 (canonical normalized config), split_manifest, split_sha256,
profile_receipt, profile_sha256, gpu_uuids (two), cpus (eight), and
max_seconds. The only accepted jobs are t1280_f0, t1792_f0, t1280_f2,
t1792_f2. This is a one-shot controller: there is no retry or queue mode.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GIB = 1 << 30
EXPECTED_JOBS = {"t1280_f0", "t1792_f0", "t1280_f2", "t1792_f2"}
DATA_FILES = {"datasets/train.parquet", "datasets/valid.parquet"}
STOP = False
OWNED: dict[int, tuple[subprocess.Popen[Any], int]] = {}
LOGGER = logging.getLogger("teacher_cohort")


def utc_now() -> str:
    """Return a compact UTC timestamp for durable receipts."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Match the canonical JSON digest used by the sequence provenance code."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Atomically replace a progress receipt and sync both file and directory."""
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        sync_dir(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def sync_dir(path: Path) -> None:
    """Persist a directory entry update on Linux."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_event(path: Path, event: dict[str, object]) -> None:
    """Append one durable event without rewriting prior execution history."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject non-object values."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def run_checked(
    argv: list[str], *, cwd: Path, env: dict[str, str], timeout: int = 120
) -> str:
    """Run a preflight command without a shell and return its stdout."""
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def config_details(source: Path, config_path: Path, mode: str) -> dict[str, Any]:
    """Use the frozen config loader to get normalized settings and their exact digest."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source / "src")
    if mode.endswith("bf16x9"):
        env["CUBLAS_EMULATE_SINGLE_PRECISION"] = "1"
    else:
        env.pop("CUBLAS_EMULATE_SINGLE_PRECISION", None)
    code = (
        "import json,sys; from pathlib import Path; "
        "from wnn.training.sequence_config import load_config; "
        "from wnn.training.sequence_provenance import canonical_sha256; "
        "v=load_config(Path(sys.argv[1])); "
        "print(json.dumps({'config':v,'sha256':canonical_sha256(v)},allow_nan=False))"
    )
    output = run_checked(
        [sys.executable, "-B", "-c", code, str(config_path)], cwd=source, env=env
    )
    value = json.loads(output)
    if not isinstance(value, dict) or not isinstance(value.get("config"), dict):
        raise TypeError(f"canonical config loader returned invalid data: {config_path}")
    return value


def process_source_files(source: Path) -> dict[str, str]:
    """Reproduce the trainer's source_identity file set and digest."""
    paths = sorted((source / "src").rglob("*.py"))
    entrypoint = (source / "scripts" / "train_sequence.py").resolve()
    if entrypoint not in paths:
        paths.append(entrypoint)
    mapping = {path.relative_to(source).as_posix(): sha256_file(path) for path in paths}
    return mapping


def verify_bound_source(source: Path, job: dict[str, Any]) -> None:
    """Recheck frozen source/config/split/profile bytes immediately before each phase."""
    manifest_path = Path(job["source_manifest_path"])
    if sha256_file(manifest_path) != job["source_manifest_sha256"]:
        raise ValueError("frozen transport manifest changed after preflight")
    manifest = read_json(manifest_path)
    if manifest.get("commit") != job["source_commit"]:
        raise ValueError("frozen source commit changed after preflight")
    members = manifest.get("files")
    if not isinstance(members, dict):
        raise TypeError("frozen source manifest has no file digest map")
    for name, expected in members.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source member: {name!r}")
        path = source / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen source member changed before phase: {name}")
    if canonical_sha256(process_source_files(source)) != job["source_sha256"]:
        raise ValueError("runtime trainer source identity changed after preflight")
    for key, path_key in (
        ("config_file_sha256", "config"),
        ("profile_sha256", "profile"),
        ("split_sha256", "split"),
    ):
        if sha256_file(Path(job[path_key])) != job[key]:
            raise ValueError(f"{path_key} changed after preflight")


def validate_frozen_recipe(
    source: Path, name: str, actual: dict[str, Any], precision_mode: str
) -> None:
    """Allow only the explicitly approved world/batch/precision/compiler changes."""
    frozen = (
        source
        / "research/experiment_drafts/b300-transformer-distillation-runtime"
        / "large_teacher_r3_muon30"
        / "configs"
        / f"{name}.json"
    )
    expected = read_json(frozen)
    expected["batch_size"] = 8
    expected["required_world_size"] = 2
    expected["precision_mode"] = precision_mode
    model = expected["model"]
    model["checkpoint_blocks"] = False
    model["attention_backend"] = "efficient"
    model["compile_blocks"] = True
    if expected != actual:
        raise ValueError(
            f"{name}: config has unapproved deviation from frozen R3 recipe"
        )


def verify_source(plan: dict[str, Any]) -> tuple[Path, str, dict[str, str]]:
    """Verify the transported Git archive, source files and trainer identity."""
    source = Path(plan["source"]["root"]).resolve(strict=True)
    manifest_path = Path(plan["source"]["manifest_path"]).resolve(strict=True)
    manifest_hash = str(plan["source"]["manifest_sha256"])
    if sha256_file(manifest_path) != manifest_hash:
        raise ValueError("source transport manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    commit = str(plan["source"]["commit"])
    if manifest.get("commit") != commit:
        raise ValueError("source transport manifest commit mismatch")
    members = manifest.get("files")
    if not isinstance(members, dict) or not members:
        raise ValueError("source transport manifest lacks its file digest map")
    for name, expected in members.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source manifest member: {name!r}")
        path = source / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source member differs from frozen manifest: {name}")
    for required in (
        "scripts/train_sequence.py",
        "scripts/predict_sequence.py",
        "scripts/score_sequence.py",
    ):
        if required not in members:
            raise ValueError(f"frozen source archive omits {required}")
    files = process_source_files(source)
    source_hash = canonical_sha256(files)
    return source, source_hash, files


def _nvidia_rows(env: dict[str, str]) -> list[dict[str, Any]]:
    """Read current GPU identity and free-memory state using NVIDIA's CLI."""
    output = run_checked(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=Path.cwd(),
        env=env,
        timeout=20,
    )
    rows = []
    for line in output.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 6:
            raise ValueError(f"unparseable nvidia-smi row: {line}")
        rows.append(
            {
                "uuid": parts[0],
                "name": parts[1],
                "total_mib": int(parts[2]),
                "used_mib": int(parts[3]),
                "free_mib": int(parts[4]),
                "utilization": int(parts[5]),
            }
        )
    return rows


def _compute_apps(env: dict[str, str]) -> list[tuple[str, int]]:
    """Return GPU UUID and PID pairs for current CUDA compute processes."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        cwd=Path.cwd(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi compute-app query failed: {result.stderr.strip()}"
        )
    entries = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 2 and parts[1].isdigit():
            entries.append((parts[0], int(parts[1])))
        elif line.strip():
            raise ValueError(f"unparseable nvidia-smi compute-app row: {line}")
    return entries


def _disk_fstype(path: Path) -> str:
    """Find the mounted filesystem type containing a path from /proc/self/mountinfo."""
    best_len, best_type = -1, ""
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        if len(fields) < 5:
            continue
        mount = Path(fields[4].replace("\\040", " "))
        try:
            path.relative_to(mount)
        except ValueError:
            continue
        if len(str(mount)) > best_len:
            best_len = len(str(mount))
            best_type = right.split()[0]
    if not best_type:
        raise ValueError(f"could not determine filesystem type for {path}")
    return best_type


def preflight(plan: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Validate all source, data, device, CPU, memory, disk and output gates."""
    if sys.platform != "linux":
        raise RuntimeError("teacher cohort requires the approved Linux B300 host")
    if plan.get("schema") != "wnn.pr32.teacher-cohort.v1":
        raise ValueError("unsupported teacher cohort plan schema")
    source, source_hash, source_files = verify_source(plan)
    commit = str(plan["source"]["commit"])
    data_root = Path(plan["data_root"]).resolve(strict=True)
    files = plan.get("data_files")
    if not isinstance(files, dict) or set(files) != DATA_FILES:
        raise ValueError(
            "data_files must pin exactly datasets/train.parquet and datasets/valid.parquet"
        )
    data_hashes = {}
    for relative, expected in sorted(files.items()):
        path = data_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required dataset is not a regular file: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"dataset SHA256 mismatch: {relative}")
        data_hashes[relative] = actual

    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 4:
        raise ValueError("exactly four approved jobs are required")
    if {job.get("name") for job in jobs if isinstance(job, dict)} != EXPECTED_JOBS:
        raise ValueError("job names must be the four frozen R3 width/fold cells")
    all_gpus: list[str] = []
    all_cpus: list[int] = []
    prepared = []
    for job in jobs:
        if not isinstance(job, dict):
            raise TypeError("each job must be an object")
        name = job["name"]
        config_path = Path(job["config"]).resolve(strict=True)
        split_path = Path(job["split_manifest"]).resolve(strict=True)
        profile_path = Path(job["profile_receipt"]).resolve(strict=True)
        if sha256_file(config_path) != job["config_file_sha256"]:
            raise ValueError(f"{name}: raw config SHA256 mismatch")
        if sha256_file(split_path) != job["split_sha256"]:
            raise ValueError(f"{name}: split manifest SHA256 mismatch")
        split = read_json(split_path)
        config_raw = read_json(config_path)
        mode = str(config_raw.get("precision_mode", "fp32"))
        if mode != "bf16_bf16x9":
            raise ValueError(
                f"{name}: production compiled cohort requires bf16_bf16x9 for every arm"
            )
        normalized = config_details(source, config_path, mode)
        if normalized["sha256"] != job["config_sha256"]:
            raise ValueError(f"{name}: canonical config SHA256 mismatch")
        config = normalized["config"]
        validate_frozen_recipe(source, name, config_raw, mode)
        fold = int(config["expected_fold"])
        if split.get("fold") != fold or name.endswith(f"f{fold}") is False:
            raise ValueError(f"{name}: config, split and cell fold disagree")
        if config["required_world_size"] != 2 or config["epochs"] != 30:
            raise ValueError(f"{name}: expected an approved W2 30-epoch fit")
        expected_hidden = 1280 if name.startswith("t1280_") else 1792
        expected_layers, expected_heads = (
            (32, 20) if expected_hidden == 1280 else (40, 28)
        )
        expected_seed = 101001 if fold == 0 else 101002
        model = config["model"]
        if (
            model["architecture"] != "transformer"
            or model["hidden"] != expected_hidden
            or model["layers"] != expected_layers
            or model["heads"] != expected_heads
            or model["context_rows"] != 1024
        ):
            raise ValueError(
                f"{name}: model architecture differs from the frozen R3 scale"
            )
        if config["seed"] != expected_seed or config["required_device"] != "B300":
            raise ValueError(f"{name}: seed/device requirement differs from frozen R3")
        if (
            config["optimizer"] != "muon"
            or config["lr"] != 0.0003
            or config["schedule"] != "warmup_cosine_v2"
            or config["min_lr_ratio"] != 0.1
            or config["accumulate_chunks"] != 8
            or config["chunk_rows"] != 512
            or config["batch_size"] != 8
        ):
            raise ValueError(f"{name}: optimizer/update recipe differs from frozen R3")
        if config["batch_size"] * config["required_world_size"] != 16:
            raise ValueError(f"{name}: global sequence batch must remain 16")
        if config["hint_weight"] != 0 or config["kd_weight"] != 0:
            raise ValueError(f"{name}: initial teacher fit must be labels-only")
        if sha256_file(profile_path) != job["profile_sha256"]:
            raise ValueError(f"{name}: profile receipt SHA256 mismatch")
        profile = read_json(profile_path)
        identity = profile.get("identity")
        if (
            profile.get("receipt_schema") != "wnn.sequence-profile.v1"
            or profile.get("status") != "profile_complete"
            or profile.get("approved") is not True
            or not isinstance(identity, dict)
        ):
            raise ValueError(f"{name}: profile is not approved and complete")
        if identity.get("config_sha256") != job["config_sha256"]:
            raise ValueError(f"{name}: profile does not bind this normalized config")
        if (
            identity.get("source_sha256") != source_hash
            or identity.get("source_files") != source_files
        ):
            raise ValueError(
                f"{name}: profile source identity differs from frozen source"
            )
        data_identity = identity.get("data")
        if not isinstance(data_identity, dict) or (
            data_identity.get("split_sha256") != job["split_sha256"]
            or data_identity.get("fold") != fold
        ):
            raise ValueError(f"{name}: profile split identity mismatch")
        gpus = job.get("gpu_uuids")
        cpus = job.get("cpus")
        if (
            not isinstance(gpus, list)
            or len(gpus) != 2
            or any(
                not isinstance(value, str) or not value.startswith("GPU-")
                for value in gpus
            )
        ):
            raise ValueError(f"{name}: exactly two explicit GPU UUIDs are required")
        if (
            not isinstance(cpus, list)
            or len(cpus) != 8
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
        ):
            raise ValueError(f"{name}: exactly eight CPU IDs are required")
        runtime = identity.get("runtime")
        devices = runtime.get("devices") if isinstance(runtime, dict) else None
        profile_uuids = (
            [str(item.get("uuid", "")).removeprefix("GPU-").lower() for item in devices]
            if isinstance(devices, list)
            else []
        )
        if profile_uuids != [value.removeprefix("GPU-").lower() for value in gpus]:
            raise ValueError(f"{name}: profile GPU order differs from assigned UUIDs")
        if runtime.get("world_size") != 2:
            raise ValueError(f"{name}: profile must bind world size two")
        if (
            type(job.get("max_seconds")) is not int
            or not 0 < job["max_seconds"] <= 1209600
        ):
            raise ValueError(f"{name}: max_seconds must be in 1..1209600")
        all_gpus.extend(gpus)
        all_cpus.extend(cpus)
        prepared.append(
            {
                "name": name,
                "config": config_path,
                "split": split_path,
                "profile": profile_path,
                "profile_identity": identity,
                "source_sha256": source_hash,
                "source_commit": commit,
                "source_manifest_path": str(
                    Path(plan["source"]["manifest_path"]).resolve()
                ),
                "source_manifest_sha256": plan["source"]["manifest_sha256"],
                "config_sha256": job["config_sha256"],
                "config_file_sha256": job["config_file_sha256"],
                "split_sha256": job["split_sha256"],
                "profile_sha256": job["profile_sha256"],
                "gpus": gpus,
                "cpus": cpus,
                "max_seconds": job["max_seconds"],
                "precision_mode": mode,
            }
        )
    for fold in (0, 2):
        split_hashes = {
            job["split_sha256"] for job in jobs if job["name"].endswith(f"f{fold}")
        }
        if len(split_hashes) != 1:
            raise ValueError(
                f"fold {fold}: width comparison must use one shared split identity"
            )
    if len(set(all_gpus)) != 8:
        raise ValueError("all eight assigned GPU UUIDs must be unique")
    if len(set(all_cpus)) != 32:
        raise ValueError("all 32 assigned CPU IDs must be unique")
    affinity = set(os.sched_getaffinity(0))
    if not set(all_cpus).issubset(affinity):
        raise ValueError("the supervisor cpuset does not contain all assigned CPUs")

    minimum_ram = max(
        512 * GIB, int(plan.get("minimum_mem_available_bytes", 512 * GIB))
    )
    mem_fields = {
        row.split(":", 1)[0]: int(row.split()[1]) * 1024
        for row in Path("/proc/meminfo").read_text().splitlines()
        if row.startswith(("MemAvailable:", "MemTotal:"))
    }
    if mem_fields.get("MemAvailable", 0) < minimum_ram:
        raise RuntimeError("host MemAvailable is below the 512 GiB cohort gate")
    minimum_disk = max(200 * GIB, int(plan.get("minimum_disk_free_bytes", 200 * GIB)))
    tmp_root = Path(
        plan.get(
            "tmp_root",
            Path(plan["data_root"]).resolve().parent / "tmp" / "pr32-r3-20261024-r2",
        )
    ).resolve()
    if tmp_root.exists():
        raise FileExistsError(
            "tmp_root must be fresh to keep compiler scratch owned by this cohort"
        )
    storage_probe = tmp_root.parent
    while not storage_probe.exists() and storage_probe != storage_probe.parent:
        storage_probe = storage_probe.parent
    if not storage_probe.is_dir() or _disk_fstype(storage_probe) in {
        "tmpfs",
        "ramfs",
        "devtmpfs",
    }:
        raise RuntimeError("compiler TMPDIR must be on physical storage")
    if shutil.disk_usage(storage_probe).free < minimum_disk:
        raise RuntimeError(
            "physical scratch disk free space is below the 200 GiB cohort gate"
        )
    run_root = Path(plan["run_root"]).resolve()
    if run_root.exists() or not run_root.parent.is_dir():
        raise FileExistsError("run_root must be a fresh path with an existing parent")
    fs_type = _disk_fstype(run_root.parent)
    if fs_type in {"tmpfs", "ramfs", "devtmpfs"}:
        raise RuntimeError(f"run_root must use physical storage, found {fs_type}")
    if shutil.disk_usage(run_root.parent).free < minimum_disk:
        raise RuntimeError("physical disk free space is below the 200 GiB cohort gate")
    env = os.environ.copy()
    gpu_rows = _nvidia_rows(env)
    by_uuid = {row["uuid"]: row for row in gpu_rows}
    if len(by_uuid) != len(gpu_rows) or not set(all_gpus).issubset(by_uuid):
        raise RuntimeError("assigned GPU UUIDs are missing or duplicated on this host")
    selected = [by_uuid[uuid] for uuid in all_gpus]
    if any("B300" not in row["name"].upper() for row in selected):
        raise RuntimeError("every assigned device must identify as B300")
    if any(
        row["used_mib"] > 1024
        or row["free_mib"] < int(row["total_mib"] * 0.90)
        or row["utilization"] > 5
        for row in selected
    ):
        raise RuntimeError("assigned B300 devices are not idle and memory-free")
    if any(uuid in set(all_gpus) for uuid, _ in _compute_apps(env)):
        raise RuntimeError("an assigned B300 already has a CUDA compute process")
    if shutil.which("taskset") is None:
        raise RuntimeError("taskset is required for the pinned CPU layout")
    return run_root, {
        "source_root": str(source),
        "source_sha256": source_hash,
        "source_files": source_files,
        "data_hashes": data_hashes,
        "mem_available_bytes": mem_fields["MemAvailable"],
        "disk_free_bytes": shutil.disk_usage(run_root.parent).free,
        "filesystem": fs_type,
        "tmp_root": str(tmp_root),
        "gpu_rows": selected,
        "jobs": prepared,
    }


def _proc_identity(pid: int) -> dict[str, int | str] | None:
    """Read stable Linux process identity fields from procfs."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    close = raw.rfind(")")
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        return None
    return {
        "state": fields[0],
        "pgrp": int(fields[2]),
        "session": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def group_members(pgid: int, session: int) -> list[int]:
    """List only processes still inside an owned process group and session."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        identity = _proc_identity(int(entry.name))
        if identity and identity["pgrp"] == pgid and identity["session"] == session:
            members.append(int(entry.name))
    return members


def _owned_group(proc: subprocess.Popen[Any], start_ticks: int) -> bool:
    """Prove the original session still exists before signaling its process group."""
    identity = _proc_identity(proc.pid)
    if identity is not None:
        return (
            identity["start_ticks"] == start_ticks
            and identity["pgrp"] == proc.pid
            and identity["session"] == proc.pid
        )
    # The session ID cannot be reused while any member of this session exists.
    return bool(group_members(proc.pid, proc.pid))


def terminate_owned(
    proc: subprocess.Popen[Any], start_ticks: int, grace: float = 20.0
) -> None:
    """Stop only the child process group created by this supervisor."""
    if not _owned_group(proc, start_ticks):
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    until = time.monotonic() + grace
    while time.monotonic() < until and group_members(proc.pid, proc.pid):
        time.sleep(0.2)
    if group_members(proc.pid, proc.pid) and _owned_group(proc, start_ticks):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)


def env_for(
    job: dict[str, Any], source: Path, *, scoring: bool = False
) -> dict[str, str]:
    """Build an isolated runtime environment from the approved profile identity."""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(source / "src"),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCH_LOGS": "graph_breaks,recompiles",
            "TMPDIR": job["tmp_root"],
            "TORCHINDUCTOR_CACHE_DIR": str(Path(job["tmp_root"]) / "torchinductor"),
        }
    )
    runtime = job["profile_identity"]["runtime"]
    workspace = runtime.get("cublas_workspace_config")
    if workspace is None:
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        env["CUBLAS_WORKSPACE_CONFIG"] = str(workspace)
    if job["precision_mode"].endswith("bf16x9"):
        env["CUBLAS_EMULATE_SINGLE_PRECISION"] = "1"
    else:
        env.pop("CUBLAS_EMULATE_SINGLE_PRECISION", None)
    env["CUDA_VISIBLE_DEVICES"] = "" if scoring else ",".join(job["gpus"])
    return env


def _launch(
    source: Path,
    run_root: Path,
    job: dict[str, Any],
    phase: str,
    argv: list[str],
) -> dict[str, Any]:
    """Launch one exclusive, affinity-pinned phase in a new supervised session."""
    if STOP:
        raise InterruptedError("cohort cancellation requested before phase admission")
    verify_bound_source(source, job)
    name = job["name"]
    log_path = run_root / "logs" / f"{name}.{phase}.log"
    if log_path.exists():
        raise FileExistsError(f"phase log already exists: {log_path}")
    cpus = job["cpus"][:1] if phase == "score" else job["cpus"]
    cmd = ["taskset", "--cpu-list", ",".join(map(str, cpus)), *argv]
    env = env_for(job, source, scoring=(phase == "score"))
    if STOP:
        raise InterruptedError("cohort cancellation requested before process spawn")
    with log_path.open("xb") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + job["max_seconds"]
    start_ticks = None
    for _ in range(50):
        identity = _proc_identity(proc.pid)
        if identity:
            start_ticks = int(identity["start_ticks"])
            if identity["pgrp"] != proc.pid or identity["session"] != proc.pid:
                terminate_owned(proc, start_ticks)
                raise RuntimeError(f"{name}/{phase}: child session identity mismatch")
            break
        time.sleep(0.02)
    if start_ticks is None:
        terminate_owned(proc, 0)
        raise RuntimeError(f"{name}/{phase}: launched process did not appear in procfs")
    OWNED[proc.pid] = (proc, start_ticks)
    launch_path = run_root / "receipts" / f"{name}.{phase}.launch.json"
    receipt_written = False
    try:
        if launch_path.exists():
            raise FileExistsError(f"phase launch receipt already exists: {launch_path}")
        atomic_json(
            launch_path,
            {
                "schema": "wnn.pr32.teacher-phase-launch.v1",
                "job": name,
                "phase": phase,
                "started_utc": utc_now(),
                "started_unix": time.time(),
                "pid": proc.pid,
                "pgid": proc.pid,
                "session": proc.pid,
                "start_ticks": start_ticks,
                "argv": cmd,
                "log": str(log_path),
                "source_commit": job["source_commit"],
                "source_sha256": job["source_sha256"],
                "config_sha256": job["config_sha256"],
                "split_sha256": job["split_sha256"],
                "profile_sha256": job["profile_sha256"],
                "gpu_uuids": [] if phase == "score" else job["gpus"],
                "cpus": cpus,
            },
        )
        receipt_written = True
    finally:
        if not receipt_written:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                terminate_owned(proc, start_ticks)
            OWNED.pop(proc.pid, None)
    return {
        "proc": proc,
        "phase": phase,
        "argv": cmd,
        "log": str(log_path),
        "started_unix": time.time(),
        "started_monotonic": time.monotonic(),
        "deadline_monotonic": deadline,
        "start_ticks": start_ticks,
        "pgid": proc.pid,
        "session": proc.pid,
    }


def _fit_argv(source: Path, job: dict[str, Any], output: Path) -> list[str]:
    """Build the canonical full-fit command from its frozen profile/config contract."""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        str(source / "scripts/train_sequence.py"),
        "--config",
        str(job["config"]),
        "--data-root",
        str(Path(job["data_root"]).resolve()),
        "--split-manifest",
        str(job["split"]),
        "--output",
        str(output),
        "--profile-receipt",
        str(job["profile"]),
    ]


def _eval_argv(
    source: Path, job: dict[str, Any], fit_dir: Path, output: Path
) -> list[str]:
    """Build the canonical two-GPU held-out endpoint prediction command."""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        str(source / "scripts/predict_sequence.py"),
        "--checkpoint",
        str(fit_dir / "model.pt"),
        "--data-root",
        str(Path(job["data_root"]).resolve()),
        "--split-manifest",
        str(job["split"]),
        "--output",
        str(output),
        "--membership",
        "evaluation",
    ]


def _score_argv(
    source: Path, job: dict[str, Any], evaluation: Path, score_path: Path
) -> list[str]:
    """Build the canonical CPU scoring command for the held-out endpoint."""
    return [
        sys.executable,
        str(source / "scripts/score_sequence.py"),
        "--data-root",
        str(Path(job["data_root"]).resolve()),
        "--split-manifest",
        str(job["split"]),
        "--predictions",
        str(evaluation),
        "--output",
        str(score_path),
    ]


def validate_fit(job: dict[str, Any], fit_dir: Path) -> dict[str, Any]:
    """Require a fixed 30-epoch checkpoint bound to the approved profile."""
    model, terminal = fit_dir / "model.pt", fit_dir / "terminal_status.json"
    if not model.is_file() or not terminal.is_file():
        raise ValueError(
            "canonical fit output is missing model.pt or terminal_status.json"
        )
    status = read_json(terminal)
    if (
        status.get("status") != "completed"
        or status.get("completed_epochs") != 30
        or status.get("identity") != job["profile_identity"]
        or status.get("model_sha256") != sha256_file(model)
    ):
        raise ValueError(
            "fit terminal receipt does not match approved fixed-final identity"
        )
    if type(status.get("global_updates")) is not int or status["global_updates"] <= 0:
        raise ValueError("fit terminal receipt lacks positive optimizer update count")
    return {
        "model_sha256": status["model_sha256"],
        "terminal_status_sha256": sha256_file(terminal),
    }


def validate_evaluation(
    job: dict[str, Any], evaluation: Path, model_sha256: str
) -> dict[str, Any]:
    """Check complete held-out coverage and exact checkpoint/split binding."""
    manifest_path = evaluation / "manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != "sequence-predictions-v1"
        or manifest.get("complete") is not True
        or manifest.get("membership") != "evaluation"
        or manifest.get("split_sha256") != job["split_sha256"]
    ):
        raise ValueError(
            "prediction manifest schema, membership or split binding is invalid"
        )
    producer = manifest.get("producer")
    entries = manifest.get("sequences")
    if (
        not isinstance(producer, dict)
        or producer.get("checkpoint_sha256") != model_sha256
        or producer.get("split_sha256") != job["split_sha256"]
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("prediction manifest does not bind the completed fit")
    members = [(entry.get("split"), entry.get("row_group")) for entry in entries]
    if len(set(members)) != len(entries) or any(
        split != "valid" for split, _ in members
    ):
        raise ValueError(
            "prediction membership is duplicated or outside held-out valid data"
        )
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "sequences": len(entries),
        "checkpoint_sha256": model_sha256,
    }


def validate_score(
    job: dict[str, Any], score_path: Path, eval_receipt: dict[str, Any]
) -> dict[str, Any]:
    """Check the canonical score receipt against its exact prediction artifact."""
    score = read_json(score_path)
    if (
        score.get("split_sha256") != job["split_sha256"]
        or score.get("predictions_manifest_sha256") != eval_receipt["manifest_sha256"]
        or score.get("sequences") != eval_receipt["sequences"]
    ):
        raise ValueError(
            "score receipt does not bind the exact held-out prediction manifest"
        )
    producer = score.get("producer")
    if (
        not isinstance(producer, dict)
        or producer.get("checkpoint_sha256") != eval_receipt["checkpoint_sha256"]
    ):
        raise ValueError("score producer checkpoint identity mismatch")
    return {"score_sha256": sha256_file(score_path), "score": score.get("score")}


def _phase_complete(
    run_root: Path, job: dict[str, Any], phase: dict[str, Any], code: int
) -> None:
    """Record immutable walltime/return-code evidence for one completed process phase."""
    path = run_root / "receipts" / f"{job['name']}.{phase['phase']}_timing.json"
    receipt = {
        "schema": "wnn.pr32.teacher-phase-timing.v1",
        "job": job["name"],
        "phase": phase["phase"],
        "started_utc_unix": phase["started_unix"],
        "elapsed_seconds": max(0.0, time.monotonic() - phase["started_monotonic"]),
        "returncode": code,
        "argv": phase["argv"],
        "log": phase["log"],
        "pid": phase["proc"].pid,
        "pgid": phase["pgid"],
        "start_ticks": phase["start_ticks"],
    }
    atomic_json(path, receipt)


def _set_status(
    state: dict[str, Any],
    path: Path,
    events: Path,
    job: dict[str, Any],
    **fields: object,
) -> None:
    """Update and durably publish one job's state transition."""
    record = state["jobs"][job["name"]]
    record.update(fields)
    state["updated_utc"] = utc_now()
    atomic_json(path, state)
    append_event(
        events,
        {"time_utc": state["updated_utc"], "job": job["name"], **fields},
    )


def _signal_handler(signum: int, _frame: object) -> None:
    """Request orderly cancellation of only this controller's own process groups."""
    global STOP
    STOP = True


def cleanup_owned() -> None:
    """Best-effort termination of only process groups launched by this run."""
    for proc, start_ticks in list(OWNED.values()):
        try:
            terminate_owned(proc, start_ticks)
        except (OSError, subprocess.SubprocessError):
            LOGGER.exception("failed to clean up owned process group %s", proc.pid)
        finally:
            OWNED.pop(proc.pid, None)


def execute(plan_path: Path) -> int:
    """Preflight, run the four fit/evaluation/score chains, and preserve all evidence."""
    plan_path = plan_path.resolve(strict=True)
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    if not isinstance(plan, dict):
        raise TypeError("cohort plan must be a JSON object")
    run_root, checked = preflight(plan)
    if STOP:
        return 130
    tmp_root = Path(checked["tmp_root"])
    tmp_root.parent.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(mode=0o750)
    run_root.mkdir(mode=0o750)
    sync_dir(run_root.parent)
    (run_root / "logs").mkdir()
    (run_root / "receipts").mkdir()
    for job in checked["jobs"]:
        job["data_root"] = str(Path(plan["data_root"]).resolve())
        job["tmp_root"] = str(tmp_root)
        job_dir = run_root / job["name"]
        job_dir.mkdir()
        # Canonical trainer and predictor create these phase directories exclusively.
        if (job_dir / "fit").exists() or (job_dir / "evaluation").exists():
            raise FileExistsError(f"existing output under {job_dir}")
    with (run_root / "plan.json").open("xb") as stream:
        stream.write(plan_raw)
        stream.flush()
        os.fsync(stream.fileno())
    state_path = run_root / "STATE.json"
    events_path = run_root / "events.jsonl"
    controller_identity = _proc_identity(os.getpid())
    if controller_identity is None:
        raise RuntimeError("could not bind controller PID to a Linux start tick")
    state: dict[str, Any] = {
        "schema": "wnn.pr32.teacher-cohort-state.v1",
        "status": "running",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "source_commit": plan["source"]["commit"],
        "source_sha256": checked["source_sha256"],
        "controller_pid": os.getpid(),
        "controller_start_ticks": controller_identity["start_ticks"],
        "data_hashes": checked["data_hashes"],
        "preflight": {
            "mem_available_bytes": checked["mem_available_bytes"],
            "disk_free_bytes": checked["disk_free_bytes"],
            "filesystem": checked["filesystem"],
            "gpu_rows": checked["gpu_rows"],
        },
        "jobs": {
            job["name"]: {
                "status": "pending",
                "phase": "pending",
                "config_sha256": plan_job_hash(plan, job["name"]),
                "split_sha256": plan_job_split(plan, job["name"]),
                "profile_sha256": plan_job_profile(plan, job["name"]),
                "gpu_uuids": job["gpus"],
                "cpus": job["cpus"],
                "max_seconds": job["max_seconds"],
            }
            for job in checked["jobs"]
        },
    }
    atomic_json(state_path, state)
    append_event(events_path, {"time_utc": utc_now(), "event": "preflight_passed"})
    source = Path(checked["source_root"])
    active: dict[str, dict[str, Any]] = {}
    jobs_by_name = {job["name"]: job for job in checked["jobs"]}
    failures = 0
    for job in checked["jobs"]:
        if STOP:
            failures += 1
            _set_status(
                state,
                state_path,
                events_path,
                job,
                status="cancelled",
                phase="pending",
                error="cohort cancellation requested before fit admission",
            )
            continue
        job["deadline_monotonic"] = time.monotonic() + job["max_seconds"]
        job["started_unix"] = time.time()
        job["stage"] = "fit"
        _set_status(
            state,
            state_path,
            events_path,
            job,
            status="running",
            phase="fit",
            started_utc=utc_now(),
        )
        try:
            fit_dir = run_root / job["name"] / "fit"
            active[job["name"]] = _launch(
                source, run_root, job, "fit", _fit_argv(source, job, fit_dir)
            )
        except Exception as error:
            failures += 1
            LOGGER.warning(
                "cell %s failed to launch its fit process: %s",
                job["name"],
                error,
                exc_info=True,
            )
            _set_status(
                state,
                state_path,
                events_path,
                job,
                status="failed",
                phase="fit",
                error=str(error),
            )

    last_heartbeat = time.monotonic()
    while active:
        if STOP:
            for name, phase in list(active.items()):
                terminate_owned(phase["proc"], phase["start_ticks"])
                OWNED.pop(phase["proc"].pid, None)
                _set_status(
                    state,
                    state_path,
                    events_path,
                    jobs_by_name[name],
                    status="cancelled",
                    phase=phase["phase"],
                    error="controller received termination signal",
                )
                active.pop(name, None)
                failures += 1
            break
        now = time.monotonic()
        if now - last_heartbeat >= 5:
            state["updated_utc"] = utc_now()
            atomic_json(state_path, state)
            append_event(
                events_path,
                {
                    "time_utc": state["updated_utc"],
                    "event": "heartbeat",
                    "active_jobs": sorted(active),
                },
            )
            last_heartbeat = now
        for name, phase in list(active.items()):
            job = jobs_by_name[name]
            proc = phase["proc"]
            if now >= job["deadline_monotonic"] or now >= phase["deadline_monotonic"]:
                terminate_owned(proc, phase["start_ticks"])
                OWNED.pop(proc.pid, None)
                failures += 1
                active.pop(name, None)
                _set_status(
                    state,
                    state_path,
                    events_path,
                    job,
                    status="failed",
                    phase=phase["phase"],
                    error="job phase timeout",
                )
                continue
            code = proc.poll()
            if code is None:
                continue
            # torchrun must leave no worker behind before a later phase can use the same GPUs.
            linger = time.monotonic() + 2.0
            while (
                group_members(phase["pgid"], phase["session"])
                and time.monotonic() < linger
            ):
                time.sleep(0.05)
            if group_members(phase["pgid"], phase["session"]):
                terminate_owned(proc, phase["start_ticks"])
                OWNED.pop(proc.pid, None)
                code = code or 1
            else:
                OWNED.pop(proc.pid, None)
            try:
                _phase_complete(run_root, job, phase, int(code))
                if code != 0:
                    raise RuntimeError(f"{phase['phase']} exited with status {code}")
                fit_dir = run_root / name / "fit"
                eval_dir = run_root / name / "evaluation"
                if phase["phase"] == "fit":
                    fit_info = validate_fit(job, fit_dir)
                    atomic_json(run_root / "receipts" / f"{name}.fit.json", fit_info)
                    job["fit_info"] = fit_info
                    job["stage"] = "evaluation"
                    _set_status(
                        state,
                        state_path,
                        events_path,
                        job,
                        phase="evaluation",
                        fit=fit_info,
                    )
                    active[name] = _launch(
                        source,
                        run_root,
                        job,
                        "evaluation",
                        _eval_argv(source, job, fit_dir, eval_dir),
                    )
                elif phase["phase"] == "evaluation":
                    eval_info = validate_evaluation(
                        job, eval_dir, job["fit_info"]["model_sha256"]
                    )
                    atomic_json(
                        run_root / "receipts" / f"{name}.evaluation.json", eval_info
                    )
                    job["evaluation_info"] = eval_info
                    job["stage"] = "score"
                    _set_status(
                        state,
                        state_path,
                        events_path,
                        job,
                        phase="score",
                        evaluation=eval_info,
                    )
                    score_path = fit_dir / "score.json"
                    active[name] = _launch(
                        source,
                        run_root,
                        job,
                        "score",
                        _score_argv(source, job, eval_dir, score_path),
                    )
                else:
                    score_info = validate_score(
                        job, fit_dir / "score.json", job["evaluation_info"]
                    )
                    atomic_json(
                        run_root / "receipts" / f"{name}.score.json", score_info
                    )
                    active.pop(name, None)
                    _set_status(
                        state,
                        state_path,
                        events_path,
                        job,
                        status="completed",
                        phase="complete",
                        score=score_info,
                    )
            except Exception as error:
                failures += 1
                active.pop(name, None)
                LOGGER.warning(
                    "cell %s failed in phase %s: %s",
                    name,
                    phase["phase"],
                    error,
                    exc_info=True,
                )
                _set_status(
                    state,
                    state_path,
                    events_path,
                    job,
                    status="failed",
                    phase=phase["phase"],
                    error=f"{type(error).__name__}: {error}",
                )
        if active:
            time.sleep(0.5)
    state["status"] = (
        "complete"
        if failures == 0
        and all(value["status"] == "completed" for value in state["jobs"].values())
        else "partial"
    )
    state["updated_utc"] = utc_now()
    atomic_json(state_path, state)
    append_event(
        events_path,
        {
            "time_utc": state["updated_utc"],
            "event": "cohort_finished",
            "status": state["status"],
        },
    )
    return 0 if state["status"] == "complete" else 1


def plan_job(plan: dict[str, Any], name: str) -> dict[str, Any]:
    """Find one named approved job in the immutable plan."""
    return next(item for item in plan["jobs"] if item["name"] == name)


def plan_job_hash(plan: dict[str, Any], name: str) -> str:
    """Return the plan-bound canonical configuration hash."""
    return str(plan_job(plan, name)["config_sha256"])


def plan_job_split(plan: dict[str, Any], name: str) -> str:
    """Return the plan-bound split hash."""
    return str(plan_job(plan, name)["split_sha256"])


def plan_job_profile(plan: dict[str, Any], name: str) -> str:
    """Return the plan-bound approved profile hash."""
    return str(plan_job(plan, name)["profile_sha256"])


def main() -> int:
    """Run the explicitly supplied cohort plan once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        return execute(args.plan)
    except Exception as error:
        print(
            f"teacher cohort preflight/controller failure: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        LOGGER.exception("teacher cohort stopped at its top-level boundary")
        return 2
    finally:
        cleanup_owned()


if __name__ == "__main__":
    atexit.register(cleanup_owned)
    raise SystemExit(main())
