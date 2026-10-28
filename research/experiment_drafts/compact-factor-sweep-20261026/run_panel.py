#!/usr/bin/env python3
"""One-shot runner for the frozen four-fit compact Transformer panel.

Run once with ``python run_panel.py --plan /data/experiment/plan.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import pwd
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "0a70b4ab96c1810484544f382df4bcf0c08e8fd4"
CELLS = {"ctl_f0": 0, "reg_f0": 0, "ctl_f2": 2, "reg_f2": 2}
GIB = 1 << 30
STOP = threading.Event()
JSON = dict[str, Any]  # Any is confined to validated heterogeneous JSON boundaries.


def sha(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canon(value: object) -> str:
    """Hash canonical JSON with the sequence provenance convention."""
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def read(path: Path) -> JSON:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: object) -> None:
    """Atomically persist a controller-owned receipt."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def source_files(source: Path) -> tuple[str, dict[str, str]]:
    """Match the trainer's source identity without importing model code."""
    paths = sorted((source / "src").rglob("*.py"))
    entry = source / "scripts/train_sequence.py"
    if entry not in paths:
        paths.append(entry)
    files = {path.relative_to(source).as_posix(): sha(path) for path in paths}
    return canon(files), files


def verify_source(source: Path, manifest_sha: str | None) -> tuple[str, dict[str, str]]:
    """Require the pinned clean trainer commit and optional transport manifest."""
    if manifest_sha is None:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--", "src", "scripts"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        if commit != EXPECTED_COMMIT or dirty:
            raise ValueError(
                "trainer commit or src/scripts cleanliness differs from the approved source"
            )
    else:
        path = source / "TRANSPORT_MANIFEST.json"
        manifest = read(path)
        if sha(path) != manifest_sha or manifest.get("commit") != EXPECTED_COMMIT:
            raise ValueError("source transport manifest hash/commit mismatch")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("source manifest has no file digest map")
        for name, expected in files.items():
            member = source / name
            if (
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or member.is_symlink()
                or not member.is_file()
                or sha(member) != expected
            ):
                raise ValueError(f"source manifest member changed: {name}")
    return source_files(source)


def load_contract(source: Path, config_path: Path, split_path: Path, data_root: Path) -> JSON:
    """Resolve config and exact data provenance without constructing a model."""
    code = (
        "import json,sys; from pathlib import Path; "
        "from wnn.training.sequence_config import load_config; "
        "from wnn.training.sequence_provenance import canonical_sha256; "
        "from wnn.training.sequence_data import build_data; "
        "c=load_config(Path(sys.argv[1])); "
        "d=build_data(c,Path(sys.argv[3]),Path(sys.argv[2])); d.normalization(); "
        "print(json.dumps({'config':c,'sha256':canonical_sha256(c),'split_sha256':d.split.sha256,"
        "'fold':d.split.fold,'data':d.provenance},allow_nan=False))"
    )
    env = os.environ.copy() | {
        "PYTHONPATH": str(source / "src"),
        "CUBLAS_EMULATE_SINGLE_PRECISION": "1",
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(config_path), str(split_path), str(data_root)],
        cwd=source,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or not isinstance(value.get("config"), dict):
        raise TypeError("canonical config loader returned invalid output")
    return value


def gpu_snapshot() -> tuple[dict[str, JSON], list[JSON]]:
    """Read GPU free memory and current PID/owner baseline without signaling jobs."""
    query = "nvidia-smi --query-gpu=uuid,name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits"
    output = subprocess.run(
        query.split(), check=True, capture_output=True, text=True, timeout=20
    ).stdout
    devices = {}
    for line in output.splitlines():
        uuid, name, total, free, util = [part.strip() for part in line.split(",")]
        devices[uuid] = {
            "uuid": uuid,
            "name": name,
            "total_mib": int(total),
            "free_mib": int(free),
            "utilization_percent": int(util),
        }
    query = "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits"
    output = subprocess.run(
        query.split(), check=True, capture_output=True, text=True, timeout=20
    ).stdout
    apps = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2 or not fields[1].isdigit():
            continue
        pid = int(fields[1])
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            uid = int(next(row for row in status.splitlines() if row.startswith("Uid:")).split()[1])
            user = pwd.getpwuid(uid).pw_name
        except (OSError, KeyError, StopIteration):
            uid, user = -1, "unknown"
        apps.append({"gpu_uuid": fields[0], "pid": pid, "uid": uid, "user": user})
    return devices, apps


def filesystem_type(path: Path) -> str:
    """Return the filesystem holding a path so scratch cannot land in RAM."""
    resolved, best = path.resolve(), ""
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, _, right = line.partition(" - ")
        fields = left.split()
        if len(fields) >= 5:
            mount = Path(fields[4].replace("\\040", " "))
            if (resolved == mount or mount in resolved.parents) and len(str(mount)) > len(best):
                best, fs = str(mount), right.split()[0]
    if not best:
        raise ValueError(f"cannot find physical filesystem for {path}")
    return fs


def preflight(plan: JSON) -> tuple[Path, Path, list[JSON], JSON]:
    """Verify four approved profiles, source/data identities, and allocated resources."""
    if sys.platform != "linux":
        raise RuntimeError("panel requires the allocated Linux GPU host")
    source, data_root = (
        Path(plan["source_root"]).resolve(strict=True),
        Path(plan["data_root"]).resolve(strict=True),
    )
    manifest_sha = plan.get("source_manifest_sha256")
    source_sha, source_map = verify_source(source, manifest_sha)
    raw_jobs = plan["jobs"]
    if (
        not isinstance(raw_jobs, list)
        or len(raw_jobs) != 4
        or {job.get("name") for job in raw_jobs if isinstance(job, dict)} != set(CELLS)
    ):
        raise ValueError("plan must contain exactly ctl_f0, reg_f0, ctl_f2 and reg_f2")
    launcher_raw = Path(plan["launcher"]) if plan.get("launcher") else None
    if launcher_raw and launcher_raw.is_symlink():
        raise ValueError("CUDA budget wrapper cannot be a symlink")
    launcher = launcher_raw.resolve(strict=True) if launcher_raw else None
    launcher_sha = sha(launcher) if launcher else None
    if launcher and plan.get("launcher_sha256") != launcher_sha:
        raise ValueError("launcher differs from plan-bound SHA256")
    jobs, gpu_ids, cpu_ids = [], [], []
    for raw in raw_jobs:
        if not isinstance(raw, dict):
            raise TypeError("each plan job must be an object")
        name, fold = str(raw["name"]), CELLS[str(raw["name"])]
        inputs = [Path(raw[key]) for key in ("config", "split_manifest", "profile_receipt")]
        if any(path.is_symlink() for path in inputs):
            raise ValueError(f"{name}: input receipt/config paths cannot be symlinks")
        config, split_path, profile_path = (path.resolve(strict=True) for path in inputs)
        raw_hash, split_hash, profile_hash = map(sha, (config, split_path, profile_path))
        for field, actual in (
            ("config_sha256", raw_hash),
            ("split_sha256", split_hash),
            ("profile_sha256", profile_hash),
        ):
            if field in raw and raw[field] != actual:
                raise ValueError(f"{name}: {field} mismatch")
        normalized, profile = (
            load_contract(source, config, split_path, data_root),
            read(profile_path),
        )
        cfg, identity = normalized["config"], profile.get("identity")
        if (
            cfg.get("epochs") != 5
            or cfg.get("expected_fold") != fold
            or cfg.get("required_world_size") != 2
            or cfg.get("model", {}).get("architecture") != "transformer"
            or cfg.get("dropout_policy") != "explicit_regularization_v1"
        ):
            raise ValueError(f"{name}: expected five-epoch W2 Transformer config")
        if (
            profile.get("receipt_schema") != "wnn.sequence-profile.v1"
            or profile.get("status") != "profile_complete"
            or profile.get("approved") is not True
            or not isinstance(identity, dict)
        ):
            raise ValueError(f"{name}: profile is not approved and complete")
        if (
            identity.get("config_sha256") != normalized["sha256"]
            or identity.get("source_sha256") != source_sha
            or identity.get("source_files") != source_map
        ):
            raise ValueError(f"{name}: profile config/source identity mismatch")
        if (
            normalized["fold"] != fold
            or normalized["split_sha256"] != split_hash
            or identity.get("data") != normalized["data"]
            or identity.get("data_sha256") != canon(normalized["data"])
        ):
            raise ValueError(
                f"{name}: approved profile does not bind canonical split/data provenance"
            )
        assigned, cpus = raw.get("gpu_uuids"), raw.get("cpus")
        if (
            not isinstance(assigned, list)
            or len(assigned) != 2
            or any(not isinstance(g, str) or not g.startswith("GPU-") for g in assigned)
        ):
            raise ValueError(f"{name}: exactly two GPU UUIDs are required")
        if (
            not isinstance(cpus, list)
            or len(cpus) != 8
            or any(type(c) is not int or c < 0 for c in cpus)
        ):
            raise ValueError(f"{name}: exactly eight CPU IDs are required")
        runtime = identity.get("runtime")
        devices = runtime.get("devices") if isinstance(runtime, dict) else []
        uuids = (
            [str(device.get("uuid", "")).removeprefix("GPU-").lower() for device in devices]
            if isinstance(devices, list)
            else []
        )
        if (
            not isinstance(runtime, dict)
            or runtime.get("world_size") != 2
            or uuids != [g.removeprefix("GPU-").lower() for g in assigned]
        ):
            raise ValueError(f"{name}: profile GPU order/world size differs from plan")
        cap = float(raw.get("cuda_cap_gib", plan.get("cuda_cap_gib", 100 if launcher else 0)))
        if launcher and (
            profile.get("launcher_sha256") != launcher_sha
            or profile.get("cuda_cap_gib", cap) != cap
        ):
            raise ValueError(f"{name}: profile used a different CUDA wrapper/cap")
        if not launcher and profile.get("launcher_sha256") is not None:
            raise ValueError(f"{name}: profile used a wrapper that the fit plan omitted")
        memory = profile.get("memory_by_rank")
        if not isinstance(memory, list) or len(memory) != 2:
            raise ValueError(f"{name}: profile must contain memory measurements for both ranks")
        peak = max(int(item["peak_reserved_bytes"]) for item in memory) / GIB
        budget = float(raw.get("gpu_memory_budget_gib", plan.get("gpu_memory_budget_gib", 50)))
        seconds = int(raw.get("max_seconds", plan.get("max_seconds", 86400)))
        if (
            not math.isfinite(budget)
            or budget < peak
            or not 0 < seconds <= 604800
            or (launcher and (not math.isfinite(cap) or cap <= 0))
        ):
            raise ValueError(f"{name}: invalid memory budget, CUDA cap or finite runtime")
        gpu_ids.extend(assigned)
        cpu_ids.extend(cpus)
        jobs.append(
            {
                "name": name,
                "config": config,
                "config_raw_sha256": raw_hash,
                "config_sha256": normalized["sha256"],
                "split": split_path,
                "split_sha256": split_hash,
                "profile": profile_path,
                "profile_sha256": profile_hash,
                "profile_identity": identity,
                "gpus": assigned,
                "cpus": cpus,
                "seconds": seconds,
                "budget_gib": budget,
                "cap_gib": cap,
                "peak_gib": peak,
                "compile_cache": raw.get("compile_cache"),
            }
        )
    if (
        len(set(gpu_ids)) != 8
        or len(set(cpu_ids)) != 32
        or not set(cpu_ids).issubset(os.sched_getaffinity(0))
    ):
        raise ValueError(
            "all eight GPUs and 32 CPUs must be disjoint and inside the allowed cpuset"
        )
    devices, apps = gpu_snapshot()
    if len(devices) != 8 or any(
        g not in devices or "B300" not in str(devices[g]["name"]).upper() for g in gpu_ids
    ):
        raise ValueError("the assigned UUIDs must be the eight visible B300 devices")
    baseline = plan.get("baseline", {})
    pids = {int(app["pid"]) for app in apps if app["gpu_uuid"] in gpu_ids}
    declared = baseline.get("foreign_pids") if isinstance(baseline, dict) else None
    if declared is not None and pids - {
        int(item if isinstance(item, int) else item["pid"]) for item in declared
    }:
        raise ValueError("unrecorded GPU process appeared after PID baseline")
    if pids and not launcher:
        raise ValueError("co-resident GPU workloads require the approved CUDA budget wrapper")
    reserve = float(plan.get("gpu_reserve_gib", 24))
    for job in jobs:
        for gpu in job["gpus"]:
            observer_budget = 12 if gpu == job["gpus"][0] else 0
            required = (
                max(job["budget_gib"], job["cap_gib"] if launcher else 0)
                + observer_budget
                + reserve
            )
            if devices[gpu]["free_mib"] / 1024 <= required:
                raise ValueError(
                    f"{job['name']}/{gpu}: free VRAM is below budget plus observer/reserve"
                )
    controller_sha = sha(Path(__file__).resolve())
    closure = canon(
        {
            "controller": controller_sha,
            "source": source_sha,
            "launcher": launcher_sha,
            "manifest": manifest_sha,
            "plan": canon(plan),
        }
    )
    return (
        source,
        data_root,
        jobs,
        {
            "source_sha256": source_sha,
            "source_files": source_map,
            "source_manifest_sha256": manifest_sha,
            "launcher": str(launcher) if launcher else None,
            "launcher_sha256": launcher_sha,
            "controller_sha256": controller_sha,
            "operator_closure_sha256": closure,
            "gpu_baseline": devices,
            "compute_apps_baseline": apps,
            "gpu_reserve_gib": reserve,
        },
    )


def proc_stat(pid: int) -> tuple[int, int, int] | None:
    """Read process start ticks, group and session to guard termination."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19]), int(fields[2]), int(fields[3])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def group_members(group: int) -> list[int]:
    """List only children in the newly created session."""
    result = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            stat = proc_stat(int(entry.name))
            if stat and stat[1:] == (group, group):
                result.append(int(entry.name))
    return result


def stop_owned(worker: JSON) -> None:
    """Stop only the recorded child session, never baseline GPU processes."""
    pid, start = int(worker["pid"]), int(worker["start_ticks"])
    identity = proc_stat(pid)
    if (identity and identity != (start, pid, pid)) or (
        identity is None and not group_members(pid)
    ):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and group_members(pid):
        time.sleep(0.2)
    if group_members(pid):
        identity = proc_stat(pid)
        if identity == (start, pid, pid) or (identity is None and group_members(pid)):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
    worker["process"].wait(timeout=10)


def terminal_fit(job: JSON, fit: Path) -> JSON:
    """Require a matching completed five-epoch model and terminal receipt."""
    model, path = fit / "model.pt", fit / "terminal_status.json"
    if model.is_symlink() or path.is_symlink() or not model.is_file() or not path.is_file():
        raise ValueError("fit output lacks regular model.pt/terminal_status.json files")
    status, model_hash = read(path), sha(model)
    if (
        status.get("status") != "completed"
        or status.get("completed_epochs") != 5
        or status.get("model_sha256") != model_hash
        or status.get("identity") != job["profile_identity"]
    ):
        raise ValueError("terminal receipt does not prove this exact five-epoch fit")
    return {"completed_epochs": 5, "model_sha256": model_hash, "terminal_sha256": sha(path)}


def main() -> None:
    """Admit once, launch four concurrent fits, poll every two seconds, and preserve outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if not isinstance(plan, dict):
        raise TypeError("plan must be a JSON object")
    raw_root = Path(plan["run_root"])
    if raw_root.is_symlink():
        raise ValueError("run_root cannot be a symlink")
    root = raw_root.resolve()
    for immutable in (
        Path(plan["source_root"]).resolve(strict=True),
        Path(plan["data_root"]).resolve(strict=True),
    ):
        if root == immutable or root.is_relative_to(immutable) or immutable.is_relative_to(root):
            raise ValueError("run_root must be separate from source and data roots")
    if filesystem_type(root) in {"tmpfs", "ramfs", "devtmpfs"}:
        raise ValueError("run_root/TMPDIR must use physical disk")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "STATE.json").exists():
        raise FileExistsError("run_root belongs to an earlier attempt")
    names = [str(job["name"]) for job in plan.get("jobs", []) if isinstance(job, dict)]
    if set(names) != set(CELLS) or len(names) != 4:
        raise ValueError("plan must contain exactly four panel cells")
    with (root / "RUN_PANEL.lock").open("x") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in names:
            (root / name).mkdir(exist_ok=False)
        states: dict[str, JSON] = {name: {"status": "profile"} for name in names}

        def emit(status: str, extra: JSON | None = None) -> None:
            now = datetime.now(UTC).isoformat()
            body: JSON = {
                "schema": "wnn.compact-panel-state.v1",
                "status": status,
                "updated_utc": now,
                "jobs": states,
            }
            if extra:
                body.update(extra)
            write(root / "STATE.json", body)
            for name, state in states.items():
                write(root / name / "STATE.json", state)

        with (root / "PLAN.json").open("xb") as saved:
            saved.write(plan_bytes)
        emit("profile", {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest()})
        workers: dict[str, JSON] = {}
        try:
            source, data_root, jobs, admission = preflight(plan)
            emit("profile", admission)
            for job in jobs:
                name = str(job["name"])
                if STOP.is_set():
                    break
                if sha(Path(__file__).resolve()) != admission["controller_sha256"]:
                    raise ValueError("controller changed after admission")
                source_sha, _ = verify_source(source, admission["source_manifest_sha256"])
                if (
                    source_sha != admission["source_sha256"]
                    or sha(job["config"]) != job["config_raw_sha256"]
                    or sha(job["split"]) != job["split_sha256"]
                    or sha(job["profile"]) != job["profile_sha256"]
                ):
                    raise ValueError(f"{name}: source/config/split/profile changed after admission")
                if (
                    admission["launcher"]
                    and sha(Path(admission["launcher"])) != admission["launcher_sha256"]
                ):
                    raise ValueError("CUDA budget wrapper changed after admission")
                folder, fit = root / name, root / name / "fit"
                scratch = folder / "scratch"
                cache = (
                    Path(job["compile_cache"])
                    if job.get("compile_cache")
                    else folder / "torchinductor"
                )
                scratch.mkdir(exist_ok=True)
                cache.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    f"--nproc-per-node={len(job['gpus'])}",
                ]
                if admission["launcher"]:
                    command += [admission["launcher"], str(source / "scripts/train_sequence.py")]
                else:
                    command.append(str(source / "scripts/train_sequence.py"))
                command += [
                    "--config",
                    str(job["config"]),
                    "--data-root",
                    str(data_root),
                    "--split-manifest",
                    str(job["split"]),
                    "--output",
                    str(fit),
                    "--profile-receipt",
                    str(job["profile"]),
                ]
                command = ["taskset", "--cpu-list", ",".join(map(str, job["cpus"])), *command]
                env = os.environ.copy() | {
                    "CUDA_VISIBLE_DEVICES": ",".join(job["gpus"]),
                    "PYTHONPATH": str(source / "src"),
                    "CUBLAS_EMULATE_SINGLE_PRECISION": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "TORCHINDUCTOR_COMPILE_THREADS": "1",
                    "TMPDIR": str(scratch),
                    "TORCHINDUCTOR_CACHE_DIR": str(cache),
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
                if admission["launcher"]:
                    env["WNN_CUDA_CAP_GIB"] = str(job["cap_gib"])
                receipt = {
                    "stage": "fit",
                    "command": command,
                    "gpus": job["gpus"],
                    "cpus": job["cpus"],
                    "config_sha256": job["config_sha256"],
                    "split_sha256": job["split_sha256"],
                    "profile_sha256": job["profile_sha256"],
                    "source_sha256": admission["source_sha256"],
                    "launcher_sha256": admission["launcher_sha256"],
                    "operator_closure_sha256": admission["operator_closure_sha256"],
                    "cuda_cap_gib": job["cap_gib"],
                    "environment": {
                        key: env[key]
                        for key in (
                            "CUDA_VISIBLE_DEVICES",
                            "CUBLAS_EMULATE_SINGLE_PRECISION",
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "TORCHINDUCTOR_COMPILE_THREADS",
                            "TMPDIR",
                            "TORCHINDUCTOR_CACHE_DIR",
                            "WNN_CUDA_CAP_GIB",
                        )
                        if key in env
                    },
                    "started_utc": datetime.now(UTC).isoformat(),
                }
                write(folder / "LAUNCH.json", receipt)
                try:
                    with (folder / "train.log").open("xb") as log:
                        proc = subprocess.Popen(
                            command,
                            cwd=source,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    identity = None
                    for _ in range(100):
                        identity = proc_stat(proc.pid)
                        if identity:
                            break
                        time.sleep(0.02)
                    if identity is None or identity[1:] != (proc.pid, proc.pid):
                        proc.terminate()
                        proc.wait(timeout=10)
                        if group_members(proc.pid):
                            os.killpg(proc.pid, signal.SIGTERM)
                        raise RuntimeError("cannot prove spawned fit process-group identity")
                    workers[name] = {
                        "process": proc,
                        "pid": proc.pid,
                        "start_ticks": identity[0],
                        "started": time.monotonic(),
                        "job": job,
                    }
                    receipt.update(
                        {
                            "pid": proc.pid,
                            "pgid": proc.pid,
                            "session": proc.pid,
                            "start_ticks": identity[0],
                        }
                    )
                    write(folder / "LAUNCH.json", receipt)
                    states[name] = {
                        "status": "fit",
                        "pid": proc.pid,
                        "updated_utc": datetime.now(UTC).isoformat(),
                    }
                except Exception as error:
                    states[name] = {
                        "status": "failed",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                    write(folder / "EXIT.json", states[name])
                emit("fit", {"operator_closure_sha256": admission["operator_closure_sha256"]})
            while workers:
                for name, worker in list(workers.items()):
                    proc, job = worker["process"], worker["job"]
                    reason = "cancelled by stop signal" if STOP.is_set() else None
                    if reason is None and time.monotonic() - worker["started"] > job["seconds"]:
                        reason = "max_seconds exceeded"
                    if reason:
                        stop_owned(worker)
                        states[name] = {
                            "status": "cancelled" if STOP.is_set() else "failed",
                            "reason": reason,
                        }
                    elif proc.poll() is not None:
                        if group_members(proc.pid):
                            stop_owned(worker)
                            raise RuntimeError(f"{name}: torchrun exited with live child ranks")
                        try:
                            result = (
                                terminal_fit(job, root / name / "fit")
                                if proc.returncode == 0
                                else {}
                            )
                            states[name] = {
                                "status": "completed" if proc.returncode == 0 else "failed",
                                **result,
                            }
                            if proc.returncode != 0:
                                states[name]["reason"] = f"torchrun exited {proc.returncode}"
                        except Exception as error:
                            states[name] = {
                                "status": "failed",
                                "reason": f"{type(error).__name__}: {error}",
                            }
                    else:
                        continue
                    states[name]["updated_utc"] = datetime.now(UTC).isoformat()
                    write(
                        root / name / "EXIT.json",
                        {
                            **states[name],
                            "exit_code": proc.returncode,
                            "elapsed_seconds": time.monotonic() - worker["started"],
                        },
                    )
                    workers.pop(name)
                emit("cancelled" if STOP.is_set() else "fit")
                if workers:
                    time.sleep(2)
            for name, state in states.items():
                if state["status"] == "profile":
                    states[name] = {
                        "status": "cancelled",
                        "reason": "admission closed before launch",
                    }
                    write(root / name / "EXIT.json", states[name])
            final = (
                "cancelled"
                if STOP.is_set()
                else "failed"
                if any(s["status"] == "failed" for s in states.values())
                else "completed"
            )
            emit(final)
        except Exception as error:
            for name, worker in list(workers.items()):
                with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                    stop_owned(worker)
                states[name] = {
                    "status": "failed",
                    "reason": f"controller error: {type(error).__name__}: {error}",
                }
                write(root / name / "EXIT.json", states[name])
            for name, state in states.items():
                if state["status"] == "profile":
                    states[name] = {
                        "status": "failed",
                        "reason": f"preflight/controller error: {type(error).__name__}: {error}",
                    }
                    write(root / name / "EXIT.json", states[name])
            emit("failed", {"error": f"{type(error).__name__}: {error}"})
            raise


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    main()
