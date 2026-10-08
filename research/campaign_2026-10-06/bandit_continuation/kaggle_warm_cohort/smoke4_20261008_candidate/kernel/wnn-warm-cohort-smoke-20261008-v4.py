"""Kaggle warm-cohort driver: mount private inputs once, then drain queued specs on two T4 workers.

Inputs are discovered under the Kaggle input root (default /kaggle/input), never assumed flat:

* ``train.parquet`` / ``valid.parquet`` directories (the raw-data dataset),
* any directory holding an ``assets-manifest.json`` (hash-named ``.wnn`` blobs restored by path),
* the source dataset holding ``wnn_src.tar.gz`` (scripts/, src/, warm_specs/).

The repository is assembled outside /kaggle/working so that the saved output stays small: only
results, logs, grades, held-out 468-group predictions and small checkpoints are copied to
/kaggle/working/out. Two ``warm_queue_worker.py`` processes then run, one per GPU, with
``CUDA_VISIBLE_DEVICES`` 0 and 1 and separate queue directories.

Environment overrides (used by the local dry run): WARM_INPUT, WARM_WORK, WARM_REPO,
WARM_DRY_RUN=1 (validate paths and argv only; no training), WARM_MAX_WALL_SECONDS,
WARM_IDLE_EXIT_SECONDS. Runtime is hard-capped at 7800 seconds from Python entry.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

INPUT = Path(os.environ.get("WARM_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("WARM_WORK", "/kaggle/working"))
REPO = Path(os.environ.get("WARM_REPO", "/tmp/wnn_repo"))
OUT = WORK / "out"
PHASE_LOG = WORK / "phase.jsonl"
SRC_ARCHIVE = "wnn_src.tar.gz"
MANIFEST = "assets-manifest.json"
SMALL_BYTES = 50 << 20
EXPECTED_ASSET_MANIFESTS = {
    "vladkochetov07/wnn-warm-cohort-src-20261008-v2":
        "96fa0dcbd2c0a56f8ef9d23620080337799022290ffee87196d52de58aca4f70",
    "vladkochetov07/wnn-c20261004-fold0-assets":
        "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7",
}
PORTABLE_REGISTRY_RELATIVE_PATH = Path("research/campaign_2026-10-04/portable_inputs.json")
PORTABLE_REGISTRY_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
PORTABLE_REGISTRY_ENV = "WNN_PORTABLE_DATA_MANIFEST"
DRY_RUN = os.environ.get("WARM_DRY_RUN") == "1"
MAX_WALL_CEILING_SECONDS = 7800
MAX_WALL = min(
    int(os.environ.get("WARM_MAX_WALL_SECONDS", str(MAX_WALL_CEILING_SECONDS))),
    MAX_WALL_CEILING_SECONDS,
)
if MAX_WALL <= 0:
    raise ValueError("WARM_MAX_WALL_SECONDS must be positive")
IDLE_EXIT = str(min(max(int(os.environ.get("WARM_IDLE_EXIT_SECONDS", "180")), 30), 60))
GPUS = (0, 1)
MIN_START_AVAILABLE_BYTES = 20_000_000_000
MIN_RUNTIME_AVAILABLE_BYTES = 2_000_000_000
MAX_WORKER_RSS_BYTES = 24_000_000_000
SUPERVISOR_HEARTBEAT_SECONDS = 30
SUPERVISOR_SHUTDOWN_RESERVE_SECONDS = 60
WORKER_STOP_GRACE_SECONDS = 8
PLACEHOLDERS = {"{python}": sys.executable, "{repo}": str(REPO), "{out}": str(OUT)}


def phase(name: str, **fields: object) -> None:
    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phase": name, **fields}
    line = json.dumps(record, sort_keys=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with PHASE_LOG.open("a") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(line, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def find_dirs(filename: str, max_depth: int = 6) -> list[Path]:
    hits: list[Path] = []
    base_depth = len(INPUT.parts)
    for dirpath, dirnames, filenames in os.walk(INPUT):
        current = Path(dirpath)
        if len(current.parts) - base_depth >= max_depth:
            dirnames[:] = []
        if filename in filenames:
            hits.append(current)
    return hits


def link(target: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        if path.is_symlink() and Path(os.readlink(path)) == target:
            return
        raise FileExistsError(f"refusing to replace {path}")
    # The dry run mirrors mounts with symlinks; the target must exist to be useful.
    if DRY_RUN and not target.exists():
        raise FileNotFoundError(target)
    os.symlink(target, path)


def mount_manifests() -> list[dict[str, object]]:
    """Link every manifest entry into the repository and verify the mounted bytes."""
    seen: dict[str, str] = {}
    seen_datasets: set[str] = set()
    mounted: list[dict[str, object]] = []
    for directory in find_dirs(MANIFEST):
        manifest_path = directory / MANIFEST
        manifest = json.loads(manifest_path.read_text())
        dataset = str(manifest.get("dataset", ""))
        expected_manifest_sha256 = EXPECTED_ASSET_MANIFESTS.get(dataset)
        if expected_manifest_sha256 is None:
            raise ValueError(f"unreviewed asset dataset manifest: {dataset or directory}")
        actual_manifest_sha256 = sha256(manifest_path)
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                f"asset manifest changed for {dataset}: "
                f"{actual_manifest_sha256} != {expected_manifest_sha256}"
            )
        if dataset in seen_datasets:
            raise ValueError(f"duplicate mounted asset dataset {dataset}")
        seen_datasets.add(dataset)
        entries = manifest.get("files", [])
        for entry in entries:
            rel = entry["path"]
            if rel in seen:
                raise ValueError(f"duplicate manifest path {rel} in {directory}")
            source = directory / entry["filename"]
            if not source.is_file():
                raise FileNotFoundError(source)
            size = source.stat().st_size
            if size != int(entry["size"]):
                raise ValueError(f"size mismatch for {rel}: {size} != {entry['size']}")
            if size <= SMALL_BYTES and not DRY_RUN and sha256(source) != entry["sha256"]:
                raise ValueError(f"sha256 mismatch for {rel}")
            link(source, REPO / rel)
            seen[rel] = str(source)
            mounted.append({"path": rel, "bytes": size, "dataset_dir": str(directory)})
        phase("manifest_mounted", dataset_dir=str(directory), files=len(entries))
    if seen_datasets != set(EXPECTED_ASSET_MANIFESTS):
        missing = sorted(set(EXPECTED_ASSET_MANIFESTS) - seen_datasets)
        raise ValueError(f"reviewed asset datasets are missing: {missing}")
    return mounted


def bind_portable_registry(specs: list[dict[str, object]]) -> dict[str, object]:
    """Verify the exact portable source registry and pass it to every child step."""
    registry = REPO / PORTABLE_REGISTRY_RELATIVE_PATH
    if not registry.is_file():
        raise FileNotFoundError(f"portable data registry is missing: {registry}")
    actual_sha256 = sha256(registry)
    if actual_sha256 != PORTABLE_REGISTRY_SHA256:
        raise ValueError(
            f"portable data registry changed: {actual_sha256} != {PORTABLE_REGISTRY_SHA256}"
        )
    bound_steps = 0
    for spec in specs:
        for step in spec["steps"]:
            env = step.setdefault("env", {})
            existing = env.get(PORTABLE_REGISTRY_ENV)
            if existing is not None and str(existing) != str(registry):
                raise ValueError(f"{spec['run_id']}: conflicting {PORTABLE_REGISTRY_ENV}")
            env[PORTABLE_REGISTRY_ENV] = str(registry)
            bound_steps += 1
    return {"path": str(registry), "sha256": actual_sha256, "child_steps": bound_steps}


def mount_raw_data() -> None:
    roots = [d for d in find_dirs("train.parquet") if (d / "valid.parquet").is_file()]
    if len(roots) != 1:
        raise RuntimeError(f"expected exactly one raw-data root, found {roots}")
    raw = roots[0]
    for name in ("train.parquet", "valid.parquet", "valid_mask.parquet"):
        if (raw / name).is_file():
            link(raw / name, REPO / "datasets" / name)
    phase("raw_data_mounted", root=str(raw))


def write_boot_marker() -> None:
    """Persist proof of Python entry before any mounted-input discovery."""
    WORK.mkdir(parents=True, exist_ok=True)
    marker = {
        "event": "kernel_boot",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "input_root": str(INPUT),
    }
    marker_path = WORK / "boot_marker.json"
    with marker_path.open("w") as handle:
        json.dump(marker, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(marker, sort_keys=True), flush=True)


def unpack_source() -> None:
    """Unpack a manifest-bound archive or copy a legacy Kaggle-extracted tree."""
    archives = find_dirs(SRC_ARCHIVE)
    REPO.mkdir(parents=True, exist_ok=True)
    repository_archive = REPO / SRC_ARCHIVE
    if repository_archive.is_file():
        archive = repository_archive
    elif len(archives) == 1:
        archive = archives[0] / SRC_ARCHIVE
    else:
        archive = None
    if archive is not None:
        with tarfile.open(archive, "r:gz") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(REPO, filter="data")
            else:  # pragma: no cover - older Python
                tar.extractall(REPO)  # noqa: S202 - archive is our own hash-pinned source
        phase("source_unpacked", archive=str(archive), sha256=sha256(archive))
    else:
        # Kaggle may extract an uploaded tar.gz into a folder; accept that layout too.
        script_dirs = [Path(d) for d, _, f in os.walk(INPUT)
                       if "warm_queue_worker.py" in f and Path(d).name == "scripts"]
        if len(script_dirs) != 1:
            raise RuntimeError(f"expected one source tree or archive, found {archives} {script_dirs}")
        tree = script_dirs[0].parent
        for part in ("scripts", "src", "warm_specs"):
            shutil.copytree(tree / part, REPO / part, dirs_exist_ok=True)
        phase("source_copied", tree=str(tree))
    for sub in ("datasets", "cache", "artifacts/runs"):
        (REPO / sub).mkdir(parents=True, exist_ok=True)


def load_specs() -> list[dict[str, object]]:
    specs = []
    for path in sorted((REPO / "warm_specs").glob("*.json")):
        spec = json.loads(path.read_text())
        if spec.get("run_id") != path.stem:
            raise ValueError(f"{path.name}: run_id must equal the file stem")
        text = json.dumps(spec)
        for key, value in PLACEHOLDERS.items():
            text = text.replace(key, value)
        spec = json.loads(text)
        specs.append(spec)
    return specs


def validate_spec(spec: dict[str, object]) -> list[str]:
    """Return problems: missing scripts, missing path arguments, unknown trainer flags."""
    problems: list[str] = []
    for step in spec["steps"]:  # type: ignore[index]
        argv = [str(a) for a in step["argv"]]
        script = next((a for a in argv if a.endswith(".py")), None)
        if script is None:
            continue
        script_path = Path(script)
        if not script_path.is_file():
            problems.append(f"{spec['run_id']}/{step['name']}: missing script {script}")
            continue
        known = set(re.findall(r"add_argument\(\s*\"(--[a-z0-9-]+)\"", script_path.read_text()))
        known |= set(re.findall(r"add_argument\(\s*\"[^\"]+\",\s*\"(--[a-z0-9-]+)\"", script_path.read_text()))
        for arg in argv:
            if arg.startswith("--") and known and arg not in known and arg != "--help":
                problems.append(f"{spec['run_id']}/{step['name']}: unknown flag {arg} for {script}")
    for out in spec.get("expected_outputs", []):  # type: ignore[union-attr]
        if not str(out).startswith(str(OUT)) and not str(out).startswith(str(REPO)):
            problems.append(f"{spec['run_id']}: expected output outside repo/out: {out}")
    return problems


def write_queues(specs: list[dict[str, object]]) -> dict[int, int]:
    counts = {gpu: 0 for gpu in GPUS}
    for index, spec in enumerate(specs):
        gpu = int(spec.get("gpu", GPUS[index % len(GPUS)]))
        if gpu not in GPUS:
            raise ValueError(f"{spec['run_id']}: gpu must be 0 or 1")
        queue = WORK / "queues" / f"gpu{gpu}"
        filename = f"{spec['run_id']}.json"
        encoded = (json.dumps(spec, indent=1) + "\n").encode()
        destinations = {
            name: queue / name / filename
            for name in ("pending", "running", "done", "failed")
        }
        existing = [path for path in destinations.values() if path.is_file()]
        if existing:
            if any(path.read_bytes() != encoded for path in existing):
                raise RuntimeError(f"existing queue spec differs from frozen spec {filename}")
            if any(path.parent.name in {"pending", "running"} for path in existing):
                counts[gpu] += 1
            continue

        pending = destinations["pending"]
        pending.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=pending.parent, delete=False, suffix=".tmp"
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, pending)
        counts[gpu] += 1
    return counts


def gpu_inventory() -> list[dict[str, object]]:
    """Sample every visible GPU and its memory usage with nvidia-smi."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    devices: list[dict[str, object]] = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 5:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {row!r}")
        try:
            index = int(row[0].strip())
            memory_used_mib = int(row[3].strip())
            memory_total_mib = int(row[4].strip())
        except ValueError as error:
            raise RuntimeError(f"nvidia-smi returned nonnumeric GPU memory: {row!r}") from error
        devices.append(
            {
                "index": index,
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "memory_used_mib": memory_used_mib,
                "memory_total_mib": memory_total_mib,
            }
        )
    if not devices:
        raise RuntimeError("nvidia-smi reported no visible GPUs")
    return devices


def _host_available_bytes() -> int:
    """Read Linux MemAvailable, failing closed if host memory telemetry is absent."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _cgroup_available_bytes() -> int | None:
    """Return cgroup memory headroom when a finite limit is exposed."""
    candidates = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, current_path in candidates:
        if not limit_path.is_file() or not current_path.is_file():
            continue
        limit_text = limit_path.read_text().strip()
        if limit_text == "max":
            return None
        limit = int(limit_text)
        if limit >= 1 << 60:
            return None
        current = int(current_path.read_text().strip())
        return max(0, limit - current)
    return None


def _process_memory_table() -> dict[int, tuple[int, int]]:
    """Map each visible PID to its parent PID and resident bytes."""
    table: dict[int, tuple[int, int]] = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            pid = int(status_path.parent.name)
            parent_pid: int | None = None
            rss_bytes = 0
            for line in status_path.read_text().splitlines():
                if line.startswith("PPid:"):
                    parent_pid = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    rss_bytes = int(line.split()[1]) * 1024
            if parent_pid is not None:
                table[pid] = (parent_pid, rss_bytes)
        except (OSError, ValueError, IndexError):
            continue
    return table


def _descendants(root_pid: int, process_table: dict[int, tuple[int, int]]) -> set[int]:
    """Return one process root and every visible descendant PID."""
    descendants = {root_pid}
    while True:
        children = {pid for pid, (parent, _) in process_table.items() if parent in descendants}
        updated = descendants | children
        if updated == descendants:
            return descendants
        descendants = updated


def resource_snapshot(
    workers: dict[int, subprocess.Popen[bytes]],
) -> dict[str, object]:
    """Sample host/cgroup memory, worker-tree RSS, and per-GPU memory."""
    host_available = _host_available_bytes()
    cgroup_available = _cgroup_available_bytes()
    available_candidates = [host_available]
    if cgroup_available is not None:
        available_candidates.append(cgroup_available)
    effective_available = min(available_candidates)

    process_table = _process_memory_table()
    worker_rss_by_gpu: dict[str, int] = {}
    worker_pids_by_gpu: dict[str, list[int]] = {}
    for gpu, process in workers.items():
        if process.poll() is not None:
            worker_rss_by_gpu[str(gpu)] = 0
            worker_pids_by_gpu[str(gpu)] = []
            continue
        if process.pid not in process_table:
            raise RuntimeError(f"cannot sample RSS for live worker PID {process.pid}")
        pids = _descendants(process.pid, process_table)
        worker_pids_by_gpu[str(gpu)] = sorted(pids)
        worker_rss_by_gpu[str(gpu)] = sum(process_table[pid][1] for pid in pids)

    gpu_rows = gpu_inventory()
    return {
        "host_available_bytes": host_available,
        "cgroup_available_bytes": cgroup_available,
        "effective_available_bytes": effective_available,
        "worker_rss_bytes_by_gpu": worker_rss_by_gpu,
        "worker_pids_by_gpu": worker_pids_by_gpu,
        "worker_aggregate_rss_bytes": sum(worker_rss_by_gpu.values()),
        "gpu_count": len(gpu_rows),
        "gpus": gpu_rows,
    }


def _startup_resource_issues(snapshot: dict[str, object]) -> list[str]:
    issues = []
    available = snapshot.get("effective_available_bytes")
    gpu_count = snapshot.get("gpu_count")
    if not isinstance(available, int) or available < MIN_START_AVAILABLE_BYTES:
        issues.append("effective available memory is below 20 GB")
    if not isinstance(gpu_count, int) or gpu_count < len(GPUS):
        issues.append(f"fewer than {len(GPUS)} GPUs are visible")
    return issues


def _runtime_resource_issues(snapshot: dict[str, object]) -> list[str]:
    issues = []
    available = snapshot.get("effective_available_bytes")
    aggregate_rss = snapshot.get("worker_aggregate_rss_bytes")
    gpu_count = snapshot.get("gpu_count")
    if not isinstance(available, int) or available < MIN_RUNTIME_AVAILABLE_BYTES:
        issues.append("effective available memory is below 2 GB")
    if not isinstance(aggregate_rss, int) or aggregate_rss > MAX_WORKER_RSS_BYTES:
        issues.append("worker-tree RSS exceeds 24 GB or is unavailable")
    if not isinstance(gpu_count, int) or gpu_count < len(GPUS):
        issues.append(f"fewer than {len(GPUS)} GPUs remain visible")
    return issues


def _terminate_workers(
    workers: dict[int, subprocess.Popen[bytes]], reason: str
) -> None:
    """Stop worker groups and, after a short grace period, all tracked descendants."""
    active = {gpu: proc for gpu, proc in workers.items() if proc.poll() is None}
    phase("workers_stop_requested", reason=reason, pids={str(gpu): proc.pid for gpu, proc in active.items()})
    for process in active.values():
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)

    deadline = time.monotonic() + WORKER_STOP_GRACE_SECONDS
    while active and time.monotonic() < deadline:
        active = {gpu: proc for gpu, proc in active.items() if proc.poll() is None}
        if active:
            time.sleep(0.2)

    if active:
        process_table = _process_memory_table()
        descendant_pids = set()
        for process in active.values():
            descendant_pids |= _descendants(process.pid, process_table)
        for pid in sorted(descendant_pids - {proc.pid for proc in active.values()}, reverse=True):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    # The worker leader may exit on SIGTERM while a child remains in its session.
    # Kill every owned process group after the grace period, including those whose
    # Popen leader has already exited, then reap each leader.
    for process in workers.values():
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    phase(
        "workers_stopped",
        reason=reason,
        returncodes={str(gpu): proc.returncode for gpu, proc in workers.items()},
    )


def start_workers() -> dict[int, subprocess.Popen[bytes]]:
    snapshot = resource_snapshot({})
    phase("resource_preflight", stage="before_workers", **snapshot)
    issues = _startup_resource_issues(snapshot)
    if issues:
        raise RuntimeError("startup resource gate rejected packet: " + "; ".join(issues))

    workers: dict[int, subprocess.Popen[bytes]] = {}
    try:
        for gpu in GPUS:
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONPATH": f"{REPO / 'src'}:{REPO}",
                    "WNN_PROJECT_ROOT": str(REPO),
                    "WNN_THREADS": "2",
                    "WNN_LOADER_WORKERS": "0",
                    "OMP_NUM_THREADS": "2",
                }
            )
            queue = WORK / "queues" / f"gpu{gpu}"
            log = (WORK / f"worker_gpu{gpu}.log").open("ab")
            workers[gpu] = subprocess.Popen(
                [sys.executable, str(REPO / "scripts/warm_queue_worker.py"), "--queue", str(queue),
                 "--poll-seconds", "10", "--idle-exit-seconds", IDLE_EXIT],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            phase("worker_started", gpu=gpu, pid=workers[gpu].pid)
            if gpu != GPUS[-1]:
                time.sleep(30)
                snapshot = resource_snapshot(workers)
                next_gpu = GPUS[GPUS.index(gpu) + 1]
                phase("resource_preflight", stage=f"before_gpu_{next_gpu}", **snapshot)
                issues = _startup_resource_issues(snapshot)
                if issues:
                    raise RuntimeError(
                        "second-worker resource gate rejected packet: " + "; ".join(issues)
                    )
    except Exception:
        if workers:
            _terminate_workers(workers, "worker_startup_failed")
        raise
    return workers


def supervise(
    workers: dict[int, subprocess.Popen[bytes]], started_monotonic: float
) -> None:
    deadline = started_monotonic + MAX_WALL
    stop_at = deadline - SUPERVISOR_SHUTDOWN_RESERVE_SECONDS
    while any(process.poll() is None for process in workers.values()):
        remaining = stop_at - time.monotonic()
        if remaining <= 0:
            phase("wall_limit_reached", max_wall_seconds=MAX_WALL)
            _terminate_workers(workers, "hard_wall_limit")
            break
        time.sleep(min(SUPERVISOR_HEARTBEAT_SECONDS, remaining))
        if not any(process.poll() is None for process in workers.values()):
            break
        try:
            snapshot = resource_snapshot(workers)
        except Exception as error:
            phase("resource_telemetry_failed", error=repr(error))
            _terminate_workers(workers, "resource_telemetry_failed")
            break

        worker_states = {}
        for gpu, process in workers.items():
            heartbeat = WORK / "queues" / f"gpu{gpu}" / "worker_heartbeat.json"
            beat = json.loads(heartbeat.read_text()) if heartbeat.exists() else {}
            results = WORK / "queues" / f"gpu{gpu}" / "results.jsonl"
            done = len(results.read_text().splitlines()) if results.exists() else 0
            worker_states[str(gpu)] = {
                "alive": process.poll() is None,
                "current_run": beat.get("current_run_id"),
                "current_step": beat.get("current_step"),
                "results": done,
            }
        phase(
            "heartbeat",
            elapsed_seconds=round(time.monotonic() - started_monotonic, 3),
            workers=worker_states,
            resources=snapshot,
        )
        issues = _runtime_resource_issues(snapshot)
        if issues:
            phase("resource_limit_reached", reasons=issues, resources=snapshot)
            _terminate_workers(workers, "resource_limit_reached")
            break

    phase(
        "supervisor_complete",
        elapsed_seconds=round(time.monotonic() - started_monotonic, 3),
        max_wall_seconds=MAX_WALL,
    )


def collect() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"gpus": {}}
    for gpu in GPUS:
        queue = WORK / "queues" / f"gpu{gpu}"
        dest = OUT / f"gpu{gpu}"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("results.jsonl", "worker_heartbeat.json"):
            if (queue / name).exists():
                shutil.copy2(queue / name, dest / name)
        if (queue / "logs").is_dir():
            shutil.copytree(queue / "logs", dest / "logs", dirs_exist_ok=True)
        if (WORK / f"worker_gpu{gpu}.log").exists():
            shutil.copy2(WORK / f"worker_gpu{gpu}.log", dest / "worker.log")
        results = queue / "results.jsonl"
        statuses = [json.loads(line)["status"] for line in results.read_text().splitlines()] if results.exists() else []
        summary["gpus"][str(gpu)] = statuses  # type: ignore[index]
    # Small checkpoints and provenance for each completed run, never the large prediction caches.
    runs_dir = REPO / "artifacts" / "runs"
    if runs_dir.is_dir():
        for run in sorted(runs_dir.iterdir()):
            if not run.is_dir() or run.is_symlink():
                continue
            dest = OUT / "runs" / run.name
            dest.mkdir(parents=True, exist_ok=True)
            for file in run.iterdir():
                if file.is_file() and file.stat().st_size <= SMALL_BYTES:
                    shutil.copy2(file, dest / file.name)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def dry_run_checks(specs: list[dict[str, object]]) -> None:
    problems: list[str] = []
    for spec in specs:
        problems.extend(validate_spec(spec))
    if problems:
        raise SystemExit("dry run failed:\n" + "\n".join(problems))
    phase("dry_run_specs_valid", specs=len(specs))


def main() -> int:
    started_monotonic = time.monotonic()
    write_boot_marker()
    phase(
        "start",
        dry_run=DRY_RUN,
        python=sys.version.split()[0],
        input=str(INPUT),
        max_wall_seconds=MAX_WALL,
        worker_rss_limit_bytes=MAX_WORKER_RSS_BYTES,
        startup_available_min_bytes=MIN_START_AVAILABLE_BYTES,
        runtime_available_min_bytes=MIN_RUNTIME_AVAILABLE_BYTES,
    )
    if not DRY_RUN:
        phase("gpus", inventory=gpu_inventory())
    OUT.mkdir(parents=True, exist_ok=True)
    if REPO.exists():
        shutil.rmtree(REPO)
    mounted = mount_manifests()
    unpack_source()
    mount_raw_data()
    phase("inputs_ready", manifest_files=len(mounted))
    specs = load_specs()
    registry = bind_portable_registry(specs)
    phase("portable_registry_verified", **registry)
    if DRY_RUN:
        dry_run_checks(specs)
        return 0
    dry_run_checks(specs)
    counts = write_queues(specs)
    phase("queues_written", counts={str(k): v for k, v in counts.items()})
    worker_start_reserve = SUPERVISOR_SHUTDOWN_RESERVE_SECONDS + 30
    if time.monotonic() >= started_monotonic + MAX_WALL - worker_start_reserve:
        phase(
            "wall_limit_before_workers",
            max_wall_seconds=MAX_WALL,
            worker_start_reserve_seconds=worker_start_reserve,
        )
        summary = collect()
        phase("collected", summary=summary)
        return 1
    try:
        workers = start_workers()
    except Exception as error:
        phase("startup_resource_gate_failed", error=repr(error))
        summary = collect()
        phase("collected", summary=summary)
        return 1
    supervise(workers, started_monotonic)
    summary = collect()
    phase("collected", summary=summary)
    return 0 if all(p.returncode == 0 for p in workers.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
