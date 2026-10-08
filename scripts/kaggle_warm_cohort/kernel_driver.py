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
WARM_IDLE_EXIT_SECONDS.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
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
DRY_RUN = os.environ.get("WARM_DRY_RUN") == "1"
MAX_WALL = int(os.environ.get("WARM_MAX_WALL_SECONDS", str(8 * 3600)))
IDLE_EXIT = os.environ.get("WARM_IDLE_EXIT_SECONDS", "180")
GPUS = (0, 1)
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
    mounted: list[dict[str, object]] = []
    for directory in find_dirs(MANIFEST):
        manifest = json.loads((directory / MANIFEST).read_text())
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
    return mounted


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


def start_workers() -> dict[int, subprocess.Popen[bytes]]:
    workers: dict[int, subprocess.Popen[bytes]] = {}
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
        time.sleep(30)  # stagger the two startups so mounts and parquet hashing do not collide
    return workers


def supervise(workers: dict[int, subprocess.Popen[bytes]]) -> None:
    started = time.time()
    while any(p.poll() is None for p in workers.values()):
        time.sleep(60)
        for gpu, proc in workers.items():
            heartbeat = WORK / "queues" / f"gpu{gpu}" / "worker_heartbeat.json"
            beat = json.loads(heartbeat.read_text()) if heartbeat.exists() else {}
            results = WORK / "queues" / f"gpu{gpu}" / "results.jsonl"
            done = len(results.read_text().splitlines()) if results.exists() else 0
            phase("heartbeat", gpu=gpu, alive=proc.poll() is None, current_run=beat.get("current_run_id"),
                  current_step=beat.get("current_step"), results=done)
        if time.time() - started > MAX_WALL:
            phase("wall_limit_reached", seconds=MAX_WALL)
            for proc in workers.values():
                if proc.poll() is None:
                    os.killpg(proc.pid, 15)
            break


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


def gpu_inventory() -> str:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60, check=False)
        return result.stdout.strip() or result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"nvidia-smi unavailable: {error}"


def main() -> int:
    write_boot_marker()
    phase("start", dry_run=DRY_RUN, python=sys.version.split()[0], input=str(INPUT))
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
    if DRY_RUN:
        dry_run_checks(specs)
        return 0
    dry_run_checks(specs)
    counts = write_queues(specs)
    phase("queues_written", counts={str(k): v for k, v in counts.items()})
    workers = start_workers()
    supervise(workers)
    summary = collect()
    phase("collected", summary=summary)
    return 0 if all(p.returncode == 0 for p in workers.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
