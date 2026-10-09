from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))


def _boot_marker() -> None:
    marker = {
        "event": "kernel_boot",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "input_root": os.environ.get("KAGGLE_INPUT", "/kaggle/input"),
    }
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        path = WORK / "boot_marker.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(marker, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        print(json.dumps(marker, sort_keys=True), file=sys.stderr, flush=True)
        raise
    print(json.dumps(marker, sort_keys=True), flush=True)


_boot_marker()

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
from pathlib import Path, PurePosixPath
from typing import Any

INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
OUT = WORK / "out"
REPO = Path("/tmp/wnn_parsed_cache_source")
SOURCE_DATASET = "vladkochetov07/wnn-parsed-cache-source-20261009-f2"
EXPECTED_ASSET_MANIFEST_SHA256 = "f6af5a954f626538e69b79d086e5f89a98b260048d7b7421f104a9b54b531b53"
EXPECTED_ARCHIVE_SHA256 = "8f3e60f40ef7135abdd3926aba2a068fe65933d1e98d010d69784724f75e43a7"
EXPECTED_CODE_MANIFEST_SHA256 = "b5de668eed86f527d639e7077cc827318c5b91426e4a880ac39f54c99cd83084"
EXPECTED_MEMBERSHIP_SHA256 = "6fce1d709391fd2d8a8d0891ad6ebce207f70a3bf2ef15eff4bbaabf1e1baf08"
EXPECTED_SOURCE_COMMIT = "d5daaa0f90b0362877522703c7e17aa78c19cac9"
SOURCE_ARCHIVE_PATH = "wnn_src.tar.gz"
CODE_MANIFEST_PATH = "source-code-manifest.json"
MEMBERSHIP_PATH = "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1/membership_fold2_f03_s20261009.json"
CODE_FILES = (
    "src/wnn/__init__.py",
    "src/wnn/data.py",
    "src/wnn/splits.py",
    "src/wnn/parsed_sequence_cache.py",
    "src/wnn/parsed_sequence_cache_reader.py",
    "scripts/parsed_sequence_cache.py",
    "scripts/profile_parsed_sequence_cache.py",
)
TRAIN_BYTES = 29_240_219_011
VALID_BYTES = 5_615_509_128
CACHE_ARRAY_PAYLOAD_BYTES = 7_732_440_000
OUTPUT_RESERVE_BYTES = 2_000_000_000
CHECKPOINT_RESERVE_BYTES = 0
FREE_RESERVE_BYTES = 4_000_000_000
OUTPUT_CAP_BYTES = 12_000_000_000
MIN_MEMORY_HEADROOM_BYTES = 2_000_000_000
KERNEL_WALL_SECONDS = 3_600
WORKER_WALL_SECONDS = 3_300
SAMPLE_SECONDS = 10
PHASE_LOG = WORK / "phase.jsonl"
_STARTED = time.monotonic()
_SAMPLES: list[dict[str, Any]] = []


class StageRejected(RuntimeError):
    """The mounted inputs or runtime envelope failed the sealed source stage."""


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _phase(name: str, **fields: Any) -> None:
    record = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.monotonic() - _STARTED,
        "phase": name,
        **fields,
    }
    WORK.mkdir(parents=True, exist_ok=True)
    with PHASE_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)


def _find_files(root: Path, filename: str, max_depth: int = 7) -> list[Path]:
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


def discover_raw_root(input_root: Path = INPUT) -> Path:
    """Locate one nested raw dataset mount without assuming Kaggle's path layout."""
    train_files = _find_files(input_root, "train.parquet")
    roots = sorted(
        {path.parent for path in train_files if (path.parent / "valid.parquet").is_file()}
    )
    if len(roots) != 1:
        raise StageRejected(f"expected one raw-data root, found {len(roots)}")
    return roots[0]


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise StageRejected("asset path is not a normalized relative path")
    return path


def discover_source_asset(input_root: Path = INPUT) -> tuple[Path, dict[str, Any]]:
    """Verify the exact private source asset manifest and each mounted payload."""
    manifests = _find_files(input_root, "assets-manifest.json")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in manifests:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("dataset") == SOURCE_DATASET:
            matches.append((path, value))
    if len(matches) != 1:
        raise StageRejected(f"expected one source asset manifest, found {len(matches)}")
    manifest_path, manifest = matches[0]
    observed_manifest_sha = _sha256_file(manifest_path)
    if observed_manifest_sha != EXPECTED_ASSET_MANIFEST_SHA256:
        raise StageRejected("source asset manifest SHA-256 differs from the sealed pin")
    if (
        manifest.get("version") != 1
        or manifest.get("private") is not True
        or manifest.get("dataset") != SOURCE_DATASET
    ):
        raise StageRejected("unsupported source asset manifest identity")
    expected_paths = {SOURCE_ARCHIVE_PATH, CODE_MANIFEST_PATH, MEMBERSHIP_PATH}
    entries = manifest.get("files")
    if (
        not isinstance(entries, list)
        or len(entries) != len(expected_paths)
        or {row.get("path") for row in entries} != expected_paths
    ):
        raise StageRejected("source asset manifest has a different file set")
    for entry in entries:
        relative = _safe_relative(str(entry["path"]))
        filename = str(entry["filename"])
        if Path(filename).name != filename or not filename.endswith(".wnn"):
            raise StageRejected("source asset filenames must be flat opaque .wnn files")
        payload = manifest_path.parent / filename
        if not payload.is_file() or payload.stat().st_size != int(entry["size"]):
            raise StageRejected(f"source asset size or path mismatch: {relative}")
        if _sha256_file(payload) != entry["sha256"]:
            raise StageRejected(f"source asset SHA-256 mismatch: {relative}")
    archive_entry = next(row for row in entries if row["path"] == SOURCE_ARCHIVE_PATH)
    manifest_entry = next(row for row in entries if row["path"] == CODE_MANIFEST_PATH)
    if archive_entry["sha256"] != EXPECTED_ARCHIVE_SHA256:
        raise StageRejected("source archive SHA-256 differs from the sealed pin")
    if manifest_entry["sha256"] != EXPECTED_CODE_MANIFEST_SHA256:
        raise StageRejected("source-code manifest SHA-256 differs from the sealed pin")
    return manifest_path.parent, manifest


def unpack_source(asset_root: Path, manifest: dict[str, Any], repo: Path = REPO) -> Path:
    """Extract the pinned minimal source closure and verify every source file."""
    if repo.exists() or repo.is_symlink():
        raise StageRejected("source extraction directory already exists")
    asset_entries = {row["path"]: row for row in manifest["files"]}
    code_manifest_path = asset_root / asset_entries[CODE_MANIFEST_PATH]["filename"]
    code_manifest = json.loads(code_manifest_path.read_text(encoding="utf-8"))
    if (
        code_manifest.get("schema") != "wnn.kaggle-source-code-manifest"
        or code_manifest.get("format_version") != 1
        or code_manifest.get("commit") != EXPECTED_SOURCE_COMMIT
        or code_manifest.get("files") is None
        or tuple(row["path"] for row in code_manifest["files"]) != CODE_FILES
    ):
        raise StageRejected("source-code manifest does not match the frozen commit/file closure")
    archive = asset_root / asset_entries[SOURCE_ARCHIVE_PATH]["filename"]
    repo.mkdir(parents=True)
    seen: set[str] = set()
    import tarfile

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = str(_safe_relative(member.name))
            if name not in CODE_FILES or name in seen or not member.isfile():
                raise StageRejected("source archive contains an unexpected or repeated entry")
            destination = repo / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise StageRejected(f"source archive entry is unreadable: {name}")
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, 1 << 20)
            seen.add(name)
    if seen != set(CODE_FILES):
        raise StageRejected("source archive is missing declared code files")
    for entry in code_manifest["files"]:
        path = repo / _safe_relative(entry["path"])
        if path.stat().st_size != int(entry["size"]) or _sha256_file(path) != entry["sha256"]:
            raise StageRejected(f"extracted source file differs from commit: {entry['path']}")
    membership_entry = asset_entries[MEMBERSHIP_PATH]
    membership_source = asset_root / membership_entry["filename"]
    membership_destination = repo / MEMBERSHIP_PATH
    membership_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(membership_source, membership_destination)
    if _sha256_file(membership_destination) != EXPECTED_MEMBERSHIP_SHA256:
        raise StageRejected("copied membership differs from the frozen SHA-256")
    (repo / "datasets").mkdir(exist_ok=False)
    return repo


def _cgroup_candidates() -> list[Path]:
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
    candidates.extend([Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")])
    return list(dict.fromkeys(candidates))


def cgroup_memory_snapshot() -> dict[str, Any]:
    """Read memory totals and file-cache/writeback categories from cgroup v1/v2."""
    for directory in _cgroup_candidates():
        try:
            result: dict[str, int | str] = {"cgroup_path": str(directory)}
            for field, names in (
                ("current_bytes", ("memory.current", "memory.usage_in_bytes")),
                ("max_bytes", ("memory.max", "memory.limit_in_bytes")),
            ):
                for name in names:
                    candidate = directory / name
                    if candidate.is_file():
                        raw = candidate.read_text(encoding="utf-8").strip()
                        if raw != "max":
                            value = int(raw)
                            result[f"cgroup_{field}"] = (
                                None if field == "max_bytes" and value >= (1 << 60) else value
                            )
                        break
            stat_path = directory / "memory.stat"
            if stat_path.is_file():
                for line in stat_path.read_text(encoding="utf-8").splitlines():
                    key, value = line.split()
                    if key in {
                        "anon",
                        "file",
                        "file_dirty",
                        "file_writeback",
                        "inactive_file",
                        "active_file",
                    }:
                        result[f"cgroup_{key}_bytes"] = int(value)
            if "cgroup_current_bytes" in result:
                return result
        except (OSError, ValueError):
            continue
    return {"cgroup_path": "unavailable"}


def _process_snapshot(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": pid}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            key, _, raw = line.partition(":")
            if key in {"Rss", "Pss"}:
                result[f"process_{key.lower()}_bytes"] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        result["process_rss_bytes"] = None
        result["process_pss_bytes"] = None
    try:
        io_values = {}
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, _, raw = line.partition(":")
            if raw:
                io_values[key] = int(raw.strip())
        result["process_io"] = io_values
    except (OSError, ValueError):
        result["process_io"] = None
    return result


def _tree_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            try:
                path = Path(directory) / name
                if not path.is_symlink():
                    total += path.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


def _mem_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _sample(pid: int | None, label: str) -> dict[str, Any]:
    row = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.monotonic() - _STARTED,
        "phase": label,
        "output_bytes": _tree_bytes(OUT),
        "output_free_bytes": shutil.disk_usage(WORK).free,
        "memory": cgroup_memory_snapshot(),
        "mem_available_bytes": _mem_available_bytes(),
        "child": _process_snapshot(pid) if pid is not None else None,
    }
    _SAMPLES.append(row)
    _write_json(WORK / "heartbeat.json", row)
    return row


def _abort_reason(sample: dict[str, Any], deadline: float) -> str | None:
    if time.monotonic() >= deadline:
        return "source-stage worker deadline reached"
    if sample["output_bytes"] > OUTPUT_CAP_BYTES:
        return f"working output exceeded {OUTPUT_CAP_BYTES} bytes"
    memory = sample["memory"]
    current = memory.get("cgroup_current_bytes")
    maximum = memory.get("cgroup_max_bytes")
    if (
        isinstance(current, int)
        and isinstance(maximum, int)
        and maximum - current < MIN_MEMORY_HEADROOM_BYTES
    ):
        return "cgroup memory headroom fell below 2 GB"
    available = sample.get("mem_available_bytes")
    if isinstance(available, int) and available < MIN_MEMORY_HEADROOM_BYTES:
        return "host MemAvailable fell below 2 GB"
    return None


def require_disk_headroom(
    free_bytes: int,
    *,
    cache_bytes: int = CACHE_ARRAY_PAYLOAD_BYTES,
    output_reserve_bytes: int = OUTPUT_RESERVE_BYTES,
    checkpoint_reserve_bytes: int = CHECKPOINT_RESERVE_BYTES,
    free_reserve_bytes: int = FREE_RESERVE_BYTES,
) -> int:
    """Fail before source parsing unless cache and all declared reserves fit."""
    required = (
        cache_bytes
        + output_reserve_bytes
        + checkpoint_reserve_bytes
        + free_reserve_bytes
        + 2_000_000
    )
    if free_bytes < required:
        raise StageRejected(f"working disk free {free_bytes} < required {required} bytes")
    return required


def _stop_child(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run_child(
    label: str, argv: list[str], log_path: Path, deadline: float, repo: Path
) -> dict[str, Any]:
    """Run one bounded CLI phase and sample its physical memory, I/O and output."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    child_samples: list[dict[str, Any]] = []
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            argv,
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        _phase("child_started", label=label, pid=process.pid, argv=argv)
        try:
            while process.poll() is None:
                sample = _sample(process.pid, label)
                child_samples.append(sample)
                reason = _abort_reason(sample, deadline)
                if reason:
                    _phase("child_guard_stop", label=label, reason=reason)
                    _stop_child(process)
                    raise StageRejected(reason)
                time.sleep(min(SAMPLE_SECONDS, max(0.1, deadline - time.monotonic())))
            returncode = process.wait()
            log.flush()
            os.fsync(log.fileno())
            terminal_sample = _sample(process.pid, f"{label}_terminal")
            child_samples.append(terminal_sample)
            reason = _abort_reason(terminal_sample, deadline)
            if reason:
                raise StageRejected(reason)
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, argv)
        except BaseException:
            _stop_child(process)
            log.flush()
            os.fsync(log.fileno())
            raise
    child_io = [
        row["child"]["process_io"]
        for row in child_samples
        if isinstance(row.get("child"), dict) and isinstance(row["child"].get("process_io"), dict)
    ]
    first_io = child_io[0] if child_io else None
    last_io = child_io[-1] if child_io else None
    return {
        "label": label,
        "argv": argv,
        "returncode": 0,
        "wall_seconds": time.monotonic() - started,
        "log_path": str(log_path),
        "log_sha256": _sha256_file(log_path),
        "sample_count": len(child_samples),
        "process_io_first": first_io,
        "process_io_last": last_io,
        "process_io_delta": (
            {key: int(last_io.get(key, 0)) - int(first_io.get(key, 0)) for key in last_io}
            if isinstance(first_io, dict) and isinstance(last_io, dict)
            else None
        ),
        "peak_rss_bytes": max(
            (
                int(row["child"]["process_rss_bytes"])
                for row in child_samples
                if isinstance(row.get("child"), dict)
                and isinstance(row["child"].get("process_rss_bytes"), int)
            ),
            default=None,
        ),
        "peak_pss_bytes": max(
            (
                int(row["child"]["process_pss_bytes"])
                for row in child_samples
                if isinstance(row.get("child"), dict)
                and isinstance(row["child"].get("process_pss_bytes"), int)
            ),
            default=None,
        ),
    }


def _resource_summary() -> dict[str, Any]:
    memory_rows = [row["memory"] for row in _SAMPLES]
    process_rows = [row["child"] for row in _SAMPLES if row.get("child")]
    keys = (
        "cgroup_current_bytes",
        "cgroup_max_bytes",
        "cgroup_anon_bytes",
        "cgroup_file_bytes",
        "cgroup_inactive_file_bytes",
        "cgroup_file_dirty_bytes",
        "cgroup_file_writeback_bytes",
    )
    maxima = {
        key: max(
            (int(row[key]) for row in memory_rows if isinstance(row.get(key), int)),
            default=None,
        )
        for key in keys
    }
    return {
        "sample_count": len(_SAMPLES),
        "sample_interval_seconds": SAMPLE_SECONDS,
        "cgroup_peak_bytes": maxima,
        "cgroup_min_headroom_bytes": min(
            (
                int(row["cgroup_max_bytes"]) - int(row["cgroup_current_bytes"])
                for row in memory_rows
                if isinstance(row.get("cgroup_max_bytes"), int)
                and isinstance(row.get("cgroup_current_bytes"), int)
            ),
            default=None,
        ),
        "mem_available_min_bytes": min(
            (
                int(row["mem_available_bytes"])
                for row in _SAMPLES
                if isinstance(row.get("mem_available_bytes"), int)
            ),
            default=None,
        ),
        "child_peak_rss_bytes": max(
            (
                int(row["process_rss_bytes"])
                for row in process_rows
                if isinstance(row.get("process_rss_bytes"), int)
            ),
            default=None,
        ),
        "child_peak_pss_bytes": max(
            (
                int(row["process_pss_bytes"])
                for row in process_rows
                if isinstance(row.get("process_pss_bytes"), int)
            ),
            default=None,
        ),
        "output_peak_bytes": max((int(row["output_bytes"]) for row in _SAMPLES), default=0),
        "sample_artifact": "heartbeat.json is latest sample; phase.jsonl records phase transitions",
    }


def _run() -> dict[str, Any]:
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "logs").mkdir()
    deadline = _STARTED + WORKER_WALL_SECONDS
    kernel_deadline = _STARTED + KERNEL_WALL_SECONDS
    _phase(
        "source_stage_start",
        worker_deadline_seconds=WORKER_WALL_SECONDS,
        kernel_deadline_seconds=KERNEL_WALL_SECONDS,
    )

    raw_root = discover_raw_root()
    asset_root, asset_manifest = discover_source_asset()
    repo = unpack_source(asset_root, asset_manifest)
    (repo / "datasets" / "train.parquet").symlink_to(raw_root / "train.parquet")
    (repo / "datasets" / "valid.parquet").symlink_to(raw_root / "valid.parquet")
    membership = repo / MEMBERSHIP_PATH
    if _sha256_file(membership) != EXPECTED_MEMBERSHIP_SHA256:
        raise StageRejected("runtime membership SHA-256 mismatch")

    cache_dir = OUT / "parsed_cache"
    build_argv = [
        sys.executable,
        str(repo / "scripts/parsed_sequence_cache.py"),
        "build",
        "--membership",
        str(membership),
        "--expected-membership-sha256",
        EXPECTED_MEMBERSHIP_SHA256,
        "--source-root",
        str(repo),
        "--output-dir",
        str(cache_dir),
        "--output-reserve-bytes",
        str(OUTPUT_RESERVE_BYTES),
        "--checkpoint-reserve-bytes",
        str(CHECKPOINT_RESERVE_BYTES),
        "--free-reserve-bytes",
        str(FREE_RESERVE_BYTES),
    ]
    initial_free = shutil.disk_usage(WORK).free
    minimum_free = require_disk_headroom(initial_free)
    if shutil.disk_usage(OUT).free < minimum_free:
        raise StageRejected("output filesystem fails cache plus reserve admission")
    build_result = run_child(
        "cache_build", build_argv, OUT / "logs/cache_build.log", deadline, repo
    )

    manifest_path = cache_dir / "manifest.json"
    manifest_sha = _sha256_file(manifest_path)
    profile_path = cache_dir / "reader_profile.json"
    profile_argv = [
        sys.executable,
        str(repo / "scripts/profile_parsed_sequence_cache.py"),
        "--manifest",
        str(manifest_path),
        "--expected-manifest-sha256",
        manifest_sha,
        "--validation-receipt",
        str(cache_dir / "reader_validation.json"),
        "--profile-output",
        str(profile_path),
        "--per-role",
        "2",
        "--seed",
        "20261009",
    ]
    profile_result = run_child(
        "cache_profile", profile_argv, OUT / "logs/cache_profile.log", deadline, repo
    )
    if time.monotonic() >= kernel_deadline:
        raise StageRejected("kernel wall limit reached before source-stage finalization")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_receipt_path = cache_dir / "source_verification.json"
    validation_path = cache_dir / "reader_validation.json"
    receipt = {
        "schema": "wnn.kaggle-parsed-cache-source-stage",
        "format_version": 1,
        "state": "complete",
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_asset_dataset": SOURCE_DATASET,
        "source_asset_manifest_sha256": EXPECTED_ASSET_MANIFEST_SHA256,
        "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "source_code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
        "membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "membership_counts": manifest["member_counts"],
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "portable_registry_sha256": manifest["portable_registry_sha256"],
        "raw_source_bytes_expected": {"train": TRAIN_BYTES, "valid": VALID_BYTES},
        "raw_source_verification_sha256": _sha256_file(source_receipt_path),
        "raw_source_verification": json.loads(source_receipt_path.read_text(encoding="utf-8")),
        "cache_manifest_sha256": manifest_sha,
        "cache_key": manifest["cache_key"],
        "producer_manifest": manifest,
        "reader_validation_sha256": _sha256_file(validation_path),
        "reader_profile_sha256": _sha256_file(profile_path),
        "reader_profile": json.loads(profile_path.read_text(encoding="utf-8")),
        "build_command": build_result,
        "profile_command": profile_result,
        "disk": {
            "initial_free_bytes": initial_free,
            "final_free_bytes": shutil.disk_usage(WORK).free,
            "output_bytes": _tree_bytes(OUT),
            "output_cap_bytes": OUTPUT_CAP_BYTES,
            "output_reserve_bytes": OUTPUT_RESERVE_BYTES,
            "checkpoint_reserve_bytes": CHECKPOINT_RESERVE_BYTES,
            "free_reserve_bytes": FREE_RESERVE_BYTES,
        },
        "resource_profile": _resource_summary(),
        "source_root_mount": str(raw_root),
        "terminal_elapsed_seconds": time.monotonic() - _STARTED,
        "gpu_enabled": False,
        "tpu_enabled": False,
    }
    _write_json(OUT / "source_stage_receipt.json", receipt)
    _write_json(
        WORK / "final_status.json",
        {"state": "complete", "receipt": "out/source_stage_receipt.json"},
    )
    _phase(
        "source_stage_complete",
        cache_manifest_sha256=manifest_sha,
        source_verification_sha256=receipt["raw_source_verification_sha256"],
        output_bytes=receipt["disk"]["output_bytes"],
        elapsed_seconds=receipt["terminal_elapsed_seconds"],
    )
    return receipt


def run() -> int:
    """Execute the source-stage with durable diagnostics and bounded subprocesses."""
    try:
        receipt = _run()
        print(
            json.dumps(
                {
                    "state": "complete",
                    "cache_manifest_sha256": receipt["cache_manifest_sha256"],
                    "reader_validation_sha256": receipt["reader_validation_sha256"],
                    "output_bytes": receipt["disk"]["output_bytes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        failure = {
            "state": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - _STARTED,
            "resource_profile": _resource_summary(),
            "output_bytes": _tree_bytes(OUT),
        }
        try:
            _write_json(OUT / "failure.json", failure)
            _write_json(WORK / "final_status.json", failure)
            _phase("source_stage_failed", error_type=failure["error_type"], error=failure["error"])
        except OSError:
            print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 1
raise SystemExit(run())
