#!/usr/bin/env python3
"""Bounded CPU-only Kaggle driver for the admitted age-readout grade.

The local packet builder fills the four transport/admission pins after the bundle is frozen.
The archive is an opaque, manifest-first gzip tar; only declared regular files
are extracted to `/kaggle/working/disk_scratch`. No fitting or export occurs.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from contextlib import suppress
from pathlib import Path, PurePosixPath

# Manifest and receipt values are heterogenous JSON checked before use.
from typing import Any, BinaryIO

BUNDLE_DATASET_SLUG = ""
BUNDLE_ARCHIVE_SHA256 = ""
ADMISSION_RELATIVE_PATH = ""
ADMISSION_SHA256 = ""
BUNDLE_ARCHIVE_NAME = "grade_bundle.wnn"
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
GRADE_SCRIPT = "grader/age_readout_grade.py"
INPUT_ROOT = Path("/kaggle/input")
SCRATCH_ROOT = Path("/kaggle/working/disk_scratch")
MAX_JOB_SECONDS = 1_800
FINALIZATION_RESERVE_SECONDS = 60
MAX_RSS_BYTES = 4_000_000_000
MAX_BYTES = 1_000_000_000
MAX_MANIFEST_BYTES = 16_000_000
CHUNK_BYTES = 8 * 1024**2
POLL_SECONDS = 1.0
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_RE = re.compile(
    r"(^|[/_.-])(token|secret|credentials?|password|kaggle\.json|\.env)([/_.-]|$)",
    re.IGNORECASE,
)


def safe_path(value: str) -> str:
    """Return a canonical relative POSIX path or reject traversal and aliases."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or str(path) != value
        or FORBIDDEN_RE.search(value)
        or value.startswith(".git/")
    ):
        raise ValueError("Unsafe bundle path")
    return str(path)


def parse_manifest(payload: bytes) -> tuple[dict[str, dict[str, Any]], int]:
    """Validate the archive allowlist and its declared expanded size."""
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("Bundle manifest exceeds its size limit")
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or value.get("schema") != "wnn.age-readout-grade-bundle.v1"
        or not isinstance(value.get("source_root"), str)
        or not value["source_root"]
        or not isinstance(value.get("source_path"), str)
        or not value["source_path"]
        or not isinstance(value.get("files"), list)
    ):
        raise ValueError("Unsupported grade-bundle manifest")
    files: dict[str, dict[str, Any]] = {}
    total = 0
    for item in value["files"]:
        if not isinstance(item, dict) or set(item) != {
            "source_path",
            "bundle_path",
            "sha256",
            "bytes",
        }:
            raise ValueError("Invalid bundle file descriptor")
        name = safe_path(item["bundle_path"])
        size, digest = item["bytes"], item["sha256"]
        if (
            name in files
            or not isinstance(item["source_path"], str)
            or not item["source_path"]
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise ValueError("Duplicate path or invalid bundle file identity")
        files[name] = item
        total += size
        if total > MAX_BYTES:
            raise ValueError("Expanded bundle exceeds one gigabyte")
    if not files or BUNDLE_MANIFEST_NAME in files or GRADE_SCRIPT not in files:
        raise ValueError("Bundle omits its manifest or grader")
    return files, total


def check_limits(deadline: float, peak_rss: list[int]) -> None:
    """Enforce the job wall/RSS limits for this process tree."""
    if time.monotonic() >= deadline:
        raise TimeoutError("Job reached its 1800-second cap before finalization")
    import psutil

    root = psutil.Process(os.getpid())
    rss = root.memory_info().rss
    for child in root.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    peak_rss[0] = max(peak_rss[0], rss)
    if rss >= MAX_RSS_BYTES:
        raise MemoryError("Process-tree RSS reached four gigabytes")


def copy_member(
    root: Path,
    name: str,
    source: BinaryIO,
    size: int,
    deadline: float,
    peak_rss: list[int],
) -> None:
    """Stream one already-allowlisted regular file onto working disk."""
    target = root / safe_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    with target.open("xb") as output:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            check_limits(deadline, peak_rss)
            copied += len(block)
            if copied > size:
                raise ValueError("Archive member exceeds its declared size")
            output.write(block)
    if copied != size:
        raise ValueError("Archive member ended before its declared size")


def safe_extract(
    archive: Path, destination: Path, deadline: float, peak_rss: list[int]
) -> tuple[int, int]:
    """Extract only manifest-declared files; reject links, extras, and aliases."""
    if destination.exists():
        raise FileExistsError("Refusing to overwrite the extracted bundle")
    with tarfile.open(archive, mode="r|gz") as tar:
        members = iter(tar)
        first = next(members, None)
        if (
            first is None
            or first.name != BUNDLE_MANIFEST_NAME
            or not first.isfile()
            or first.size > MAX_MANIFEST_BYTES
        ):
            raise ValueError("Archive must start with the small bundle manifest")
        stream = tar.extractfile(first)
        if stream is None:
            raise ValueError("Bundle manifest has no content")
        with stream:
            payload = stream.read(MAX_MANIFEST_BYTES + 1)
        files, total = parse_manifest(payload)
        destination.mkdir(parents=True, mode=0o700)
        allowed = set(files) | {BUNDLE_MANIFEST_NAME}
        seen = {BUNDLE_MANIFEST_NAME}
        copy_member(destination, first.name, io.BytesIO(payload), first.size, deadline, peak_rss)
        for member in members:
            check_limits(deadline, peak_rss)
            name = safe_path(member.name)
            item = files.get(name)
            if not member.isfile() or item is None or name in seen or member.size != item["bytes"]:
                raise ValueError("Archive contains a link, extra file, alias, or size mismatch")
            seen.add(name)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError("Archive file has no content")
            with stream:
                copy_member(destination, name, stream, member.size, deadline, peak_rss)
        if seen != allowed:
            raise ValueError("Archive omits a declared bundle file")
    for path in sorted(destination.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return total, len(files)


def sha256_file(path: Path, deadline: float, peak_rss: list[int]) -> str:
    """Hash a mounted artifact in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            check_limits(deadline, peak_rss)
            digest.update(block)
    return digest.hexdigest()


def worker_environment(scratch: Path, bundle_root: Path) -> dict[str, str]:
    """Put all common temporary/cache paths on Kaggle working disk."""
    paths = {
        "TMPDIR": "tmp",
        "TMP": "tmp",
        "TEMP": "tmp",
        "XDG_CACHE_HOME": "cache/xdg",
        "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
        "TRITON_CACHE_DIR": "cache/triton",
        "NUMBA_CACHE_DIR": "cache/numba",
        "POLARS_TEMP_DIR": "tmp/polars",
        "MPLCONFIGDIR": "cache/matplotlib",
        "HF_HOME": "cache/huggingface",
        "TORCH_HOME": "cache/torch",
        "PIP_CACHE_DIR": "cache/pip",
        "JOBLIB_TEMP_FOLDER": "tmp/joblib",
        "JUPYTER_RUNTIME_DIR": "cache/jupyter",
    }
    environment = {key: str(scratch / value) for key, value in paths.items()}
    for value in set(environment.values()):
        Path(value).mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(bundle_root), str(bundle_root / "src"))),
            "WNN_TMP_ROOT": str(scratch),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "BLIS_NUM_THREADS": "2",
            "VECLIB_MAXIMUM_THREADS": "2",
            "POLARS_MAX_THREADS": "2",
            "JOBLIB_MULTIPROCESSING": "0",
        }
    )
    return environment


def grader_command(root: Path, admission: Path, output: Path) -> list[str]:
    """Build the independent grader's frozen CLI arguments without a shell."""
    return [
        sys.executable,
        str(root / GRADE_SCRIPT),
        "--bundle-root",
        str(root),
        "--bundle-manifest",
        str(root / BUNDLE_MANIFEST_NAME),
        "--admission",
        str(admission),
        "--admission-sha256",
        ADMISSION_SHA256,
        "--output",
        str(output),
    ]


def stop_group(process: subprocess.Popen[bytes]) -> None:
    """Stop only this grader process group, escalating after five seconds."""
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


def remove_scratch_tree(path: Path) -> None:
    """Remove only a generated working tree after the grade has completed."""
    if not path.exists():
        return
    if path.is_symlink():
        raise ValueError("Refusing to recursively remove a scratch symlink")
    path.chmod(0o700)
    entries = list(path.rglob("*"))
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            entry.chmod(0o700)
    for entry in entries:
        if entry.is_symlink():
            entry.unlink()
    shutil.rmtree(path)


def main() -> int:
    """Run one admitted grade and retain logs plus a resource/terminal receipt."""
    started = time.monotonic()
    deadline = started + MAX_JOB_SECONDS - FINALIZATION_RESERVE_SECONDS
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    run_root = SCRATCH_ROOT / f"age_readout_grade_{time.time_ns()}"
    run_root.mkdir(exist_ok=False)
    stdout_path, stderr_path = run_root / "grader.stdout.log", run_root / "grader.stderr.log"
    stdout_path.touch(exist_ok=False)
    stderr_path.touch(exist_ok=False)
    peak_rss = [0]
    phases: dict[str, float] = {}
    command: list[str] = []
    archive_bytes = expanded_bytes = file_count = 0
    exit_code, status, failure = 1, "ERROR", None
    bundle_root = run_root / "bundle"
    process: subprocess.Popen[bytes] | None = None
    cpu_ids: list[int] = []
    try:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", BUNDLE_DATASET_SLUG):
            raise ValueError("Root has not frozen the private bundle dataset slug")
        if SHA256_RE.fullmatch(BUNDLE_ARCHIVE_SHA256) is None:
            raise ValueError("Root has not frozen the bundle archive SHA256")
        if SHA256_RE.fullmatch(ADMISSION_SHA256) is None or not ADMISSION_RELATIVE_PATH:
            raise ValueError("Root has not frozen the admission path and SHA256")
        admission_name = safe_path(ADMISSION_RELATIVE_PATH)
        environment = os.environ.copy()
        environment.update(worker_environment(run_root, bundle_root))
        environment["PYTHONHASHSEED"] = "0"
        os.environ.update(environment)
        cpu_ids = sorted(os.sched_getaffinity(0))[:2]
        if len(cpu_ids) != 2:
            raise RuntimeError("Kaggle CPU worker exposes fewer than two CPUs")
        os.sched_setaffinity(0, set(cpu_ids))
        input_root = Path("/kaggle/input").resolve(strict=True)
        mounted = input_root / BUNDLE_DATASET_SLUG / BUNDLE_ARCHIVE_NAME
        if mounted.is_symlink():
            raise ValueError("Expected a direct read-only Kaggle dataset file")
        archive = mounted.resolve(strict=True)
        if not archive.is_relative_to(input_root) or not archive.is_file():
            raise ValueError("Bundle archive resolves outside Kaggle input")
        mounted_admission = input_root / BUNDLE_DATASET_SLUG / admission_name
        if mounted_admission.is_symlink():
            raise ValueError("Admission must be a direct read-only Kaggle input file")
        admission = mounted_admission.resolve(strict=True)
        if not admission.is_relative_to(input_root) or not admission.is_file():
            raise ValueError("Admission resolves outside Kaggle input")
        archive_bytes = archive.stat().st_size
        if not archive_bytes or archive_bytes > MAX_BYTES:
            raise ValueError("Compressed bundle is empty or exceeds one gigabyte")
        if shutil.disk_usage(SCRATCH_ROOT).free < MAX_BYTES + 500_000_000:
            raise OSError("Insufficient free disk under /kaggle/working/disk_scratch")
        phase_start = time.monotonic()
        if sha256_file(archive, deadline, peak_rss) != BUNDLE_ARCHIVE_SHA256:
            raise ValueError("Mounted bundle archive digest mismatch")
        phases["archive_sha256"] = time.monotonic() - phase_start
        phase_start = time.monotonic()
        expanded_bytes, file_count = safe_extract(archive, bundle_root, deadline, peak_rss)
        phases["safe_extract"] = time.monotonic() - phase_start
        if (
            not admission.is_file()
            or sha256_file(admission, deadline, peak_rss) != ADMISSION_SHA256
        ):
            raise ValueError("Mounted admission receipt digest mismatch")
        output = run_root / "grade_report.json"
        if output.exists():
            raise FileExistsError("Refusing to overwrite the grade report")
        command = grader_command(bundle_root, admission, output)
        phase_start = time.monotonic()
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                command,
                cwd=bundle_root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            while process.poll() is None:
                try:
                    check_limits(deadline, peak_rss)
                except (TimeoutError, MemoryError) as error:
                    failure = str(error)
                    stop_group(process)
                    break
                time.sleep(POLL_SECONDS)
            exit_code = process.wait()
        phases["grader_cli"] = time.monotonic() - phase_start
        if failure:
            raise TimeoutError(failure)
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, command)
        if not output.is_file() or output.is_symlink():
            raise FileNotFoundError("Grader exited zero without a regular report")
        cleanup_started = time.monotonic()
        remove_scratch_tree(bundle_root)
        remove_scratch_tree(run_root / "tmp")
        remove_scratch_tree(run_root / "cache")
        phases["cleanup"] = time.monotonic() - cleanup_started
        status = "COMPLETE"
    except Exception as error:
        failure = failure or f"{type(error).__name__}: {error}"
        if process is not None and process.poll() is None:
            stop_group(process)
        print(f"CPU grade failed; retained logs are under {run_root}")
    phases["total"] = time.monotonic() - started
    receipt = {
        "schema": "wnn.age-readout-grade-kaggle-driver.v1",
        "status": status,
        "exit_code": exit_code,
        "failure": failure,
        "phase_seconds": phases,
        "peak_process_tree_rss_bytes": peak_rss[0],
        "rss_limit_bytes": MAX_RSS_BYTES,
        "job_limit_seconds": MAX_JOB_SECONDS,
        "cpu_affinity_ids": cpu_ids,
        "archive_bytes": archive_bytes,
        "expanded_bytes": expanded_bytes,
        "file_count": file_count,
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "grade_report_path": str(run_root / "grade_report.json"),
    }
    (run_root / "driver_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
