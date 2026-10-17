"""Run a submission ZIP under a one-core CPU budget on Linux.

The receipt covers only the supplied Parquet panel; it makes no claim about
hidden evaluation data or competition-host qualification.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from uuid import uuid4

MAX_ZIP_BYTES = 20_000_000
MEMORY_BYTES = 16_000_000_000
MAX_EXTRACTED_BYTES = MEMORY_BYTES
TIMEOUT_SECONDS = 3600
SEQUENCE_LENGTH = 20_000
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_identity(path: Path) -> tuple[int, ...]:
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def inspect_zip(path: Path) -> tuple[int, str, int]:
    size = path.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise ValueError(f"submission ZIP is {size} bytes; limit is {MAX_ZIP_BYTES}")
    names: set[str] = set()
    has_solution = False
    expanded_bytes = 0
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            expanded_bytes += entry.file_size
            if expanded_bytes > MAX_EXTRACTED_BYTES:
                raise ValueError("ZIP expansion exceeds the verifier's 16 GB extraction guard")
            name = entry.filename
            posix, windows = PurePosixPath(name), PureWindowsPath(name)
            parts = name.rstrip("/").split("/")
            if (
                not name
                or "\x00" in name
                or "\\" in name
                or posix.is_absolute()
                or windows.is_absolute()
                or windows.drive
                or any(part in ("", ".", "..") for part in parts)
            ):
                raise ValueError(f"unsafe ZIP entry path: {name!r}")
            normalized = "/".join(parts)
            if normalized in names:
                raise ValueError(f"duplicate ZIP entry path: {name!r}")
            names.add(normalized)
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.flag_bits & 1:
                raise ValueError(f"unsupported ZIP entry type or encryption: {name!r}")
            has_solution |= normalized == "solution.py" and not entry.is_dir()
    if not has_solution:
        raise ValueError("submission ZIP must contain solution.py at its root")
    return size, sha256(path), expanded_bytes


def filesystem_type(path: Path) -> str:
    best_mount, best_type = Path("/"), "unknown"
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mount = Path(fields[1].replace("\\040", " ").replace("\\134", "\\"))
        if (path == mount or mount in path.parents) and len(str(mount)) >= len(str(best_mount)):
            best_mount, best_type = mount, fields[2]
    return best_type


def host_facts(cpu_id: int) -> dict[str, Any]:
    cpu_model = platform.processor() or None
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith(("model name", "hardware")) and ":" in line:
            cpu_model = line.split(":", 1)[1].strip()
            break
    ram_bytes = None
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            ram_bytes = int(line.split()[1]) * 1024
            break
    return {
        "cpu_model": cpu_model,
        "ram_bytes": ram_bytes,
        "machine": platform.machine(),
        "cpuset_cpu": cpu_id,
    }


def docker(*args: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def inspect_container(container_id: str) -> dict[str, Any]:
    result = docker("inspect", "--format", "{{json .State}}", container_id)
    if result.returncode:
        raise RuntimeError(f"docker inspect failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def verifier_result(path: Path) -> dict[str, Any]:
    last = ""
    with path.open(encoding="utf-8") as log:
        for line in log:
            if line.strip():
                last = line.strip()
    result = json.loads(last)
    if not isinstance(result, dict) or "rows" not in result or "metric_id" not in result:
        raise ValueError("last container output line is not a verifier result")
    return result


def run_budget(
    submission: Path, data: Path, output_dir: Path, image: str, expected_rows: int
) -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("this CPU budget gate runs only on Linux")
    if expected_rows <= 0:
        raise ValueError("--expected-rows must be the positive supplied-panel row count")
    allowed_cpus = os.sched_getaffinity(0)
    if not allowed_cpus:
        raise RuntimeError("Linux did not report an available CPU for single-core pinning")
    cpu_id = min(allowed_cpus)
    submission, data = (
        submission.expanduser().resolve(strict=True),
        data.expanduser().resolve(strict=True),
    )
    if not submission.is_file() or not data.is_file():
        raise ValueError("submission and data must be regular files")
    input_identities = {path: input_identity(path) for path in (submission, data)}
    zip_bytes, zip_hash, expanded_bytes = inspect_zip(submission)
    data_hash = sha256(data)
    data_bytes = data.stat().st_size
    verify_script, utils_module = ROOT / "scripts/verify_submission.py", ROOT / "utils.py"
    for source in (verify_script, utils_module):
        if not source.is_file():
            raise FileNotFoundError(source)
    source_hashes = {
        str(source.relative_to(ROOT)): sha256(source) for source in (verify_script, utils_module)
    }
    image_check = docker("image", "inspect", "--format", "{{.Id}}", image)
    if image_check.returncode or not image_check.stdout.strip():
        raise RuntimeError(f"Docker image {image!r} is unavailable: {image_check.stderr.strip()}")
    image_id = image_check.stdout.strip()

    output_dir = output_dir.expanduser().absolute()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"output directory must be new: {output_dir}")
    output_dir.mkdir(mode=0o700)
    output_dir = output_dir.resolve(strict=True)
    fs_type = filesystem_type(output_dir)
    if fs_type in {"tmpfs", "ramfs", "devtmpfs"}:
        raise ValueError(f"output directory must use disk-backed storage, not {fs_type}")
    # The verifier keeps predictions, targets and mask (17 bytes/row), then
    # copies them to the output directory before deleting the scratch versions.
    disk_required = expanded_bytes + expected_rows * 34 + 64 * 1024 * 1024
    disk_available = shutil.disk_usage(output_dir).free
    if disk_available < disk_required:
        raise ValueError(
            f"need at least {disk_required} disk bytes; only {disk_available} available"
        )
    (output_dir / "scratch").mkdir()
    log_path = output_dir / "docker.log"

    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": "/out/scratch",
        "TMP": "/out/scratch",
        "TEMP": "/out/scratch",
    }
    name = f"submission-budget-{uuid4().hex[:12]}"
    create = [
        "create",
        "--name",
        name,
        "--cpus",
        "1",
        "--cpuset-cpus",
        str(cpu_id),
        "--memory",
        str(MEMORY_BYTES),
        "--memory-swap",
        str(MEMORY_BYTES),
        "--network",
        "none",
        "--read-only",
    ]
    for source, target in (
        (verify_script, "/tools/scripts/verify_submission.py"),
        (utils_module, "/tools/utils.py"),
        (submission, "/input/submission.zip"),
        (data, "/data/panel.parquet"),
    ):
        create.extend(("--mount", f"type=bind,source={source},target={target},readonly"))
    create.extend(("--mount", f"type=bind,source={output_dir},target=/out"))
    for key, value in env.items():
        create.extend(("--env", f"{key}={value}"))
    verify_entry = (
        "import runpy, sys; "
        "sys.version_info[:2] == (3, 11) or sys.exit('CPU budget requires Python 3.11'); "
        "sys.argv = sys.argv[1:]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    create.extend(
        (
            "--entrypoint",
            "python",
            image_id,
            "-u",
            "-B",
            "-c",
            verify_entry,
            "/tools/scripts/verify_submission.py",
            "/input/submission.zip",
            "/data/panel.parquet",
            "/out/pred.npy",
        )
    )

    started = time.monotonic()
    container_id, completed = None, None
    state: dict[str, Any] = {}
    status, error, elapsed = "not_started", None, 0.0
    result: dict[str, Any] | None = None
    try:
        created = docker(*create, timeout=60)
        if created.returncode or not created.stdout.strip():
            raise RuntimeError(f"docker create failed: {created.stderr.strip()}")
        container_id, status = created.stdout.strip(), "running"
        remaining = TIMEOUT_SECONDS - (time.monotonic() - started)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                ["docker", "start", "--attach", container_id],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=max(0.01, remaining),
                check=False,
            )
        elapsed = time.monotonic() - started
    except subprocess.TimeoutExpired:
        elapsed, status = time.monotonic() - started, "timed_out"
        error = f"verification exceeded {TIMEOUT_SECONDS} seconds"
        if container_id:
            with contextlib.suppress(Exception):
                docker("kill", container_id, timeout=30)
    except Exception as exc:
        elapsed, status = time.monotonic() - started, "failed"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if container_id:
            try:
                state = inspect_container(container_id)
                if state.get("Status") == "running":
                    docker("kill", container_id, timeout=30)
                    state = inspect_container(container_id)
            except Exception as exc:
                error = error or f"container inspection failed: {exc}"
        oom = state.get("OOMKilled")
        if completed is not None and completed.returncode == 0 and not error:
            try:
                result = verifier_result(log_path)
                if elapsed > TIMEOUT_SECONDS:
                    status, error = "over_time", "elapsed budget exceeded"
                elif state.get("ExitCode") != 0 or oom is not False:
                    status, error = "container_failed", "container exit/OOM state was not clean"
                elif result.get("rows") != expected_rows:
                    status, error = (
                        "row_coverage_mismatch",
                        "verified rows differ from --expected-rows",
                    )
                else:
                    status = "passed_for_supplied_dataset"
            except Exception as exc:
                status, error = "invalid_verifier_output", f"{type(exc).__name__}: {exc}"
        elif status == "running":
            status, error = (
                "container_failed",
                error or f"docker start exited {completed.returncode}",
            )
        if oom is True:
            status, error = "out_of_memory", "Docker reports OOMKilled=true"
        try:
            source_unchanged = all(
                sha256(ROOT / name) == digest for name, digest in source_hashes.items()
            )
        except OSError:
            source_unchanged = False
        if not source_unchanged:
            status, error = "source_changed", "verifier/utils source changed during the budget run"
        try:
            inputs_unchanged = all(
                input_identity(path) == identity for path, identity in input_identities.items()
            )
        except OSError:
            inputs_unchanged = False
        if not inputs_unchanged:
            status, error = (
                "input_changed",
                "submission ZIP or dataset changed during the budget run",
            )
        removed = True
        if container_id:
            try:
                removed = docker("rm", "-f", container_id, timeout=30).returncode == 0
            except Exception:
                removed = False
        if not removed:
            status, error = (
                "container_cleanup_failed",
                "Docker could not remove the created container",
            )
        receipt = {
            "schema_version": 1,
            "status": status,
            "passed_for_supplied_dataset": status == "passed_for_supplied_dataset",
            "claim_scope": "supplied Parquet dataset only; hidden evaluation coverage is not established",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "submission_zip": {
                "path": str(submission),
                "bytes": zip_bytes,
                "sha256": zip_hash,
                "expanded_bytes": expanded_bytes,
            },
            "scratch_disk_preflight": {
                "required_bytes": disk_required,
                "available_bytes": disk_available,
                "expanded_zip_limit_bytes": MAX_EXTRACTED_BYTES,
            },
            "image_reference": image,
            "image_id": image_id,
            "host": host_facts(cpu_id),
            "output_filesystem": fs_type,
            "limits": {
                "cpus": 1,
                "cpuset_cpu": cpu_id,
                "memory_bytes": MEMORY_BYTES,
                "memory_swap_bytes": MEMORY_BYTES,
                "network": "none",
                "root_filesystem_read_only": True,
                "timeout_seconds": TIMEOUT_SECONDS,
            },
            "elapsed_seconds": round(elapsed, 3),
            "oom_killed": oom,
            "container_exit_code": state.get("ExitCode"),
            "container_removed": removed,
            "sources": {"sha256": source_hashes, "unchanged_during_run": source_unchanged},
            "inputs_unchanged_during_run": inputs_unchanged,
            "python_version_required": "3.11",
            "dataset": {
                "path": str(data),
                "bytes": data_bytes,
                "sha256": data_hash,
                "expected_rows": expected_rows,
                "verified_rows": None if result is None else result.get("rows"),
                "verified_sequences": None if result is None else result.get("sequences"),
                "coverage_complete": result is not None and result.get("rows") == expected_rows,
            },
            "verification": result,
            "error": error,
        }
        (output_dir / "budget_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if status != "passed_for_supplied_dataset":
        print(f"submission budget gate failed ({status}): {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, type=Path, help="actual submission ZIP")
    parser.add_argument("--data", required=True, type=Path, help="supplied Parquet panel")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="new disk-backed receipt/output directory"
    )
    parser.add_argument("--image", required=True, help="existing CPU budget Docker image")
    parser.add_argument(
        "--expected-rows",
        required=True,
        type=int,
        help="actual rows in the supplied full panel, not hidden evaluation rows",
    )
    args = parser.parse_args()
    try:
        code = run_budget(
            args.submission, args.data, args.output_dir, args.image, args.expected_rows
        )
    except Exception as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
