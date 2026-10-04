from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path, PurePosixPath

DATASET_SLUG = "wnn-jax-adamw-e465-cpu-20261004-v1"
ARCHIVE_SHA256 = "79729ec242d565df8ed742a86c1c36b7616a4c7427486c936103cc095ac92538"
MANIFEST_SHA256 = "634a89f12a18f21da74c6887ea95d2024d32e56dd00fa1d9707888137987d072"
WORKER_SECONDS = 300
JOB_SECONDS = 600
RSS_LIMIT_BYTES = 4 * 1024**3
POLL_SECONDS = 0.25
OUTPUT = Path("/kaggle/working/e465_adamw_fp32_cpu_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def proc_group_rss_bytes(group_id: int) -> int:
    total = 0
    page_size = os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].strip().split()
            if int(fields[2]) == group_id:
                resident_pages = int((entry / "statm").read_text().split()[1])
                total += resident_pages * page_size
        except (OSError, ValueError, IndexError):
            continue
    return total


def own_rss_bytes() -> int:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def host_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def verify_expanded(root: Path, files: dict) -> None:
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files):
        raise ValueError(
            f"Expanded member set mismatch: missing={sorted(set(files) - actual)}, extra={sorted(actual - set(files))}"
        )
    for name, descriptor in files.items():
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts or "\\" in name:
            raise ValueError(f"Unsafe source path: {name}")
        path = root / name
        if path.stat().st_size != descriptor["size_bytes"]:
            raise ValueError(f"Source size mismatch: {name}")
        if sha256(path) != descriptor["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    matches = list(Path("/kaggle/input").glob("**/source-manifest.json"))
    if len(matches) != 1:
        write_json(
            OUTPUT / "runner_result.json",
            {"status": "failed_manifest_resolution", "matches": len(matches)},
        )
        return 30
    manifest_path = matches[0]
    manifest = json.loads(manifest_path.read_text())
    (OUTPUT / "source-manifest.json").write_text(manifest_path.read_text())
    actual_manifest_hash = sha256(manifest_path)
    if actual_manifest_hash != MANIFEST_SHA256:
        write_json(
            OUTPUT / "runner_result.json",
            {"status": "failed_manifest_hash", "actual": actual_manifest_hash},
        )
        return 31
    if (
        manifest.get("dataset_id") != f"vladkochetov07/{DATASET_SLUG}"
        or manifest.get("private") is not True
    ):
        write_json(OUTPUT / "runner_result.json", {"status": "failed_dataset_identity_or_privacy"})
        return 32
    if any(
        manifest.get(key) is not False for key in ("enable_gpu", "enable_tpu", "enable_internet")
    ):
        write_json(
            OUTPUT / "runner_result.json", {"status": "failed_accelerator_or_internet_contract"}
        )
        return 33

    bundle_root = manifest_path.parent / "source_bundle"
    archive_candidates = list(manifest_path.parent.glob("source_bundle.zip"))
    observed_archive_hash = None
    if archive_candidates:
        observed_archive_hash = sha256(archive_candidates[0])
        if observed_archive_hash != ARCHIVE_SHA256 or observed_archive_hash != manifest.get(
            "archive_sha256"
        ):
            write_json(
                OUTPUT / "runner_result.json",
                {"status": "failed_archive_hash", "actual": observed_archive_hash},
            )
            return 34
    try:
        verify_expanded(bundle_root, manifest["files"])
    except (OSError, ValueError, KeyError) as error:
        write_json(
            OUTPUT / "runner_result.json",
            {"status": "failed_expanded_source_hashes", "detail": str(error)},
        )
        return 35
    if not (bundle_root / "worker.py").is_file():
        write_json(OUTPUT / "runner_result.json", {"status": "failed_worker_missing"})
        return 36

    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if len(allowed) < 2:
        write_json(
            OUTPUT / "runner_result.json",
            {"status": "failed_host_cpu_affinity", "allowed_cpus": allowed},
        )
        return 37
    memory_bytes = host_memory_bytes()
    if memory_bytes is None or 0.8 * memory_bytes < RSS_LIMIT_BYTES + 512 * 1024**2:
        write_json(
            OUTPUT / "runner_result.json",
            {
                "status": "failed_host_memory_headroom",
                "host_memory_bytes": memory_bytes,
                "rss_limit_bytes": RSS_LIMIT_BYTES,
                "launcher_reserve_bytes": 512 * 1024**2,
                "required_host_headroom_fraction": 0.2,
            },
        )
        return 38
    os.sched_setaffinity(0, set(allowed[:2]))
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "E465_SOURCE_ROOT": str(bundle_root),
        }
    )
    env = os.environ.copy()
    start = time.monotonic()
    log_path = OUTPUT / "stdout.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(bundle_root / "worker.py")],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=OUTPUT,
            env=env,
            start_new_session=True,
        )
        peak = 0
        stop_reason = None
        while process.poll() is None:
            rss = proc_group_rss_bytes(process.pid) + own_rss_bytes()
            peak = max(peak, rss)
            elapsed = time.monotonic() - start
            if rss > RSS_LIMIT_BYTES:
                stop_reason = "aggregate_rss_limit"
            elif elapsed > WORKER_SECONDS:
                stop_reason = "worker_runtime_limit"
            elif time.monotonic() - started > JOB_SECONDS:
                stop_reason = "whole_job_runtime_limit"
            if stop_reason:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(POLL_SECONDS)
        if process.poll() is None:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                stop_reason = stop_reason or "process_tree_did_not_exit"
        return_code = process.poll()

    result_path = OUTPUT / "worker_result.json"
    worker_result = json.loads(result_path.read_text()) if result_path.exists() else None
    passed = bool(worker_result and worker_result.get("numerical_pass") is True)
    status = (
        "completed"
        if stop_reason is None and return_code == 0 and passed
        else (stop_reason or "worker_failed")
    )
    write_json(
        OUTPUT / "runner_result.json",
        {
            "status": status,
            "worker_return_code": return_code,
            "worker_result": worker_result,
            "stop_reason": stop_reason,
            "elapsed_seconds": time.monotonic() - start,
            "worker_runtime_limit_seconds": WORKER_SECONDS,
            "whole_job_runtime_limit_seconds": JOB_SECONDS,
            "aggregate_peak_rss_bytes": peak,
            "aggregate_rss_limit_bytes": RSS_LIMIT_BYTES,
            "host_memory_bytes": memory_bytes,
            "host_memory_reserve_fraction": 0.2,
            "observed_archive_sha256": observed_archive_hash,
            "manifest_sha256": actual_manifest_hash,
            "classification": "E465 CPU FP32 AdamW optimizer parity only; no model, data, fit, TPU, GPU, or score",
        },
    )
    return 0 if status == "completed" else 40


if __name__ == "__main__":
    raise SystemExit(main())
