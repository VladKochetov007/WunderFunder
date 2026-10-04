# E461 private CPU-only bridge/objective correctness check.
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

DATASET_SLUG = "wnn-jax-bridge-e461-v2-20261004"
ARCHIVE_SHA256 = "81aed6b16ba2a0928074bcf8eaa0c64ee30f24d201990e33793774094c6c13ee"
MANIFEST_SHA256 = "83042baa47ff412727cb1a7ba9a22ea92f2ae9f99e78302c79e7862a949988a1"
MAX_RUNTIME_SECONDS = 840
RSS_LIMIT_BYTES = 4 * 1024**3
POLL_SECONDS = 0.25
OUTPUT = Path("/kaggle/working/e461_bridge_objective_v2")

WORKER_SOURCE = r'''from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path("/kaggle/working/e461_bridge_objective_v2")
ROOT = Path(os.environ["WNN_PARITY_SOURCE_ROOT"])
REQUIRED = ("numpy", "pytest", "torch", "jax", "jaxlib", "pyarrow")
started = time.monotonic()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    versions = {"python": sys.version, "platform": platform.platform()}
    for package in REQUIRED:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    memory_bytes = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    runtime = {
        "versions": versions,
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": memory_bytes,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "env": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "XLA_FLAGS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"
        )},
        "missing_dependencies": missing,
        "jax_backend": None,
        "jax_devices": None,
        "torch_cuda_available": None,
    }
    if missing:
        write_json(OUT / "runtime.json", runtime)
        write_json(OUT / "worker_result.json", {
            "status": "failed_preflight_missing_dependencies", "missing_dependencies": missing,
            "tests_run": False, "numerical_pass": False,
            "elapsed_seconds": time.monotonic() - started,
        })
        return 20

    import jax
    import torch
    runtime["jax_backend"] = jax.default_backend()
    runtime["jax_devices"] = [{"platform": device.platform, "kind": getattr(device, "device_kind", str(device)),
                              "process_index": device.process_index}
                             for device in jax.devices()]
    runtime["torch_cuda_available"] = bool(torch.cuda.is_available())
    write_json(OUT / "runtime.json", runtime)
    if runtime["jax_backend"] != "cpu" or any(d["platform"] != "cpu" for d in runtime["jax_devices"]):
        write_json(OUT / "worker_result.json", {
            "status": "failed_preflight_non_cpu_device", "tests_run": False,
            "numerical_pass": False, "elapsed_seconds": time.monotonic() - started,
        })
        return 21
    if runtime["torch_cuda_available"]:
        write_json(OUT / "worker_result.json", {
            "status": "failed_preflight_cuda_visible", "tests_run": False,
            "numerical_pass": False, "elapsed_seconds": time.monotonic() - started,
        })
        return 22

    sys.path.insert(0, str(ROOT / "src"))
    import pytest

    xml_path = OUT / "pytest.xml"
    test_paths = [str(ROOT / "tests/test_bridge_objective.py")]
    rc = int(pytest.main(["-q", "-p", "no:cacheprovider", f"--junitxml={xml_path}", *test_paths]))
    tests = {"exit_code": rc, "junit_xml": xml_path.name, "tests": None, "failures": None, "errors": None, "skips": None}
    if xml_path.exists():
        tree = ET.parse(xml_path)
        cases = list(tree.iter("testcase"))
        tests.update({
            "tests": len(cases),
            "failures": sum(1 for x in tree.iter("failure")),
            "errors": sum(1 for x in tree.iter("error")),
            "skips": sum(1 for x in tree.iter("skipped")),
        })
    skipped = tests["skips"] or 0
    passed = rc == 0 and (tests["tests"] or 0) > 0 and not tests["failures"] and not tests["errors"] and skipped == 0
    write_json(OUT / "worker_result.json", {
        "status": "numerical_parity_pass" if passed else "failed_tests_or_skip",
        "numerical_pass": passed, "tests_run": True, "test_summary": tests,
        "elapsed_seconds": time.monotonic() - started,
        "classification": "CPU bridge/objective correctness only; no TPU or scientific score claim",
    })
    return 0 if passed else 23


if __name__ == "__main__":
    raise SystemExit(main())
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
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
            process_group = int(fields[2])
            if process_group == group_id:
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


def safe_extract(archive: Path, destination: Path, files: dict) -> None:
    allowed = set(files)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if set(names) != allowed or len(names) != len(allowed):
            raise ValueError("Archive members differ from sealed manifest")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ValueError("Unsafe archive path")
            data = bundle.read(name)
            item = files[name]
            if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Source file hash mismatch: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def verify_expanded(source_root: Path, files: dict) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError("Kaggle-expanded source_bundle directory is missing")
    for name, item in files.items():
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts or "\\" in name:
            raise ValueError("Unsafe manifest source path")
        path = source_root / name
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"Missing or wrong-sized source file: {name}")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Source file hash mismatch: {name}")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifests = list(Path("/kaggle/input").glob(f"**/{DATASET_SLUG}/source-manifest.json"))
    if len(manifests) != 1:
        write_json(OUTPUT / "runner_result.json", {
            "status": "failed_input_resolution", "manifest_matches": len(manifests),
        })
        return 30
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text())
    (OUTPUT / "source-manifest.json").write_text(manifest_path.read_text())
    if manifest.get("private") is not True or manifest.get("gpu") is not False or manifest.get("tpu") is not False:
        write_json(OUTPUT / "runner_result.json", {"status": "failed_bundle_privacy_or_accelerator_contract"})
        return 31
    actual_manifest_sha = sha256(manifest_path)
    if actual_manifest_sha != MANIFEST_SHA256:
        write_json(OUTPUT / "runner_result.json", {
            "status": "failed_manifest_hash", "actual_manifest_sha256": actual_manifest_sha,
        })
        return 32

    source_root = OUTPUT / "source"
    declared_archive_sha = manifest.get("archive_sha256")
    observed_archive_sha = None
    archive_candidates = list(manifest_path.parent.glob("source_bundle.zip"))
    if archive_candidates:
        archive = archive_candidates[0]
        observed_archive_sha = sha256(archive)
        if observed_archive_sha != ARCHIVE_SHA256 or observed_archive_sha != declared_archive_sha:
            write_json(OUTPUT / "runner_result.json", {
                "status": "failed_archive_hash", "observed_archive_sha256": observed_archive_sha,
                "manifest_declared_archive_sha256": declared_archive_sha,
            })
            return 32
        safe_extract(archive, source_root, manifest["files"])
        materialization_mode = "zip_verified_and_extracted"
    else:
        source_root = manifest_path.parent / "source_bundle"
        try:
            verify_expanded(source_root, manifest["files"])
        except (OSError, ValueError) as error:
            write_json(OUTPUT / "runner_result.json", {
                "status": "failed_expanded_source_verification", "detail": str(error),
                "source_root": str(source_root),
            })
            return 33
        materialization_mode = "kaggle_autoexpanded_zip_verified_by_member_hashes"
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if allowed:
            try:
                os.sched_setaffinity(0, {allowed[0]})
            except OSError:
                pass
    env = os.environ.copy()
    env["WNN_PARITY_SOURCE_ROOT"] = str(source_root)
    worker = OUTPUT / "worker.py"
    worker.write_text(WORKER_SOURCE)
    start = time.monotonic()
    log_path = OUTPUT / "stdout.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([sys.executable, str(worker)], stdout=log, stderr=subprocess.STDOUT,
                                   cwd=OUTPUT, env=env, start_new_session=True)
        peak = 0
        stop_reason = None
        while process.poll() is None:
            rss = proc_group_rss_bytes(process.pid) + own_rss_bytes()
            peak = max(peak, rss)
            elapsed = time.monotonic() - start
            if rss > RSS_LIMIT_BYTES:
                stop_reason = "aggregate_rss_limit"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            if elapsed > MAX_RUNTIME_SECONDS:
                stop_reason = "internal_runtime_limit"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            time.sleep(POLL_SECONDS)
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop_reason = stop_reason or "process_tree_did_not_exit"
        return_code = process.poll()
    worker_result_path = OUTPUT / "worker_result.json"
    worker_result = json.loads(worker_result_path.read_text()) if worker_result_path.exists() else None
    elapsed = time.monotonic() - start
    status = "completed" if stop_reason is None and return_code == 0 else (stop_reason or "worker_failed")
    write_json(OUTPUT / "runner_result.json", {
        "status": status,
        "worker_return_code": return_code,
        "stop_reason": stop_reason,
        "worker_result": worker_result,
        "elapsed_seconds": elapsed,
        "peak_aggregate_rss_bytes": peak,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "runtime_limit_seconds": MAX_RUNTIME_SECONDS,
        "observed_archive_sha256": observed_archive_sha,
        "manifest_declared_archive_sha256": declared_archive_sha,
        "manifest_sha256": sha256(manifest_path),
        "materialization_mode": materialization_mode,
        "kernel_code_sha256": sha256(Path(__file__)),
        "classification": "E461 CPU-only bridge/objective correctness; not TPU validation or scientific score",
    })
    return 0 if status == "completed" else 40


if __name__ == "__main__":
    raise SystemExit(main())
