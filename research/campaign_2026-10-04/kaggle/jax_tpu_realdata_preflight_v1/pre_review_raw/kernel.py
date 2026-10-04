# E-463 bounded private JAX TPU real-data one-update preflight; no fit or score claim.
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

DATASET_SLUG = "wnn-jax-realdata-e463-tpu-20261004-v1"
MANIFEST_SHA256 = "4bd761fec15ffd60e35404a201d1cce86b13ef1cbb4a352451885c318ea18217"
ARCHIVE_SHA256 = "979f8d99e0198d6e1ed227a51556394c77be052b9503d8845787daa77c2e359b"
RSS_LIMIT_BYTES = 12884901888
MAX_RUNTIME_SECONDS = 3600
OUTPUT = Path("/kaggle/working/jax_realdata_preflight_e463_tpu_v1")

WORKER_SOURCE = r"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import json
import math
import os
import runpy
import resource
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["WNN_PREFLIGHT_ROOT"])
OUT = Path("/kaggle/working/jax_realdata_preflight_e463_tpu_v1")
RUN_OUT = OUT / "run"
REQUIRED = ("numpy", "pyarrow", "jax", "jaxlib")
started = time.monotonic()
MILESTONE_PATH = OUT / "phase_milestones.jsonl"


def milestone(phase: str, **details: object) -> None:
    try:
        rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        rss = 0
    record = {"phase": phase, "elapsed_seconds": time.monotonic() - started, "wall_time_epoch_seconds": time.time(),
              "current_process_rss_bytes": rss, "ru_maxrss_platform_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              "ru_maxrss_unit": "KiB on Linux", **details}
    with MILESTONE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"SHA-256 verification deadline exceeded: {path.name}")
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    milestone("worker_entry_before_dependency_probe")
    versions = {"python": sys.version}
    for name in REQUIRED:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    runtime = {
        "versions": versions,
        "missing_dependencies": missing,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_backend": None,
        "jax_devices": None,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }
    if missing:
        write_json(OUT / "runtime.json", runtime)
        write_json(OUT / "worker_result.json", {
            "status": "failed_preflight_missing_dependencies",
            "missing_dependencies": missing,
            "preflight_ran": False,
            "elapsed_seconds": time.monotonic() - started,
        })
        return 20

    milestone("jax_import_start")
    import jax
    milestone("jax_import_complete")
    runtime["jax_backend"] = jax.default_backend()
    runtime["jax_devices"] = [
        {"platform": str(device.platform), "kind": str(getattr(device, "device_kind", "unknown"))}
        for device in jax.devices()
    ]
    write_json(OUT / "runtime.json", runtime)
    if runtime["jax_backend"] != "tpu" or len(runtime["jax_devices"]) != 8 or any("TPU v5 lite" not in d["kind"] for d in runtime["jax_devices"]):
        write_json(OUT / "worker_result.json", {
            "status": "failed_preflight_wrong_backend_or_device_count",
            "preflight_ran": False,
            "elapsed_seconds": time.monotonic() - started,
        })
        return 21
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", None):
        write_json(OUT / "worker_result.json", {
            "status": "failed_preflight_cuda_visible",
            "preflight_ran": False,
            "elapsed_seconds": time.monotonic() - started,
        })
        return 22

    sys.argv = [
        str(ROOT / "scripts/realdata_preflight.py"),
        "--root", str(ROOT),
        "--split-manifest", str(ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"),
        "--initial-checkpoint", str(ROOT / "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"),
        "--initial-npz", str(ROOT / "artifacts/runs/c20261003_combo_s3_f0/model_ep3.jax.npz"),
        "--output", str(RUN_OUT),
        "--backend", "tpu",
        "--chunk-rows", "1000",
        "--learning-rate", "0.001",
        "--weight-decay", "0.01",
    ]
    milestone("bridge_import_and_backend_validation_complete", jax_backend=runtime["jax_backend"], device_count=len(runtime["jax_devices"]))
    started_run = time.monotonic()
    os.chdir(ROOT)
    try:
        sys.path.insert(0, str(ROOT / "src"))
        runpy.run_path(str(ROOT / "scripts/realdata_preflight.py"), run_name="__main__")
        returncode = 0
    except SystemExit as error:
        returncode = int(error.code or 0)
    report_path = RUN_OUT / "metrics.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    validation_errors = []
    if report is None:
        validation_errors.append("metrics.json missing")
    else:
        arms = report.get("arms", {})
        if set(arms) != {"fp32", "bf16"}:
            validation_errors.append("expected exactly fp32 and bf16 result arms")
        for arm_name, arm in arms.items():
            if arm.get("adamw_steps") != 1:
                validation_errors.append(f"{arm_name}: expected exactly one AdamW update")
            if not isinstance(arm.get("train_loss"), (int, float)) or not math.isfinite(arm["train_loss"]):
                validation_errors.append(f"{arm_name}: train loss missing or non-finite")
            model_path = Path(arm.get("model_npz", ""))
            if not model_path.is_file() or sha256_file(model_path) != arm.get("model_npz_sha256"):
                validation_errors.append(f"{arm_name}: exported model missing or hash mismatch")

        def finite_tree(value: object) -> bool:
            if isinstance(value, dict):
                return all(finite_tree(child) for child in value.values())
            if isinstance(value, list):
                return all(finite_tree(child) for child in value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return math.isfinite(value)
            return True

        for arm_name, arm in arms.items():
            if not finite_tree(arm.get("diagnostic_four_stream_chunk_score")):
                validation_errors.append(f"{arm_name}: non-finite diagnostic score")
        if report.get("fit_group_batch_sha256") != "65918af8c55326923a96878d800e5c35ab3e891700af9139d74f4ca03db274b3":
            validation_errors.append("fit batch row-order hash differs from audited B4 batch")
        if report.get("fit_groups") != [["valid", 1], ["valid", 0], ["train", 529], ["train", 0]]:
            validation_errors.append("fit groups differ from audited seeded row order")
        if len({arm.get("evaluation_mask_sha256") for arm in arms.values()}) != 1:
            validation_errors.append("precision arms use different evaluation masks")
    result = {
        "status": "bounded_preflight_complete" if returncode == 0 and report else "failed_realdata_preflight",
        "preflight_ran": True,
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started_run,
        "metrics_sha256": None,
        "metrics_bytes": None,
        "limitations": [
            "one AdamW update per FP32/BF16 precision on four streams is a systems check, not a fitted model",
            "the four-stream chunk score is diagnostic only and is not predictive evidence",
            "one TPU v5 lite device performs the diagnostic update; all eight are inventoried and monitored",
        ],
    }
    if report_path.is_file():
        import hashlib
        data = report_path.read_bytes()
        result["metrics_sha256"] = hashlib.sha256(data).hexdigest()
        result["metrics_bytes"] = len(data)
        result["metrics_status"] = report.get("status")
        result["arm_names"] = sorted(report.get("arms", {}))
        result["validation_errors"] = validation_errors
        if validation_errors:
            result["status"] = "failed_output_validation"
    write_json(OUT / "worker_result.json", result)
    return 0 if result["status"] == "bounded_preflight_complete" else 23


if __name__ == "__main__":
    raise SystemExit(main())
"""


def sha256_file(path: Path, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"SHA-256 verification deadline exceeded: {path.name}")
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def safe_extract(archive: Path, destination: Path, files: dict) -> None:
    expected = set(files)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if set(names) != expected or len(names) != len(expected):
            raise ValueError("Archive members differ from sealed source manifest")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ValueError("Unsafe archive path")
            data = bundle.read(name)
            item = files[name]
            if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Source member hash mismatch: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def verify_expanded(source_root: Path, files: dict) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError("Expanded source_bundle directory is missing")
    for name, item in files.items():
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts or "\\" in name:
            raise ValueError("Unsafe source manifest path")
        path = source_root / name
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Expanded source member mismatch: {name}")


def dataset_file(dataset: str, filename: str) -> Path:
    owner, slug = dataset.split("/", 1)
    candidates = [
        Path("/kaggle/input/datasets") / owner / slug / filename,
        Path("/kaggle/input") / slug / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = [path for path in Path("/kaggle/input").glob(f"**/{Path(filename).name}") if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one mounted file for {dataset}:{filename}; found {len(matches)}")
    return matches[0]


def proc_group_rss_bytes(group_id: int) -> int:
    total = 0
    page_size = os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].strip().split()
            if int(fields[2]) == group_id:
                total += int((entry / "statm").read_text().split()[1]) * page_size
        except (OSError, ValueError, IndexError):
            continue
    return total


def own_rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def terminate_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass



LAUNCHER_RESERVE_BYTES = 2 * 1024**3

def mem_total_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None

def cgroup_memory_limit() -> dict:
    candidates = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, rel = line.split(":", 2)
            rel_path = PurePosixPath(rel.lstrip("/"))
            if hierarchy == "0" and not controllers:
                candidates.extend([Path("/sys/fs/cgroup") / rel_path / "memory.max", Path("/sys/fs/cgroup/memory.max")])
            if "memory" in controllers.split(","):
                candidates.extend([Path("/sys/fs/cgroup/memory") / rel_path / "memory.limit_in_bytes", Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")])
    except (OSError, ValueError):
        pass
    candidates.extend([Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            raw = candidate.read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return {"path": str(candidate), "raw": raw, "limit_bytes": None, "unlimited": True}
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0:
            continue
        unlimited = value >= 1 << 60
        return {"path": str(candidate), "raw": raw, "limit_bytes": None if unlimited else value, "unlimited": unlimited}
    return {"path": None, "raw": None, "limit_bytes": None, "unlimited": None}

def validate_tpu_host(output: Path) -> tuple[bool, dict]:
    host_bytes = mem_total_bytes()
    cgroup = cgroup_memory_limit()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    parent_rss = own_rss_bytes()
    physical_ceiling = int(host_bytes * 0.8) if host_bytes else 0
    limits = [physical_ceiling] if physical_ceiling else []
    if cgroup["limit_bytes"] is not None:
        limits.append(cgroup["limit_bytes"])
    effective = min(limits) if limits else 0
    selected = affinity[:8]
    required = RSS_LIMIT_BYTES + LAUNCHER_RESERVE_BYTES
    record = {"host_mem_total_bytes": host_bytes, "host_reserve_fraction": 0.20,
              "host_memory_ceiling_after_reserve_bytes": physical_ceiling,
              "cgroup_memory": cgroup, "effective_memory_ceiling_bytes": effective,
              "aggregate_rss_stop_bytes": RSS_LIMIT_BYTES,
              "launcher_reserve_bytes": LAUNCHER_RESERVE_BYTES,
              "required_memory_budget_bytes": required,
              "launcher_current_rss_bytes": parent_rss,
              "available_cpu_affinity": affinity, "selected_cpu_affinity": selected,
              "selected_cpu_count": len(selected), "backend_requested_before_jax_import": "tpu",
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "guard_passed": bool(host_bytes and cgroup["path"] and len(selected) == 8
                  and effective >= required and parent_rss < LAUNCHER_RESERVE_BYTES
                  and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
                  and os.environ.get("JAX_PLATFORMS") == "tpu")}
    write_json(output / "host_resource_guard.json", record)
    return record["guard_passed"], record

def main() -> int:
    overall_started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "tpu",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) >= 8:
            os.sched_setaffinity(0, set(allowed[:8]))
    guard_ok, guard = validate_tpu_host(OUTPUT)
    if not guard_ok:
        write_json(OUTPUT / "runner_result.json", {"status": "failed_host_resource_guard", "host_resource_guard": guard,
                                                     "overall_runtime_limit_seconds": MAX_RUNTIME_SECONDS})
        return 37
    manifests = list(Path("/kaggle/input").glob(f"**/{DATASET_SLUG}/source-manifest.json"))
    if len(manifests) != 1:
        write_json(OUTPUT / "runner_result.json", {"status": "failed_input_manifest_resolution", "matches": len(manifests)})
        return 30
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text())
    shutil.copyfile(manifest_path, OUTPUT / "source-manifest.json")
    if manifest.get("private") is not True or manifest.get("internet") is not False or manifest.get("gpu") is not False or manifest.get("tpu") is not True:
        write_json(OUTPUT / "runner_result.json", {"status": "failed_privacy_or_resource_contract"})
        return 31
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != MANIFEST_SHA256:
        write_json(OUTPUT / "runner_result.json", {"status": "failed_manifest_hash", "actual_manifest_sha256": actual_manifest_sha})
        return 32

    source_root = OUTPUT / "source"
    archive_candidates = list(manifest_path.parent.glob("source_bundle.zip"))
    archive_observed_sha = None
    if archive_candidates:
        archive = archive_candidates[0]
        archive_observed_sha = sha256_file(archive)
        if archive_observed_sha != ARCHIVE_SHA256 or archive_observed_sha != manifest.get("archive_sha256"):
            write_json(OUTPUT / "runner_result.json", {"status": "failed_archive_hash", "observed_archive_sha256": archive_observed_sha})
            return 33
        safe_extract(archive, source_root, manifest["files"])
        materialization = "zip_verified_and_extracted"
    else:
        source_root = manifest_path.parent / "source_bundle"
        try:
            verify_expanded(source_root, manifest["files"])
        except (OSError, ValueError) as error:
            write_json(OUTPUT / "runner_result.json", {"status": "failed_expanded_source", "detail": str(error)})
            return 34
        materialization = "kaggle_autoexpanded_member_hash_verified"

    root = OUTPUT / "experiment"
    root.mkdir()
    for name, item in manifest["files"].items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, target)
    resolved_inputs = []
    try:
        for target_name, item in manifest["external_inputs"].items():
            if target_name.startswith("/") or ".." in PurePosixPath(target_name).parts or "\\" in target_name:
                raise ValueError(f"Unsafe external target path: {target_name}")
            source = dataset_file(item["dataset"], item["filename"])
            if source.stat().st_size != item["size_bytes"]:
                raise ValueError(f"External input size mismatch: {target_name}")
            observed_sha256 = sha256_file(source, deadline=overall_started + MAX_RUNTIME_SECONDS)
            if observed_sha256 != item["sha256"]:
                raise ValueError(f"External input SHA-256 mismatch: {target_name}")
            target = root / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Refusing to overwrite bundled source: {target_name}")
            target.symlink_to(source)
            resolved_inputs.append({
                "path": target_name,
                "dataset": item["dataset"],
                "mounted_source": str(source),
                "filename": item["filename"],
                "size_bytes": item["size_bytes"],
                "manifest_sha256": item["sha256"],
                "observed_sha256": observed_sha256,
            })
    except (OSError, ValueError, TimeoutError) as error:
        write_json(OUTPUT / "runner_result.json", {
            "status": "failed_external_input_resolution",
            "detail": str(error),
            "resolved_inputs": resolved_inputs,
            "materialization": materialization,
            "observed_archive_sha256": archive_observed_sha,
        })
        return 35

    (OUTPUT / "external_inputs.json").write_text(json.dumps(resolved_inputs, indent=2, sort_keys=True) + "\n")
    setup_elapsed = time.monotonic() - overall_started
    worker_runtime_limit = MAX_RUNTIME_SECONDS - setup_elapsed
    if worker_runtime_limit <= 15.0:
        write_json(OUTPUT / "runner_result.json", {
            "status": "failed_setup_runtime_limit",
            "setup_elapsed_seconds": setup_elapsed,
            "worker_runtime_limit_seconds": 0,
            "classification": "bounded one-update real-data JAX TPU systems preflight; no fitted model or scientific gain claim",
        })
        return 36
    os.environ.update({
        "WNN_PREFLIGHT_ROOT": str(root),
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "tpu",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    os.environ["WNN_PREFLIGHT_MILESTONE"] = str(OUTPUT / "phase_milestones.jsonl")
    os.environ["WNN_TPU_MEMORY_SAMPLES"] = str(OUTPUT / "tpu_memory_samples.jsonl")
    worker = OUTPUT / "worker.py"
    worker.write_text(WORKER_SOURCE)
    started = time.monotonic()
    stop_reason = None
    peak = 0
    with (OUTPUT / "worker_stdout.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(worker)],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=OUTPUT,
            env=os.environ.copy(),
            start_new_session=True,
        )
        while process.poll() is None:
            rss = proc_group_rss_bytes(process.pid) + own_rss_bytes()
            peak = max(peak, rss)
            elapsed = time.monotonic() - started
            if rss >= RSS_LIMIT_BYTES:
                stop_reason = "aggregate_rss_limit"
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            if elapsed >= max(0.0, worker_runtime_limit - 15.0):
                stop_reason = "worker_runtime_limit"
                terminate_group(process.pid)
                break
            time.sleep(0.5)
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            stop_reason = stop_reason or "process_tree_did_not_exit"
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return_code = process.wait(timeout=5)
    worker_result_path = OUTPUT / "worker_result.json"
    worker_result = json.loads(worker_result_path.read_text()) if worker_result_path.is_file() else None
    result = {
        "status": "completed" if stop_reason is None and return_code == 0 else (stop_reason or "worker_failed"),
        "worker_return_code": return_code,
        "stop_reason": stop_reason,
        "worker_result": worker_result,
        "elapsed_seconds": time.monotonic() - started,
        "aggregate_peak_rss_bytes": peak,
        "aggregate_rss_limit_bytes": RSS_LIMIT_BYTES,
        "setup_elapsed_seconds": setup_elapsed,
        "worker_runtime_limit_seconds": worker_runtime_limit,
        "overall_runtime_limit_seconds": MAX_RUNTIME_SECONDS,
        "observed_archive_sha256": archive_observed_sha,
        "manifest_declared_archive_sha256": manifest.get("archive_sha256"),
        "manifest_sha256": actual_manifest_sha,
        "materialization": materialization,
        "resolved_external_input_count": len(resolved_inputs),
        "classification": "bounded one-update real-data JAX TPU systems preflight; no fitted model or scientific gain claim",
    }
    write_json(OUTPUT / "runner_result.json", result)
    return 0 if result["status"] == "completed" else 40


if __name__ == "__main__":
    raise SystemExit(main())
