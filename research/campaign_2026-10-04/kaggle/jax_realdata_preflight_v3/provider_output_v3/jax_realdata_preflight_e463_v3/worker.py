
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
OUT = Path("/kaggle/working/jax_realdata_preflight_e463_v3")
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
    if runtime["jax_backend"] != "cpu" or len(runtime["jax_devices"]) != 1:
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
        "--backend", "cpu",
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
            "CPU result does not validate TPU execution or HBM",
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
