from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working/e465_adamw_fp32_cpu_v1")
SOURCE_ROOT = Path(os.environ["E465_SOURCE_ROOT"]).resolve()
SCRIPT = SOURCE_ROOT / "scripts" / "check_e465_adamw_fp32_parity.py"
EXPECTED = {
    "jax": "0.11.1",
    "jaxlib": "0.11.1",
    "numpy": "2.1.3",
    "torch": "2.11.0+cpu",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    started = time.monotonic()
    versions = {"python": sys.version, "platform": platform.platform()}
    missing = []
    for package, exact_version in EXPECTED.items():
        if importlib.util.find_spec(package) is None:
            missing.append(package)
            versions[package] = None
            continue
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            observed = None
        versions[package] = observed
        if observed != exact_version:
            missing.append(f"{package}=={exact_version} (observed {observed!r})")
    python_ok = sys.version_info[:2] == (3, 13)
    runtime = {
        "versions": versions,
        "exact_package_versions": EXPECTED,
        "python_313": python_ok,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "source_root": str(SOURCE_ROOT),
        "checker": "scripts/check_e465_adamw_fp32_parity.py",
        "jax_backend": None,
        "jax_devices": None,
        "torch_cuda_available": None,
    }
    if missing or not python_ok:
        write_json(OUT / "runtime.json", runtime)
        write_json(
            OUT / "worker_result.json",
            {
                "status": "failed_runtime_version_contract",
                "version_mismatches": missing,
                "python_313": python_ok,
                "checker_run": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return 20
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        write_json(OUT / "runtime.json", runtime)
        write_json(
            OUT / "worker_result.json",
            {
                "status": "failed_cuda_visibility_contract",
                "checker_run": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return 21

    import jax
    import torch

    runtime["jax_backend"] = jax.default_backend()
    runtime["jax_devices"] = [
        {
            "platform": device.platform,
            "kind": getattr(device, "device_kind", str(device)),
            "id": int(device.id),
        }
        for device in jax.devices()
    ]
    runtime["torch_cuda_available"] = torch.cuda.is_available()
    write_json(OUT / "runtime.json", runtime)
    if (
        runtime["jax_backend"] != "cpu"
        or not runtime["jax_devices"]
        or any(device["platform"] != "cpu" for device in runtime["jax_devices"])
        or runtime["torch_cuda_available"]
    ):
        write_json(
            OUT / "worker_result.json",
            {
                "status": "failed_accelerator_backend_contract",
                "checker_run": False,
                "jax_backend": runtime["jax_backend"],
                "jax_devices": runtime["jax_devices"],
                "torch_cuda_available": runtime["torch_cuda_available"],
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return 22

    os.environ["PYTHONPATH"] = str(SOURCE_ROOT / "src")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    result_path = OUT / "adamw_fp32_parity.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(result_path)],
        cwd=OUT,
        env=os.environ.copy(),
        check=False,
        timeout=240,
        capture_output=True,
        text=True,
    )
    (OUT / "checker_stdout.log").write_text(completed.stdout)
    (OUT / "checker_stderr.log").write_text(completed.stderr)
    checker_result = json.loads(result_path.read_text()) if result_path.is_file() else None
    passed = bool(checker_result and checker_result.get("status") == "passed")
    write_json(
        OUT / "worker_result.json",
        {
            "status": "fp32_adamw_parity_pass" if completed.returncode == 0 and passed else "checker_failed",
            "checker_exit_code": completed.returncode,
            "checker_result": checker_result,
            "checker_run": True,
            "elapsed_seconds": time.monotonic() - started,
            "classification": "CPU-only FP32 optimizer-step parity; no model, dataset, TPU, GPU, fit, or score",
        },
    )
    return 0 if completed.returncode == 0 and passed else 23


if __name__ == "__main__":
    raise SystemExit(main())
