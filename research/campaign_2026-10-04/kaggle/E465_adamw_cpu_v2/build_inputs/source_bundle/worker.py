from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

OUT = Path("/kaggle/working/e465_adamw_fp32_cpu_v2")
EXPECTED = {
    "jax": "0.11.1",
    "jaxlib": "0.11.1",
    "numpy": "2.1.3",
    "torch": "2.11.0+cpu",
}
EXPECTED_SOURCE_HASHES = {
    "scripts/check_e465_adamw_fp32_parity_v2.py": "fbdddd791856250d965159dcce20a95c557aa023602aa10594a232a7b5d9a9af",
    "src/wnn_jax/onecycle.py": "f857fe857a361249c2742e3486cd32b846e0f47413a5c4db2ae1fb6c65509ded",
    "src/wnn_jax/rnn.py": "294eff11d906fe434dba2c4d0ed4741ad2191f756c4a35204474cb229ff939dc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in EXPECTED_SOURCE_HASHES}


def validate_source_hashes(observed: Mapping[str, str]) -> bool:
    return dict(observed) == EXPECTED_SOURCE_HASHES


def checker_passes(return_code: int | None, result: Mapping[str, Any] | None) -> bool:
    if return_code != 0 or result is None:
        return False
    return (
        result.get("status") == "passed"
        and result.get("updates") == 7_460
        and result.get("source_sha256")
        == {
            "rnn.py": EXPECTED_SOURCE_HASHES["src/wnn_jax/rnn.py"],
            "onecycle.py": EXPECTED_SOURCE_HASHES["src/wnn_jax/onecycle.py"],
        }
        and result.get("model_or_data_loaded") is False
    )


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    started = time.monotonic()
    source_root = Path(os.environ["E465_SOURCE_ROOT"]).resolve()
    script = source_root / "scripts" / "check_e465_adamw_fp32_parity_v2.py"
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
    python_ok = sys.version_info[:3] == (3, 13, 15)
    runtime = {
        "versions": versions,
        "exact_package_versions": EXPECTED,
        "python_exact_3_13_15": python_ok,
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
        "source_root": str(source_root),
        "checker": "scripts/check_e465_adamw_fp32_parity_v2.py",
        "source_sha256": {},
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
                "python_exact_3_13_15": python_ok,
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

    try:
        observed_source_hashes = source_hashes(source_root)
    except OSError as error:
        write_json(OUT / "runtime.json", runtime)
        write_json(
            OUT / "worker_result.json",
            {
                "status": "failed_source_read",
                "detail": str(error),
                "checker_run": False,
                "numerical_pass": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return 22
    runtime["source_sha256"] = observed_source_hashes
    if not validate_source_hashes(observed_source_hashes):
        write_json(OUT / "runtime.json", runtime)
        write_json(
            OUT / "worker_result.json",
            {
                "status": "failed_source_hash_contract",
                "expected_source_sha256": EXPECTED_SOURCE_HASHES,
                "observed_source_sha256": observed_source_hashes,
                "checker_run": False,
                "numerical_pass": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return 23

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

    os.environ["PYTHONPATH"] = str(source_root / "src")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    result_path = OUT / "adamw_fp32_parity.json"
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(result_path)],
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
    passed = checker_passes(completed.returncode, checker_result)
    write_json(
        OUT / "worker_result.json",
        {
            "status": "fp32_adamw_parity_pass" if passed else "checker_failed",
            "checker_exit_code": completed.returncode,
            "checker_result": checker_result,
            "checker_run": True,
            "numerical_pass": passed,
            "elapsed_seconds": time.monotonic() - started,
            "classification": "CPU-only FP32 optimizer-step parity; no model, dataset, TPU, GPU, fit, or score",
        },
    )
    return 0 if completed.returncode == 0 and passed else 23


if __name__ == "__main__":
    raise SystemExit(main())
