from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path("/kaggle/working/e461_bridge_objective")
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
