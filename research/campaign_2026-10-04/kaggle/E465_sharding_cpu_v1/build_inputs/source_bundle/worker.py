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

OUT = Path("/kaggle/working/e465_sharding_cpu")
ROOT = Path(os.environ["E465_TEST_SOURCE_ROOT"])
REQUIRED = ("numpy", "pytest", "jax", "jaxlib")
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
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            )
        },
        "missing_dependencies": missing,
        "jax_backend": None,
        "jax_devices": None,
    }
    if missing:
        write_json(OUT / "runtime.json", runtime)
        write_json(
            OUT / "worker_result.json",
            {
                "status": "failed_preflight_missing_dependencies",
                "missing_dependencies": missing,
                "tests_run": False,
                "numerical_pass": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return 20

    import jax
    import pytest

    devices = jax.devices()
    runtime["jax_backend"] = jax.default_backend()
    runtime["jax_devices"] = [
        {
            "platform": device.platform,
            "kind": getattr(device, "device_kind", str(device)),
            "id": int(device.id),
        }
        for device in devices
    ]
    write_json(OUT / "runtime.json", runtime)
    if (
        runtime["jax_backend"] != "cpu"
        or len(devices) != 2
        or any(device.platform != "cpu" for device in devices)
    ):
        write_json(
            OUT / "worker_result.json",
            {
                "status": "failed_preflight_wrong_device_inventory",
                "tests_run": False,
                "numerical_pass": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return 21

    sys.path.insert(0, str(ROOT / "src"))
    xml_path = OUT / "pytest.xml"
    tests = [
        str(ROOT / "tests/test_distributed_combo.py"),
        str(ROOT / "tests/test_distributed_state_boundary.py"),
    ]
    return_code = int(
        pytest.main(["-q", "-p", "no:cacheprovider", f"--junitxml={xml_path}", *tests])
    )
    summary = {
        "exit_code": return_code,
        "junit_xml": xml_path.name,
        "tests": None,
        "failures": None,
        "errors": None,
        "skips": None,
    }
    if xml_path.exists():
        tree = ET.parse(xml_path)
        cases = list(tree.iter("testcase"))
        summary.update(
            {
                "tests": len(cases),
                "failures": sum(1 for _ in tree.iter("failure")),
                "errors": sum(1 for _ in tree.iter("error")),
                "skips": sum(1 for _ in tree.iter("skipped")),
            }
        )
    passed = (
        return_code == 0
        and (summary["tests"] or 0) > 0
        and summary["failures"] == 0
        and summary["errors"] == 0
        and summary["skips"] == 0
    )
    write_json(
        OUT / "worker_result.json",
        {
            "status": "numerical_parity_pass" if passed else "failed_tests_or_skip",
            "numerical_pass": passed,
            "tests_run": True,
            "test_summary": summary,
            "elapsed_seconds": time.monotonic() - started,
            "classification": "E465 CPU sharding/objective/state correctness only; no TPU fit or score",
        },
    )
    return 0 if passed else 23


if __name__ == "__main__":
    raise SystemExit(main())
