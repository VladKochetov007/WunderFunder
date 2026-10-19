import importlib
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/support-source/eda19-r10-v1"
)
MANIFEST_PATH = ROOT / "idea_screens_r10/PREPARATION_MANIFEST.json"
MANIFEST_SHA256 = "__MANIFEST_SHA__"
RUNTIME_PY = __MODULES_REPR__
EXPECTED_RUNTIME_PY = {
    "idea_screens_r10": "idea_screens_r10/__init__.py",
    "idea_screens_r10.geometry": "idea_screens_r10/geometry.py",
    "idea_screens_r10.panel_io": "idea_screens_r10/panel_io.py",
    "idea_screens_r10.run_probe": "idea_screens_r10/run_probe.py",
}
SAMPLE = "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-sample-r3-v2"
OUT = "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-r10-v1"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"

sys.dont_write_bytecode = True
if RUNTIME_PY != EXPECTED_RUNTIME_PY:
    raise RuntimeError("assembled R10 runtime module map differs from the frozen import closure")
manifest_bytes = MANIFEST_PATH.read_bytes()
if __import__("hashlib").sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
    raise RuntimeError("R10 import smoke saw a different preparation manifest")
manifest = json.loads(manifest_bytes)
if manifest.get("runtime_py") != EXPECTED_RUNTIME_PY:
    raise RuntimeError("R10 manifest runtime closure differs from its frozen map")

old_load = np.load
old_memmap = np.memmap
calls = []


def deny_array_read(*args, **kwargs):
    calls.append({"args": len(args), "kwargs": sorted(kwargs)})
    raise AssertionError("R10 import smoke attempted an array read")


np.load = deny_array_read
np.memmap = deny_array_read
started = time.monotonic()
before = set(sys.modules)
forbidden = ("wnn", "pyarrow", "torch")
preloaded_forbidden = sorted(
    name
    for name in before
    if any(name == root or name.startswith(root + ".") for root in forbidden)
)
if preloaded_forbidden:
    raise RuntimeError("R10 import smoke started with a forbidden runtime already loaded")
imported = {}
for module_name, relative_path in EXPECTED_RUNTIME_PY.items():
    module = importlib.import_module(module_name)
    actual_path = pathlib.Path(str(module.__file__)).resolve()
    expected_path = (ROOT / relative_path).resolve()
    if actual_path != expected_path:
        raise RuntimeError("R10 import resolved outside the staged source: " + module_name)
    imported[module_name] = str(actual_path)
new_forbidden = sorted(
    name
    for name in set(sys.modules) - before
    if any(name == root or name.startswith(root + ".") for root in forbidden)
)
if calls or new_forbidden:
    raise RuntimeError("R10 import smoke read data or imported a forbidden runtime")
receipt = {
    "schema": "nf19.r10_import_smoke.v1",
    "status": "PASS",
    "observed_utc": __import__("datetime")
    .datetime.now(__import__("datetime").timezone.utc)
    .isoformat(),
    "preparation_manifest_sha256": MANIFEST_SHA256,
    "runtime_import_paths": imported,
    "array_reads": 0,
    "array_read_sentinel_calls": calls,
    "preloaded_forbidden_imports": preloaded_forbidden,
    "new_forbidden_imports": new_forbidden,
    "numpy_version": np.__version__,
    "elapsed_s": time.monotonic() - started,
}
with (ROOT / "ROOT_IMPORT_SMOKE.json").open("xb") as stream:
    stream.write(json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8"))
    stream.write(b"\n")

args = [
    "--sample",
    SAMPLE,
    "--output",
    OUT,
    "--expected-sample-sha256",
    SAMPLE_SHA256,
    "--expected-preparation-manifest-sha256",
    MANIFEST_SHA256,
]
os.execv(sys.executable, [sys.executable, "-B", "-m", "idea_screens_r10.run_probe", *args])
