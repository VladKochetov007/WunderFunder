from __future__ import annotations

import hashlib
import importlib
import json
import os
import pathlib
import sys
import time
from collections.abc import Callable

import numpy as np
import pyarrow.parquet as pq
import torch

B = pathlib.Path("/home/grinchenkoeg/computer-vision-homework-20261018")
ROOT = B / "support-source/eda19-r12-v1"
CANONICAL_ROOT = B / "source-a266b036/src"
MANIFEST_PATH = ROOT / "idea_screens_r12/PREPARATION_MANIFEST.json"
MANIFEST_SHA256 = "__MANIFEST_SHA__"
CANONICAL_PINS = __CANONICAL_PIN_REPR__  # noqa: F821
RUNTIME_PY = __MODULES_REPR__  # noqa: F821
SAMPLE = B / "research_tmp/nf19-sample-r3-v2"
OUT = B / "research_tmp/nf19-r12-v1"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
ALLOWED_BUNDLE_ROOTS = {
    "idea_screens_r1",
    "idea_screens_r2",
    "idea_screens_r3",
    "idea_screens_r7",
    "idea_screens_r12",
}

sys.dont_write_bytecode = True


def sha256_file(path: pathlib.Path) -> str:
    """Hash one pinned source or manifest file without importing its package."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_canonical_path(pin_name: str) -> pathlib.Path:
    """Resolve a canonical semantic pin under the frozen WNN source snapshot."""
    if not pin_name.startswith("canonical:wnn/"):
        raise RuntimeError(f"unexpected canonical pin namespace: {pin_name}")
    relative = pathlib.PurePosixPath(pin_name.removeprefix("canonical:"))
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.parts[0] != "wnn"
    ):
        raise RuntimeError(f"unsafe canonical pin path: {pin_name}")
    return CANONICAL_ROOT / relative


def runtime_path(relative: str) -> pathlib.Path:
    """Resolve a runtime module to either its sealed staged or canonical source file."""
    if relative.startswith("canonical:"):
        return safe_canonical_path(relative)
    path = pathlib.PurePosixPath(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] not in ALLOWED_BUNDLE_ROOTS
    ):
        raise RuntimeError(f"runtime path escapes the sealed bundle roots: {relative}")
    return ROOT / path


manifest_bytes = MANIFEST_PATH.read_bytes()
if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
    raise RuntimeError("R12 import smoke saw a different preparation manifest")
manifest = json.loads(manifest_bytes)
if not isinstance(manifest, dict) or manifest.get("schema") != "nf19.r12_preparation.v1":
    raise RuntimeError("unexpected R12 preparation manifest schema")
manifest_files = manifest.get("files")
if not isinstance(manifest_files, dict):
    raise RuntimeError("R12 manifest files must be an object")
manifest_canonical = {
    name: digest for name, digest in manifest_files.items() if name.startswith("canonical:")
}
if manifest_canonical != CANONICAL_PINS:
    raise RuntimeError("R12 canonical source pins differ from the frozen assembly")
if manifest.get("runtime_py") != RUNTIME_PY:
    raise RuntimeError("R12 runtime closure differs from the frozen assembly")

canonical_verified: dict[str, str] = {}
for pin_name, expected_sha256 in CANONICAL_PINS.items():
    path = safe_canonical_path(pin_name)
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"R12 canonical source pin differs: {pin_name}")
    canonical_verified[pin_name] = expected_sha256

# Verify each runtime source path and pin before any R12/R1/R2/R3/R7/WNN import.
for module_name, relative in RUNTIME_PY.items():
    if not isinstance(module_name, str) or not isinstance(relative, str):
        raise RuntimeError("R12 runtime_py entries must be string pairs")
    expected_sha256 = manifest_files.get(relative)
    if not isinstance(expected_sha256, str):
        raise RuntimeError(f"runtime source is absent from the manifest: {relative}")
    path = runtime_path(relative)
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"R12 runtime source bytes differ: {relative}")

calls: list[str] = []


def deny_reader(name: str) -> Callable[..., object]:
    """Build an import-time reader sentinel for metadata and checkpoint loaders."""

    def deny(*args: object, **kwargs: object) -> object:
        calls.append(name)
        raise AssertionError(f"R12 cold import attempted a forbidden read through {name}")

    return deny


if torch.cuda.is_initialized():
    raise RuntimeError("R12 cold import started after CUDA initialization")
np.load = deny_reader("numpy.load")
np.memmap = deny_reader("numpy.memmap")
pq.ParquetFile = deny_reader("pyarrow.parquet.ParquetFile")
torch.load = deny_reader("torch.load")

started = time.monotonic()
imported: dict[str, str] = {}
for module_name, relative in RUNTIME_PY.items():
    module = importlib.import_module(module_name)
    expected_path = runtime_path(relative).resolve()
    actual_path = pathlib.Path(str(module.__file__)).resolve()
    if actual_path != expected_path:
        raise RuntimeError(f"R12 import resolved outside its pinned source: {module_name}")
    imported[module_name] = str(actual_path)
if calls:
    raise RuntimeError(f"R12 cold import attempted forbidden reads: {calls}")
if torch.cuda.is_initialized():
    raise RuntimeError("R12 cold import initialized CUDA")

smoke = {
    "schema": "nf19.r12_import_smoke.v1",
    "status": "PASS",
    "observed_utc": __import__("datetime")
    .datetime.now(__import__("datetime").timezone.utc)
    .isoformat(),
    "preparation_manifest_sha256": MANIFEST_SHA256,
    "runtime_import_paths": imported,
    "runtime_source_sha256": {
        relative: manifest_files[relative] for relative in RUNTIME_PY.values()
    },
    "canonical_source_sha256": canonical_verified,
    "reader_sentinel_calls": calls,
    "array_reads": 0,
    "parquet_reads": 0,
    "checkpoint_reads": 0,
    "cuda_initialized": False,
    "numpy_version": np.__version__,
    "torch_version": str(torch.__version__),
    "elapsed_s": time.monotonic() - started,
}
with (ROOT / "ROOT_IMPORT_SMOKE.json").open("xb") as stream:
    stream.write(json.dumps(smoke, indent=2, sort_keys=True).encode("utf-8"))
    stream.write(b"\n")

arguments = [
    "--sample",
    str(SAMPLE),
    "--output",
    str(OUT),
    "--expected-sample-sha256",
    SAMPLE_SHA256,
    "--expected-preparation-manifest-sha256",
    MANIFEST_SHA256,
]
os.execv(
    sys.executable,
    [sys.executable, "-B", "-m", "idea_screens_r12.run_probe", *arguments],
)
