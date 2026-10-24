#!/usr/bin/env python3
"""Run one admitted, strict-FP32 CUDA profile on a portable age cache."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from contextlib import suppress
from pathlib import Path, PurePosixPath

# Admission and source manifests are intentionally decoded as heterogeneous JSON.
from typing import Any, BinaryIO

DISK_ROOT = Path("/kaggle/working/disk_scratch")
BUNDLE_ROOT = DISK_ROOT / "age_actor_bundle"
WORK_ROOT = DISK_ROOT / "age_actor_work"
BUNDLE_ARCHIVE_NAME = "age_actor_bundle.wnn"
PROFILE_ADMISSION_NAME = "PROFILE_F0_CUDA_R2_ADMISSION.json"
PROFILE_ADMISSION_SHA256 = "__PROFILE_ADMISSION_SHA256__"
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
MACHINE_RECIPE = "research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json"
MACHINE_RECIPE_SHA256 = "0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd"
SOURCE_MANIFEST_GPU_R2 = "source/SOURCE_SHA256_GPU_R2.json"
ACTOR_SOURCE_ROOT = "repo"
SOURCE_OVERLAY_ARCHIVE_NAME = "source_overlay_cuda_r2.wnn"
SOURCE_OVERLAY_MANIFEST_NAME = "overlay_manifest.json"
PROFILE_MAX_SECONDS = 600
FINALIZATION_RESERVE_SECONDS = 45
PROCESS_RSS_MAX_BYTES = 4_000_000_000
MAX_EXPANDED_BYTES = 1_600_000_000
MAX_ARCHIVE_BYTES = 1_600_000_000
MAX_METADATA_BYTES = 32_000_000
MAX_OVERLAY_ARCHIVE_BYTES = 1_000_000
MAX_OVERLAY_EXPANDED_BYTES = 500_000
CHUNK_BYTES = 8 * 1024**2
POLL_SECONDS = 1.0
EXPECTED_F0_CACHE_BYTES = 16_024_320_384
WORKING_OUTPUT_RESERVE_BYTES = 1_000_000_000
WORKING_SAVED_OUTPUT_LIMIT_BYTES = 20_000_000_000
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_PATH_RE = re.compile(
    r"(^|[/_.-])(token|secret|credentials?|password|kaggle\.json|\.env)([/_.-]|$)",
    re.IGNORECASE,
)


class ProfileError(RuntimeError):
    """The Kaggle profile bundle, admission, or resource envelope failed."""


def safe_relative_path(value: str) -> str:
    """Require a canonical archive path with no traversal or credential names."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or str(path) != value
        or FORBIDDEN_PATH_RE.search(value)
        or value.startswith(".git/")
    ):
        raise ProfileError("unsafe or credential-like bundle path")
    return str(path)


def read_object(path: Path, maximum_bytes: int = MAX_METADATA_BYTES) -> dict[str, Any]:
    """Read one bounded JSON object without touching array or Parquet payloads."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ProfileError(f"metadata input is absent, nonregular, or oversized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProfileError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path, deadline: float, maximum_bytes: int | None = None) -> str:
    """Hash one file in fixed memory while respecting the worker deadline."""
    if path.is_symlink() or not path.is_file():
        raise ProfileError(f"hash input is absent or nonregular: {path}")
    if maximum_bytes is not None and path.stat().st_size > maximum_bytes:
        raise ProfileError(f"hash input exceeds its byte bound: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            require_time(deadline, "file hashing")
            digest.update(block)
    return digest.hexdigest()


def require_time(deadline: float, phase: str) -> None:
    """Stop a phase before it consumes the admitted runtime or finalization reserve."""
    if time.monotonic() >= deadline:
        raise TimeoutError(f"Kaggle profile deadline reached during {phase}")


def find_unique_input(filename: str) -> Path:
    """Find one regular input file by exact basename under Kaggle's read-only mount."""
    matches: list[Path] = []
    for directory, names, files in os.walk("/kaggle/input", followlinks=False):
        names[:] = [name for name in names if not (Path(directory) / name).is_symlink()]
        if filename in files:
            path = Path(directory) / filename
            if path.is_file() and not path.is_symlink():
                matches.append(path)
    if len(matches) != 1:
        raise ProfileError(f"expected exactly one Kaggle input {filename}, found {len(matches)}")
    return matches[0]


def validate_admission(admission: dict[str, Any]) -> dict[str, Any]:
    """Require root's exact F0, one-GPU, profile-only authorization."""
    if (
        admission.get("schema") != "wnn.age-readout-gpu-profile-admission.v1"
        or admission.get("status") != "admitted"
        or admission.get("mode") != "profile"
        or admission.get("device") != "cuda"
        or admission.get("recipe_sha256") != MACHINE_RECIPE_SHA256
        or int(admission.get("fold", -1)) != 0
        or int(admission.get("fit_sequences", -1)) != 16
        or int(admission.get("evaluation_sequences", -1)) != 8
        or int(admission.get("new_scientific_fits", -1)) != 0
        or admission.get("quality_scoring") is not False
        or admission.get("six_cell_scientific_panel") != "unadmitted"
        or admission.get("training_subset_labels_read") is not True
        or admission.get("evaluation_targets_read") is not False
        or admission.get("gpu_enabled") is not True
        or admission.get("engineering_microfits") != 3
        or admission.get("engineering_epochs_per_microfit") != 3
        or admission.get("engineering_updates_per_microfit") != 120
        or admission.get("engineering_optimizer_updates") != 360
    ):
        raise ProfileError("profile admission does not match the frozen CUDA-only F0 sample")
    wall_seconds = int(admission.get("max_wall_seconds", 0))
    cpu_cores = int(admission.get("cpu_cores", 0))
    ram_bytes = int(admission.get("ram_bytes", 0))
    swap_bytes = int(admission.get("swap_bytes", -1))
    if not 1 <= wall_seconds <= PROFILE_MAX_SECONDS or cpu_cores != 1:
        raise ProfileError("profile admission exceeds the 600-second or one-core cap")
    if ram_bytes <= 0 or ram_bytes > PROCESS_RSS_MAX_BYTES or swap_bytes != 0:
        raise ProfileError("profile admission exceeds the RSS cap or allows swap")
    archive_sha = admission.get("bundle_archive_sha256")
    transport_sha = admission.get("transport_sha256")
    manifest_sha = admission.get("bundle_manifest_sha256")
    export_receipt_sha = admission.get("export_receipt_sha256")
    source_manifest_sha = admission.get("source_manifest_sha256")
    overlay_sha = admission.get("source_overlay_archive_sha256")
    overlay_manifest_sha = admission.get("source_overlay_manifest_sha256")
    overlay_bytes = admission.get("source_overlay_archive_bytes")
    driver_template_sha = admission.get("driver_template_sha256")
    archive_bytes = admission.get("bundle_archive_bytes")
    if (
        not isinstance(archive_sha, str)
        or SHA256_RE.fullmatch(archive_sha) is None
        or not isinstance(transport_sha, str)
        or SHA256_RE.fullmatch(transport_sha) is None
        or not isinstance(manifest_sha, str)
        or SHA256_RE.fullmatch(manifest_sha) is None
        or not isinstance(export_receipt_sha, str)
        or SHA256_RE.fullmatch(export_receipt_sha) is None
        or not isinstance(source_manifest_sha, str)
        or SHA256_RE.fullmatch(source_manifest_sha) is None
        or not isinstance(driver_template_sha, str)
        or SHA256_RE.fullmatch(driver_template_sha) is None
        or not isinstance(overlay_sha, str)
        or SHA256_RE.fullmatch(overlay_sha) is None
        or not isinstance(overlay_manifest_sha, str)
        or SHA256_RE.fullmatch(overlay_manifest_sha) is None
        or not isinstance(overlay_bytes, int)
        or overlay_bytes <= 0
        or overlay_bytes > MAX_OVERLAY_ARCHIVE_BYTES
        or not isinstance(archive_bytes, int)
        or archive_bytes <= 0
    ):
        raise ProfileError("profile admission omits a frozen archive/source identity")
    return {
        "wall_seconds": wall_seconds,
        "cpu_cores": cpu_cores,
        "ram_bytes": ram_bytes,
        "bundle_archive_sha256": archive_sha,
        "bundle_archive_bytes": archive_bytes,
        "bundle_manifest_sha256": manifest_sha,
        "export_receipt_sha256": export_receipt_sha,
        "source_manifest_sha256": source_manifest_sha,
        "source_overlay_archive_sha256": overlay_sha,
        "source_overlay_archive_bytes": overlay_bytes,
        "source_overlay_manifest_sha256": overlay_manifest_sha,
        "driver_template_sha256": driver_template_sha,
        "transport_sha256": transport_sha,
    }


def verify_export_outputs(
    bundle_root: Path,
    entries: dict[str, dict[str, Any]],
    admission: dict[str, Any],
    deadline: float,
) -> str:
    """Bind the extracted transport files to the completed local export receipt."""
    receipt_path = bundle_root / "EXPORT_RECEIPT_R1.json"
    receipt_sha = sha256_file(receipt_path, deadline, MAX_METADATA_BYTES)
    if receipt_sha != admission["export_receipt_sha256"]:
        raise ProfileError("export receipt SHA differs from root admission")
    receipt = read_object(receipt_path, maximum_bytes=MAX_METADATA_BYTES)
    if receipt.get("status") != "complete" or receipt.get("new_scientific_fits") != 0:
        raise ProfileError("suffix export receipt is not complete and fit-free")
    exported = receipt.get("outputs", {})
    if not isinstance(exported, dict) or not exported:
        raise ProfileError("suffix export receipt lacks a pinned output map")
    for relative, item in exported.items():
        path = safe_relative_path(relative)
        manifest_entry = entries.get(path)
        if (
            manifest_entry is None
            or manifest_entry["sha256"] != item.get("sha256")
            or manifest_entry["bytes"] != item.get("bytes")
        ):
            raise ProfileError(f"archive manifest differs from completed export output: {path}")
    transport = read_object(bundle_root / "transport.json", maximum_bytes=MAX_METADATA_BYTES)
    if transport.get("schema") != "wnn.age-readout-portable-transport.v1":
        raise ProfileError("transport schema differs from the portable age-readout contract")
    for fold in (0, 1):
        for key in ("fit_suffix5", "evaluation_predictions"):
            item = transport["folds"][str(fold)][key]
            path = safe_relative_path(item["path"])
            manifest_entry = entries.get(path)
            if (
                manifest_entry is None
                or manifest_entry["sha256"] != item["sha256"]
                or manifest_entry["bytes"] != item["bytes"]
            ):
                raise ProfileError(f"transport array differs from the archive manifest: {path}")
    return receipt_sha


def parse_manifest(payload: bytes) -> tuple[dict[str, dict[str, Any]], int]:
    """Validate the portable archive's exact file map and no-target payload scope."""
    if len(payload) > MAX_METADATA_BYTES:
        raise ProfileError("bundle manifest exceeds its metadata byte limit")
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or value.get("schema") != "wnn.age-readout-portable-profile-bundle.v1"
        or not isinstance(value.get("files"), list)
        or not isinstance(value.get("source_root"), str)
        or not value["source_root"]
    ):
        raise ProfileError("portable bundle manifest schema is unsupported")
    entries: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    array_paths: set[str] = set()
    for entry in value["files"]:
        if not isinstance(entry, dict) or set(entry) != {
            "source_path",
            "bundle_path",
            "sha256",
            "bytes",
        }:
            raise ProfileError("bundle manifest has an invalid file descriptor")
        path = safe_relative_path(entry["bundle_path"])
        source = entry["source_path"]
        digest = entry["sha256"]
        size = entry["bytes"]
        if (
            path in entries
            or not isinstance(source, str)
            or not source
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ProfileError("bundle manifest has duplicate or malformed file pins")
        if path.endswith((".npz", ".parquet")):
            raise ProfileError("portable profile bundle must exclude targets and raw Parquet")
        if path.endswith(".npy"):
            array_paths.add(path)
        entries[path] = entry
        total_bytes += size
        if total_bytes > MAX_EXPANDED_BYTES:
            raise ProfileError("portable profile bundle exceeds its expanded-size ceiling")
    expected_arrays = {
        f"fold{fold}/{name}.npy"
        for fold in (0, 1)
        for name in ("fit_suffix5", "evaluation_predictions")
    }
    if array_paths != expected_arrays:
        raise ProfileError("portable profile bundle has an unexpected array inventory")
    return entries, total_bytes


def process_resources(process: Any) -> tuple[int, int]:
    """Return RSS and swap summed over this driver and its live descendants."""
    import psutil

    root = psutil.Process(os.getpid())
    items = {item.pid: item for item in (root, process, *root.children(recursive=True))}
    total_rss = total_swap = 0
    for item in items.values():
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            status = Path(f"/proc/{item.pid}/status").read_text(encoding="utf-8")
            fields = {
                line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                for line in status.splitlines()
                if ":" in line
            }
            total_rss += int(fields.get("VmRSS", "0 kB").split()[0]) * 1024
            total_swap += int(fields.get("VmSwap", "0 kB").split()[0]) * 1024
    return total_rss, total_swap


def safe_extract(
    archive: Path,
    destination: Path,
    deadline: float,
    peak_rss: list[int],
    peak_swap: list[int],
) -> tuple[int, int, dict[str, dict[str, Any]]]:
    """Extract exactly the manifest-declared regular files into a fresh tree."""
    if destination.exists() or destination.is_symlink():
        raise ProfileError("refusing to overwrite the portable extraction directory")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise ProfileError("refusing to overwrite a prior partial extraction")
    with tarfile.open(archive, mode="r|") as source_tar:
        members = iter(source_tar)
        first = next(members, None)
        if (
            first is None
            or first.name != BUNDLE_MANIFEST_NAME
            or not first.isfile()
            or first.size > MAX_METADATA_BYTES
        ):
            raise ProfileError("portable archive must start with its small manifest")
        manifest_file = source_tar.extractfile(first)
        if manifest_file is None:
            raise ProfileError("portable bundle manifest has no content")
        with manifest_file:
            manifest_payload = manifest_file.read(MAX_METADATA_BYTES + 1)
        entries, expanded_bytes = parse_manifest(manifest_payload)
        disk = shutil.disk_usage(DISK_ROOT)
        required = expanded_bytes + EXPECTED_F0_CACHE_BYTES + WORKING_OUTPUT_RESERVE_BYTES
        if required > WORKING_SAVED_OUTPUT_LIMIT_BYTES or disk.free < required:
            raise ProfileError(
                f"working disk lacks headroom for extraction and one full F0 cache: {disk.free}"
            )
        partial.mkdir(parents=True, mode=0o700)
        allowed = set(entries) | {BUNDLE_MANIFEST_NAME}
        seen = {BUNDLE_MANIFEST_NAME}
        _copy_member(partial, first.name, io.BytesIO(manifest_payload), first.size, None)
        for member in members:
            require_time(deadline, "archive extraction")
            path = safe_relative_path(member.name)
            entry = entries.get(path)
            if (
                not member.isfile()
                or entry is None
                or path in seen
                or member.size != entry["bytes"]
            ):
                raise ProfileError("archive contains a link, duplicate, extra, or wrong-size file")
            stream = source_tar.extractfile(member)
            if stream is None:
                raise ProfileError("portable archive member has no content")
            target = partial / path
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, copied = hashlib.sha256(), 0
            with stream, target.open("xb", buffering=CHUNK_BYTES) as output:
                while True:
                    block = stream.read(CHUNK_BYTES)
                    if not block:
                        break
                    require_time(deadline, "archive extraction")
                    copied += len(block)
                    if copied > int(entry["bytes"]):
                        raise ProfileError("archive member exceeds its pinned byte count")
                    digest.update(block)
                    output.write(block)
            if copied != int(entry["bytes"]) or digest.hexdigest() != entry["sha256"]:
                raise ProfileError(f"portable member failed its content pin: {path}")
            seen.add(path)
            rss, swap = current_resources()
            peak_rss[0] = max(peak_rss[0], rss)
            peak_swap[0] = max(peak_swap[0], swap)
        if seen != allowed:
            raise ProfileError("portable archive omits one or more manifest-declared files")
    os.replace(partial, destination)
    return expanded_bytes, len(entries), entries


def apply_source_overlay(
    bundle_root: Path,
    archive: Path,
    admission: dict[str, Any],
    deadline: float,
) -> tuple[str, int]:
    """Apply the pinned R3 closure plus CUDA profile guards over the R2 bundle."""
    if archive.is_symlink() or not archive.is_file():
        raise ProfileError("R3 source overlay is absent or nonregular")
    if archive.stat().st_size != admission["source_overlay_archive_bytes"]:
        raise ProfileError("R3 source overlay byte count differs from admission")
    if archive.stat().st_size > MAX_OVERLAY_ARCHIVE_BYTES:
        raise ProfileError("R3 source overlay exceeds its one-megabyte input limit")
    with tarfile.open(archive, mode="r|") as source_tar:
        members = iter(source_tar)
        first = next(members, None)
        if (
            first is None
            or first.name != SOURCE_OVERLAY_MANIFEST_NAME
            or not first.isfile()
            or first.size > MAX_METADATA_BYTES
        ):
            raise ProfileError("R3 overlay must start with its bounded manifest")
        manifest_file = source_tar.extractfile(first)
        if manifest_file is None:
            raise ProfileError("R3 source overlay manifest has no content")
        with manifest_file:
            payload = manifest_file.read(MAX_METADATA_BYTES + 1)
        manifest_sha = hashlib.sha256(payload).hexdigest()
        if manifest_sha != admission["source_overlay_manifest_sha256"]:
            raise ProfileError("R3 source overlay manifest differs from admission")
        manifest = json.loads(payload)
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != "wnn.age-readout-source-overlay.v1"
            or manifest.get("source_manifest_sha256")
            != admission["source_manifest_sha256"]
            or manifest.get("base_bundle_archive_sha256")
            != admission["bundle_archive_sha256"]
            or not isinstance(manifest.get("files"), list)
            or len(manifest["files"]) != 45
        ):
            raise ProfileError("CUDA source overlay does not bind the admitted R2/GPU manifests")
        entries: dict[str, dict[str, Any]] = {}
        expanded_bytes = 0
        for entry in manifest["files"]:
            if not isinstance(entry, dict) or set(entry) != {
                "base_sha256",
                "bytes",
                "path",
                "sha256",
            }:
                raise ProfileError("R3 overlay manifest has an invalid file descriptor")
            path = safe_relative_path(entry["path"])
            base_sha = entry["base_sha256"]
            size = entry["bytes"]
            digest = entry["sha256"]
            if (
                path in entries
                or not path.startswith("repo/")
                or (base_sha is not None and (
                    not isinstance(base_sha, str) or SHA256_RE.fullmatch(base_sha) is None
                ))
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or not isinstance(digest, str)
                or SHA256_RE.fullmatch(digest) is None
            ):
                raise ProfileError("R3 overlay manifest has duplicate or malformed pins")
            expanded_bytes += size
            if expanded_bytes > MAX_OVERLAY_EXPANDED_BYTES:
                raise ProfileError("R3 source overlay exceeds its expanded-size ceiling")
            target = bundle_root / path
            if base_sha is None:
                if target.exists() or target.is_symlink():
                    raise ProfileError(f"R3 overlay expected an absent source path: {path}")
            elif (
                target.is_symlink()
                or not target.is_file()
                or sha256_file(target, deadline, MAX_METADATA_BYTES) != base_sha
            ):
                raise ProfileError(f"R3 replacement base SHA differs from R2: {path}")
            entries[path] = entry
        seen: set[str] = set()
        for member in members:
            require_time(deadline, "source overlay extraction")
            path = safe_relative_path(member.name)
            entry = entries.get(path)
            if (
                not member.isfile()
                or entry is None
                or path in seen
                or member.size != entry["bytes"]
            ):
                raise ProfileError("R3 overlay contains a link, duplicate, extra, or wrong-size file")
            stream = source_tar.extractfile(member)
            if stream is None:
                raise ProfileError("R3 overlay member has no content")
            target = bundle_root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".r3partial")
            if partial.exists() or partial.is_symlink():
                raise ProfileError("refusing to overwrite an R3 partial source file")
            digest, copied = hashlib.sha256(), 0
            with stream, partial.open("xb", buffering=CHUNK_BYTES) as output:
                while True:
                    block = stream.read(CHUNK_BYTES)
                    if not block:
                        break
                    require_time(deadline, "source overlay extraction")
                    copied += len(block)
                    if copied > int(entry["bytes"]):
                        raise ProfileError("R3 overlay member exceeds its pinned byte count")
                    digest.update(block)
                    output.write(block)
            if copied != int(entry["bytes"]) or digest.hexdigest() != entry["sha256"]:
                raise ProfileError(f"R3 source member failed its content pin: {path}")
            os.replace(partial, target)
            seen.add(path)
        if seen != set(entries):
            raise ProfileError("R3 source overlay omits one or more manifest-declared files")
    return manifest_sha, len(entries)


def _copy_member(root: Path, name: str, stream: BinaryIO, size: int, digest: Any) -> None:
    """Copy a bounded archive member, optionally checking its exact SHA256."""
    target = root / safe_relative_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    with target.open("xb", buffering=CHUNK_BYTES) as output:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            copied += len(block)
            if copied > size:
                raise ProfileError("archive member exceeds its declared size")
            output.write(block)
            if digest is not None:
                digest.update(block)
    if copied != size:
        raise ProfileError("archive member is shorter than its declared size")


def current_resources() -> tuple[int, int]:
    """Read this driver's RSS and swap during extraction without spawning a tool."""
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    fields = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in status.splitlines()
        if ":" in line
    }
    rss = int(fields.get("VmRSS", "0 kB").split()[0]) * 1024
    swap = int(fields.get("VmSwap", "0 kB").split()[0]) * 1024
    if rss >= PROCESS_RSS_MAX_BYTES:
        raise MemoryError("profile driver RSS reached its four-gigabyte ceiling")
    if swap != 0:
        raise MemoryError("profile driver used swap despite the zero-swap admission")
    return rss, swap


def set_disk_environment() -> dict[str, str]:
    """Create disk-backed temp/cache roots while preserving Kaggle's single GPU."""
    paths = {
        "TMPDIR": WORK_ROOT / "tmp",
        "TMP": WORK_ROOT / "tmp",
        "TEMP": WORK_ROOT / "tmp",
        "XDG_CACHE_HOME": WORK_ROOT / "cache/xdg",
        "TORCHINDUCTOR_CACHE_DIR": WORK_ROOT / "cache/torchinductor",
        "TRITON_CACHE_DIR": WORK_ROOT / "cache/triton",
        "NUMBA_CACHE_DIR": WORK_ROOT / "cache/numba",
        "POLARS_TEMP_DIR": WORK_ROOT / "tmp/polars",
        "MPLCONFIGDIR": WORK_ROOT / "cache/matplotlib",
        "JOBLIB_TEMP_FOLDER": WORK_ROOT / "tmp/joblib",
        "PYTHONPYCACHEPREFIX": WORK_ROOT / "cache/pycache",
        "TORCH_HOME": WORK_ROOT / "cache/torch",
        "PIP_CACHE_DIR": WORK_ROOT / "cache/pip",
        "HF_HOME": WORK_ROOT / "cache/huggingface",
        "JUPYTER_RUNTIME_DIR": WORK_ROOT / "cache/jupyter",
    }
    for path in set(paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    environment = {key: str(path) for key, path in paths.items()}
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
            "ARROW_NUM_THREADS": "1",
            "JOBLIB_MULTIPROCESSING": "0",
        }
    )
    return environment


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop only the admitted profile process tree, escalating after five seconds."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def discover_raw_sources(repo_root: Path) -> dict[str, Path]:
    """Locate one matching Kaggle raw dataset and verify both pinned Parquet footers."""
    train = find_unique_input("train.parquet")
    valid = find_unique_input("valid.parquet")
    if train.parent != valid.parent or "wunder-funder" not in train.parts:
        raise ProfileError("train/valid Parquet inputs do not share the expected dataset mount")
    sys.path.insert(0, str(repo_root / "src"))
    from wnn.training.age_readout_portable_builder import verify_parquet_source

    recipe = read_object(repo_root / MACHINE_RECIPE)
    identities: dict[str, dict[str, Any]] = {}
    for fold in recipe["folds"]:
        split_manifest = read_object(repo_root / fold["split_manifest"])
        for split in ("train", "valid"):
            identity = split_manifest["sources"][f"datasets/{split}.parquet"]
            prior = identities.get(split)
            if prior is not None and prior != identity:
                raise ProfileError(f"folds disagree on the pinned {split} Parquet identity")
            identities[split] = identity
    sources = {"train": train, "valid": valid}
    for split, path in sources.items():
        verify_parquet_source(path, identities[split])
    return sources


def write_portable_paths(
    bundle_root: Path, raw_sources: dict[str, Path], transport_sha: str
) -> Path:
    """Bind the extracted immutable inputs to fresh disk-backed working directories."""
    value = {
        "schema": "wnn.age-readout-portable-path-map.v1",
        "bundle_root": str(bundle_root),
        "raw_sources": {name: str(path) for name, path in raw_sources.items()},
        "cache_root": str(WORK_ROOT / "cache/folds"),
        "output_root": str(WORK_ROOT / "runs"),
        "resume_root": str(WORK_ROOT / "resume"),
        "profile_root": str(WORK_ROOT / "profiles"),
        "workspace_root": str(WORK_ROOT),
        "transport_sha256": transport_sha,
        "cleanup_after_success": True,
        "require_kaggle_paths": True,
    }
    path = bundle_root / "portable_paths.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_source_manifest(repo_root: Path, admission: dict[str, Any], deadline: float) -> str:
    """Rehash every CUDA profile source pin and compare optional root source-map pins."""
    manifest_path = repo_root / SOURCE_MANIFEST_GPU_R2
    manifest = read_object(manifest_path)
    if (
        manifest.get("schema") != "wnn.age-readout-source-manifest.v1"
        or manifest.get("recipe_sha256") != MACHINE_RECIPE_SHA256
        or not isinstance(manifest.get("source_sha256"), dict)
        or not isinstance(manifest.get("fit_source_sha256"), dict)
    ):
        raise ProfileError("CUDA profile source manifest does not bind the frozen recipe")
    maps = (manifest["source_sha256"], manifest["fit_source_sha256"])
    for source_map in maps:
        for relative, expected in source_map.items():
            path = repo_root / safe_relative_path(relative)
            if sha256_file(path, deadline) != expected:
                raise ProfileError(f"CUDA profile source module changed after freeze: {relative}")
    admission_sources = admission.get("source_sha256", {})
    if isinstance(admission_sources, dict):
        for relative, expected in admission_sources.items():
            relative_name = safe_relative_path(relative)
            candidate = repo_root / relative_name
            if candidate.is_file() and sha256_file(candidate, deadline) != expected:
                raise ProfileError(f"root source pin differs in extracted code: {relative}")
    expected_manifest_sha = admission.get("source_manifest_sha256")
    observed_manifest_sha = sha256_file(manifest_path, deadline, MAX_METADATA_BYTES)
    if expected_manifest_sha != observed_manifest_sha:
        raise ProfileError("CUDA source manifest SHA differs from root admission")
    if normalized_driver_sha256(Path(__file__).resolve()) != admission.get(
        "driver_template_sha256"
    ):
        raise ProfileError("rendered profile kernel differs from its admitted template")
    return observed_manifest_sha


def normalized_driver_sha256(path: Path) -> str:
    """Hash the final kernel after restoring its single admission-SHA marker."""
    source = path.read_text(encoding="utf-8")
    rendered = f'PROFILE_ADMISSION_SHA256 = "{PROFILE_ADMISSION_SHA256}"'
    template = 'PROFILE_ADMISSION_SHA256 = "__PROFILE_ADMISSION_SHA256__"'
    if source.count(rendered) != 1:
        raise ProfileError("rendered kernel has an unexpected admission constant")
    return hashlib.sha256(source.replace(rendered, template, 1).encode()).hexdigest()


def make_profile_command(
    repo_root: Path,
    path_map: Path,
    admission_path: Path,
    source_manifest_sha256: str,
) -> list[str]:
    """Build the actor's exact CUDA profile command with the mounted portable map."""
    return [
        sys.executable,
        str(repo_root / "scripts/age_readout_screen.py"),
        "profile",
        "--repo-root",
        str(repo_root),
        "--recipe",
        MACHINE_RECIPE,
        "--recipe-sha256",
        MACHINE_RECIPE_SHA256,
        "--fold",
        "0",
        "--device",
        "cuda",
        "--admission",
        str(admission_path),
        "--source-manifest",
        str(repo_root / SOURCE_MANIFEST_GPU_R2),
        "--source-manifest-sha256",
        source_manifest_sha256,
        "--portable-config",
        str(path_map),
        "--profile-root",
        str(WORK_ROOT / "profiles"),
    ]


def run_profile(
    command: list[str],
    repo_root: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    deadline: float,
    peak_swap: list[int],
) -> tuple[int, int, float]:
    """Execute the profile with a hard wall limit and a four-gigabyte RSS watchdog."""
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    peak_rss = 0
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("profile exceeded its root-admitted process deadline")
                rss, swap = process_resources(process)
                peak_rss = max(peak_rss, rss)
                peak_swap[0] = max(peak_swap[0], swap)
                if rss >= PROCESS_RSS_MAX_BYTES:
                    raise MemoryError("profile process tree reached its four-gigabyte RSS ceiling")
                if swap != 0:
                    raise MemoryError("profile process tree used swap")
                time.sleep(POLL_SECONDS)
            returncode = int(process.returncode or 0)
            return returncode, peak_rss, time.monotonic() - started
    except BaseException:
        if process is not None and process.poll() is None:
            stop_process_group(process)
        raise


def chmod_read_only(root: Path) -> None:
    """Make the extracted bundle immutable after its generated path map is written."""
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def remove_generated_tree(root: Path) -> None:
    """Remove only this driver's generated scratch tree under disk scratch."""
    if root.is_symlink() or not root.is_relative_to(DISK_ROOT):
        raise ProfileError("refusing to remove a path outside the admitted disk scratch root")
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)
        elif not path.is_symlink():
            path.chmod(0o600)
    root.chmod(0o700)
    shutil.rmtree(root)


def clean_profile_outputs(profile_root: Path) -> list[str]:
    """Keep only bounded JSON profile receipts and remove other actor outputs."""
    errors: list[str] = []
    if profile_root.is_symlink():
        return [f"profile receipt root is a symlink: {profile_root}"]
    if not profile_root.exists():
        return errors
    for path in sorted(profile_root.rglob("*"), reverse=True):
        try:
            if path.is_symlink() or (
                path.is_file()
                and (path.suffix.lower() != ".json" or path.stat().st_size > MAX_METADATA_BYTES)
            ):
                path.unlink()
            elif path.is_dir():
                with suppress(OSError):
                    path.rmdir()
        except OSError as error:
            errors.append(f"cannot clean profile artifact {path}: {error}")
    return errors


def main() -> int:
    """Verify, extract, and run exactly one admitted strict-FP32 CUDA F0 profile."""
    started = time.monotonic()
    total_deadline = started + PROFILE_MAX_SECONDS
    DISK_ROOT.mkdir(parents=True, exist_ok=True)
    if DISK_ROOT.is_symlink() or Path("/kaggle/working").is_symlink():
        raise ProfileError("Kaggle working disk path contains a symlink")
    for path in (BUNDLE_ROOT, WORK_ROOT):
        if path.exists() or path.is_symlink():
            raise ProfileError(f"refusing to overwrite prior run data: {path}")
    phases: dict[str, float] = {}
    peak_rss = [0]
    peak_swap = [0]
    admission_path: Path | None = None
    archive_path: Path | None = None
    overlay_path: Path | None = None
    admission: dict[str, Any] = {}
    pins: dict[str, Any] = {}
    run_deadline = total_deadline - FINALIZATION_RESERVE_SECONDS
    admission_sha = ""
    observed_archive_sha = ""
    archive_bytes = 0
    observed_overlay_sha = ""
    extracted_bytes = file_count = 0
    manifest_sha = ""
    export_receipt_sha = ""
    source_manifest_sha = ""
    source_overlay_manifest_sha = ""
    source_overlay_file_count = 0
    path_map_sha = ""
    returncode: int | None = None
    profile_wall_seconds = 0.0
    profile_peak_rss = 0
    cpu_ids: list[int] = []
    raw_sources: dict[str, Path] = {}
    command: list[str] = []
    stdout_path = WORK_ROOT / "profile.stdout.log"
    stderr_path = WORK_ROOT / "profile.stderr.log"
    receipt_path = WORK_ROOT / "PROFILE_DRIVER_RECEIPT_CUDA_R1.json"
    status, failure = "failed", None
    try:
        admission_path = find_unique_input(PROFILE_ADMISSION_NAME)
        archive_path = find_unique_input(BUNDLE_ARCHIVE_NAME)
        overlay_path = find_unique_input(SOURCE_OVERLAY_ARCHIVE_NAME)
        if admission_path.parent != overlay_path.parent:
            raise ProfileError(
                "CUDA source overlay and root admission are not in the same dataset mount"
            )
        admission = read_object(admission_path, maximum_bytes=2_000_000)
        pins = validate_admission(admission)
        admission_sha = sha256_file(admission_path, total_deadline, maximum_bytes=2_000_000)
        if SHA256_RE.fullmatch(PROFILE_ADMISSION_SHA256) is None:
            raise ProfileError("profile kernel was not rendered with a frozen admission SHA")
        if admission_sha != PROFILE_ADMISSION_SHA256:
            raise ProfileError("profile admission differs from the SHA embedded in this kernel")
        run_deadline = (
            min(total_deadline, started + pins["wall_seconds"]) - FINALIZATION_RESERVE_SECONDS
        )
        archive_bytes = archive_path.stat().st_size
        if archive_bytes > MAX_ARCHIVE_BYTES or archive_bytes != pins["bundle_archive_bytes"]:
            raise ProfileError("portable archive byte count differs from admission or upload bound")
        cpu_ids = sorted(os.sched_getaffinity(0))[: pins["cpu_cores"]]
        if not cpu_ids:
            raise ProfileError("Kaggle worker exposes no CPU affinity IDs")
        os.sched_setaffinity(0, set(cpu_ids))
        environment = os.environ.copy()
        environment.update(set_disk_environment())
        environment["PYTHONHASHSEED"] = "0"
        phase = time.monotonic()
        observed_archive_sha = sha256_file(archive_path, run_deadline, MAX_ARCHIVE_BYTES)
        phases["archive_hash_seconds"] = time.monotonic() - phase
        if observed_archive_sha != pins["bundle_archive_sha256"]:
            raise ProfileError("portable archive SHA differs from the root admission")
        observed_overlay_sha = sha256_file(
            overlay_path, run_deadline, MAX_OVERLAY_ARCHIVE_BYTES
        )
        if observed_overlay_sha != pins["source_overlay_archive_sha256"]:
            raise ProfileError("CUDA source overlay SHA differs from root admission")

        phase = time.monotonic()
        extracted_bytes, file_count, bundle_entries = safe_extract(
            archive_path, BUNDLE_ROOT, run_deadline, peak_rss, peak_swap
        )
        manifest_sha = sha256_file(
            BUNDLE_ROOT / BUNDLE_MANIFEST_NAME, run_deadline, MAX_METADATA_BYTES
        )
        if manifest_sha != pins["bundle_manifest_sha256"]:
            raise ProfileError("bundle manifest SHA differs from root admission")
        export_receipt_sha = verify_export_outputs(
            BUNDLE_ROOT, bundle_entries, admission, run_deadline
        )
        source_overlay_manifest_sha, source_overlay_file_count = apply_source_overlay(
            BUNDLE_ROOT, overlay_path, admission, run_deadline
        )
        phases["safe_extract_seconds"] = time.monotonic() - phase
        transport_path = BUNDLE_ROOT / "transport.json"
        if (
            sha256_file(transport_path, run_deadline, MAX_METADATA_BYTES)
            != pins["transport_sha256"]
        ):
            raise ProfileError("extracted transport SHA differs from the root admission")
        repo_root = BUNDLE_ROOT / ACTOR_SOURCE_ROOT
        phase = time.monotonic()
        source_manifest_sha = verify_source_manifest(repo_root, admission, run_deadline)
        raw_sources = discover_raw_sources(repo_root)
        phases["source_and_raw_validation_seconds"] = time.monotonic() - phase
        path_map = write_portable_paths(BUNDLE_ROOT, raw_sources, pins["transport_sha256"])
        path_map_sha = sha256_file(path_map, run_deadline, MAX_METADATA_BYTES)
        chmod_read_only(BUNDLE_ROOT)
        command = make_profile_command(
            repo_root,
            path_map,
            admission_path,
            pins["source_manifest_sha256"],
        )
        prior_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(repo_root / "src"), prior_pythonpath) if value
        )
        environment["WNN_TMP_ROOT"] = str(WORK_ROOT / "tmp")
        phase = time.monotonic()
        returncode, profile_peak_rss, profile_wall_seconds = run_profile(
            command,
            repo_root,
            environment,
            stdout_path,
            stderr_path,
            run_deadline,
            peak_swap,
        )
        phases["profile_seconds"] = time.monotonic() - phase
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        fold_cache = WORK_ROOT / "cache/folds/fold0"
        if fold_cache.exists():
            remove_generated_tree(fold_cache)
        status = "profile_complete"
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        status = "failed"
    finally:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        cleanup_errors: list[str] = []
        for generated in (
            BUNDLE_ROOT,
            BUNDLE_ROOT.with_name(BUNDLE_ROOT.name + ".partial"),
            WORK_ROOT / "tmp",
            WORK_ROOT / "cache",
            WORK_ROOT / "runs",
            WORK_ROOT / "resume",
        ):
            try:
                remove_generated_tree(generated)
            except (OSError, ProfileError) as error:
                cleanup_errors.append(f"cannot remove generated tree {generated}: {error}")
        cleanup_errors.extend(clean_profile_outputs(WORK_ROOT / "profiles"))
        if cleanup_errors:
            failure = {
                "type": "CleanupError",
                "message": "; ".join(cleanup_errors),
                "prior_failure": failure,
            }
            status = "failed"
        observed_admission_sha = admission_sha
        if admission_path is not None and admission_path.is_file():
            observed_admission_sha = sha256_file(
                admission_path, time.monotonic() + 10, maximum_bytes=2_000_000
            )
        receipt = {
            "schema": "wnn.age-readout-profile-driver-receipt.v1",
            "status": status,
            "mode": "cuda_profile_only",
            "profile_admission_sha256": observed_admission_sha or None,
            "bundle_archive_sha256": observed_archive_sha,
            "bundle_archive_bytes": archive_bytes,
            "bundle_manifest_sha256": manifest_sha or None,
            "export_receipt_sha256": export_receipt_sha or None,
            "transport_sha256": pins.get("transport_sha256"),
            "source_manifest_sha256": source_manifest_sha,
            "source_overlay_archive_sha256": observed_overlay_sha,
            "source_overlay_manifest_sha256": source_overlay_manifest_sha,
            "source_overlay_file_count": source_overlay_file_count,
            "portable_paths_sha256": path_map_sha or None,
            "raw_sources": {
                name: {"path": str(path), "bytes": path.stat().st_size}
                for name, path in (locals().get("raw_sources") or {}).items()
            },
            "workspace_root": str(WORK_ROOT),
            "bundle_root": str(BUNDLE_ROOT),
            "archive_expanded_bytes": extracted_bytes,
            "archive_file_count": file_count,
            "cpu_affinity": cpu_ids,
            "torch_threads": 1,
            "gpu_requested": True,
            "gpu_visibility_verified_by_profile": returncode == 0,
            "precision_contract": "strict_fp32_tf32_off_autocast_disabled",
            "engineering_microfits": 3,
            "engineering_epochs_per_microfit": 3,
            "engineering_updates_per_microfit": 120,
            "engineering_optimizer_updates": 360,
            "training_subset_labels_read": True,
            "evaluation_label_columns_read": False,
            "quality_scoring": False,
            "new_scientific_fits": 0,
            "six_cell_scientific_panel": "unadmitted",
            "profile_exit_code": returncode,
            "profile_peak_process_tree_rss_bytes": profile_peak_rss,
            "extraction_peak_process_rss_bytes": peak_rss[0],
            "task_swap_peak_bytes": peak_swap[0],
            "phase_wall_seconds": phases,
            "profile_wall_seconds": profile_wall_seconds,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "failure": failure,
            "cleanup_errors": cleanup_errors,
            "preserved_diagnostics": [
                str(path)
                for path in (
                    receipt_path,
                    stdout_path,
                    stderr_path,
                    *(WORK_ROOT / "profiles").rglob("*.json"),
                )
                if path.is_file()
            ],
            "elapsed_seconds": time.monotonic() - started,
        }
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if status == "profile_complete":
        print(json.dumps({"status": status, "receipt": str(receipt_path)}, sort_keys=True))
        return 0
    print(json.dumps({"status": status, "receipt": str(receipt_path)}, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
