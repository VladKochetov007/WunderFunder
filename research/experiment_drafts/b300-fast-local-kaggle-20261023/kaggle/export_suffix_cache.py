#!/usr/bin/env python3
"""Export only the admitted fit-cache suffix and cached parent endpoints.

The command is intentionally read-only until a root-pinned admission passes.
It writes an unpublished sibling tree, then atomically publishes the complete
transport only after all source identities and output hashes are complete.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import io
import json
import mmap
import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

STORAGE_ROOT = Path("/home/vlad/development/wunder/research_disk/b300-fast-20261023")
MACHINE_RECIPE = Path("research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json")
CACHE_RECEIPT = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/LOCAL_CACHE_VERIFICATION_R1.json"
)
OPERATIONAL_AMENDMENT = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/KAGGLE_CPU_DELIVERY_AMENDMENT_R1.json"
)
SOURCE_RECIPE = Path(
    "research/experiment_drafts/online-relationships-20261022/EXECUTION_RECIPE_R1.json"
)
PARENT_MILESTONE = Path(
    "research/experiment_drafts/online-relationships-20261022/PARENT_CACHE_MILESTONE_R1.json"
)
ACTUAL_PARENT_CHECK = STORAGE_ROOT / "grader/ACTUAL_PARENT_METADATA_CHECK_R1.json"
EXPORTER_KEY = "kaggle_ops/export_suffix_cache.py"
MACHINE_RECIPE_SHA256 = "0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd"
SOURCE_RECIPE_SHA256 = "edc873c813de77f55859d1a13180d907004e5f18fcc537e64eda7af1308e2508"
PARENT_MILESTONE_SHA256 = "0dd9f9cec9bf54a4c2abdaeb835ad13f099e201a4b578714838014e39eb4d11e"
LOCAL_CACHE_SHA256 = "6659cdb989beceb75778e978699a46fede50d6f6661887354279883cbab4e9c1"
ACTUAL_PARENT_CHECK_SHA256 = "ed540e31b5d79da39e13d624b9c4faadf277d9efe953d7f8c24eecad3e4467c6"
FIT_SUFFIX_COLUMNS = (112, 117)
SEQUENCE_STEPS = 20_000
SOURCE_FEATURES = 117
SUFFIX_FEATURES = 5
FLOAT_BYTES = 4
CHUNK_BYTES = 8 * 1024**2
MAX_BUNDLE_BYTES = 1_600_000_000
MAX_RSS_BYTES = 1_000_000_000
MAX_HOST_MEMORY_PRESSURE = 2.0
MIN_HOST_AVAILABLE_BYTES = 8_000_000_000
MAX_CPU_PACKAGE_CELSIUS = 97.0
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ExportError(RuntimeError):
    """An input, resource, or publication check failed closed."""


def read_json(path: Path, *, maximum_bytes: int = 32_000_000) -> dict[str, Any]:
    """Read one bounded metadata object, never an array payload."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ExportError(f"metadata input is absent, nonregular, or oversized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExportError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path, deadline: float, *, maximum_bytes: int | None = None) -> str:
    """Hash a bounded file with fixed memory and deadline checks."""
    if path.is_symlink() or not path.is_file():
        raise ExportError(f"hash input is absent or nonregular: {path}")
    if maximum_bytes is not None and path.stat().st_size > maximum_bytes:
        raise ExportError(f"metadata hash input is oversized: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            check_deadline(deadline)
            digest.update(block)
    return digest.hexdigest()


def check_deadline(deadline: float) -> None:
    """Stop before exceeding the root-admitted wall cap."""
    if time.monotonic() >= deadline:
        raise TimeoutError("suffix export reached its admitted wall-time cap")


def _source_for_key(key: str, repo_root: Path) -> Path:
    """Resolve the exporter key or a safe, root-relative helper source pin."""
    path = PurePosixPath(key)
    if not key or path.is_absolute() or ".." in path.parts or "\\" in key:
        raise ExportError(f"unsafe source pin path: {key}")
    if key == EXPORTER_KEY:
        return Path(__file__).resolve(strict=True)
    if key.startswith("kaggle_ops/"):
        return STORAGE_ROOT / Path(*path.parts)
    return repo_root / Path(*path.parts)


def validate_admission(
    admission: dict[str, Any],
    admission_path: Path,
    admission_sha256: str,
    repo_root: Path,
    deadline: float,
) -> tuple[Path, str]:
    """Require the exact root-issued, no-fit export envelope before payload access."""
    actual_admission_sha = sha256_file(admission_path, deadline, maximum_bytes=2_000_000)
    if actual_admission_sha != admission_sha256:
        raise ExportError("root suffix-export admission SHA256 differs")
    if (
        admission.get("schema") != "wnn.age-readout-suffix-export-admission.v1"
        or admission.get("status") != "admitted"
        or admission.get("recipe_sha256") != MACHINE_RECIPE_SHA256
        or admission.get("folds") != [0, 1]
        or admission.get("mode") != "readonly_original_cache_suffix_copy"
        or admission.get("new_scientific_fits") != 0
    ):
        raise ExportError("root suffix-export admission is incomplete or mismatched")
    amendment_sha = sha256_file(repo_root / OPERATIONAL_AMENDMENT, deadline)
    if admission.get("operational_amendment_sha256") != amendment_sha:
        raise ExportError("operational amendment SHA256 differs from admission")
    cache = admission.get("cache_verification", {})
    if cache.get("path") != CACHE_RECEIPT.as_posix() or cache.get("sha256") != LOCAL_CACHE_SHA256:
        raise ExportError("admission does not bind the frozen local cache receipt")
    limits = admission.get("limits", {})
    if (
        float(limits.get("cpu_cores", 0)) != 0.25
        or int(limits.get("ram_bytes", 0)) > MAX_RSS_BYTES
        or int(limits.get("swap_bytes", -1)) != 0
        or not 1 <= int(limits.get("wall_seconds", 0)) <= 300
    ):
        raise ExportError("suffix export exceeds its local CPU/RAM/swap/wall envelope")
    output_root = Path(str(admission.get("output_root", ""))).resolve(strict=False)
    if (
        not output_root.is_absolute()
        or not output_root.is_relative_to(STORAGE_ROOT.resolve())
        or output_root == STORAGE_ROOT.resolve()
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise ExportError("admitted output root must be a fresh physical-disk path")
    source_pins = admission.get("source_sha256", {})
    if not isinstance(source_pins, dict) or source_pins.get(EXPORTER_KEY) != sha256_file(
        Path(__file__).resolve(), deadline
    ):
        raise ExportError("root admission does not pin this exporter source")
    for key, digest in source_pins.items():
        if (
            not isinstance(key, str)
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise ExportError("admission source_sha256 mapping is malformed")
        if sha256_file(_source_for_key(key, repo_root), deadline) != digest:
            raise ExportError(f"admitted source pin differs: {key}")
    return output_root, actual_admission_sha


def record_index(cache: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    """Return the exact two-fold local cache verification entries."""
    if (
        cache.get("schema") != "wnn.age-readout-local-cache-verification.v1"
        or cache.get("status") != "PASS_PAYLOAD_HASH_AND_HEADER"
        or cache.get("recipe_sha256") != MACHINE_RECIPE_SHA256
        or cache.get("source_cache_recipe", {}).get("sha256") != SOURCE_RECIPE_SHA256
    ):
        raise ExportError("local cache verification receipt differs from frozen inputs")
    records: dict[tuple[int, str], dict[str, Any]] = {}
    for item in cache.get("files", []):
        key = (int(item["fold"]), str(item["kind"]))
        if key in records:
            raise ExportError(f"duplicate cache receipt entry: {key}")
        records[key] = item
    expected = {
        (fold, kind)
        for fold in (0, 1)
        for kind in ("fit_sequences", "evaluation_sequences", "evaluation_predictions")
    }
    if set(records) != expected:
        raise ExportError("cache receipt does not contain exactly six fold artifacts")
    return records


def stat_identity(path: Path) -> dict[str, int]:
    """Return the inode identity used by the prior full-payload verification."""
    if path.is_symlink() or not path.is_file():
        raise ExportError(f"cache source is absent, nonregular, or a symlink: {path}")
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def compare_source_identity(path: Path, record: dict[str, Any]) -> dict[str, int]:
    """Match current stat identity to the full local payload-hash receipt."""
    expected = record["identity"]
    before = stat_identity(path)
    if any(before[key] != int(expected[key]) for key in before):
        raise ExportError(f"cache source inode identity differs from full-hash receipt: {path}")
    if before["bytes"] != int(record["header"]["file_bytes"]):
        raise ExportError(f"cache NPY byte length differs from its pinned header: {path}")
    return before


def read_npy_header(path: Path, record: dict[str, Any]) -> tuple[bytes, tuple[int, ...], str]:
    """Read only the pinned 128-byte NPY header, never the array payload."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ExportError(f"expected NPY v1.0 source header: {path}")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        header_bytes = stream.tell()
        if header_bytes != 128 or header_bytes != int(record["header"]["header_bytes"]):
            raise ExportError(
                f"source NPY header length differs from the full-hash receipt: {path}"
            )
        stream.seek(0)
        raw = stream.read(header_bytes)
    if (
        len(raw) != 128
        or list(shape) != record["header"]["shape"]
        or fortran_order is not False
        or dtype.str != record["header"]["descr"]
    ):
        raise ExportError(f"source NPY header differs from the full-hash receipt: {path}")
    return raw, tuple(shape), dtype.str


def _read_cpu_package_temperature() -> float:
    """Read the hottest package sensor without probing or initializing a GPU."""
    root = Path("/sys/class/hwmon")
    values: list[float] = []
    for label_path in sorted(root.glob("hwmon*/temp*_label")):
        try:
            label = label_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ExportError("CPU package sensor label is unreadable") from error
        if label.startswith("Package id "):
            input_path = label_path.with_name(f"{label_path.name.removesuffix('_label')}_input")
            try:
                values.append(int(input_path.read_text(encoding="utf-8").strip()) / 1000.0)
            except (OSError, ValueError) as error:
                raise ExportError("CPU package sensor input is unreadable") from error
    if not values:
        raise ExportError("CPU package thermal guard found no readable Package id sensor")
    return max(values)


def _resource_sample(deadline: float, started: float) -> dict[str, int | float]:
    """Enforce the source-export wall, RSS, swap, thermal and host-pressure guards."""
    check_deadline(deadline)
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    fields = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in status.splitlines()
        if ":" in line
    }
    rss = int(fields.get("VmRSS", "0 kB").split()[0]) * 1024
    swap = int(fields.get("VmSwap", "0 kB").split()[0]) * 1024
    if rss > MAX_RSS_BYTES or swap != 0:
        raise ExportError(f"exporter process resource guard failed: rss={rss}, swap={swap}")
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        if line.startswith("MemAvailable:")
    )
    pressure = next(
        line
        for line in Path("/proc/pressure/memory").read_text(encoding="utf-8").splitlines()
        if line.startswith("full ")
    )
    stall = float(next(field[6:] for field in pressure.split() if field.startswith("avg10=")))
    temperature = _read_cpu_package_temperature()
    if available < MIN_HOST_AVAILABLE_BYTES or stall >= MAX_HOST_MEMORY_PRESSURE:
        raise ExportError(
            f"host memory-pressure guard failed: available={available}, full_avg10={stall}"
        )
    if temperature >= MAX_CPU_PACKAGE_CELSIUS:
        raise ExportError(f"CPU package reached {temperature:.1f} C; suffix export aborted")
    return {
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(fields.get("VmHWM", "0 kB").split()[0]) * 1024,
        "current_rss_bytes": rss,
        "task_swap_bytes": swap,
        "host_available_bytes": available,
        "host_memory_full_avg10": stall,
        "cpu_package_temperature_celsius": temperature,
    }


def _aligned_madvise(mapping: mmap.mmap, start: int, length: int, page_size: int) -> None:
    """Evict the just-consumed file-backed row pages from the mmap RSS."""
    aligned_start = start - (start % page_size)
    aligned_end = ((start + length + page_size - 1) // page_size) * page_size
    map_length = len(mapping)
    aligned_end = min(aligned_end, map_length)
    mapping.madvise(mmap.MADV_DONTNEED, aligned_start, aligned_end - aligned_start)


def _check_sidecar(source: Path, expected_sha256: str, deadline: float) -> tuple[bytes, str]:
    """Verify a small original producer sidecar and return its unchanged bytes."""
    before = stat_identity(source)
    observed_sha = sha256_file(source, deadline, maximum_bytes=2_000_000)
    if observed_sha != expected_sha256:
        raise ExportError(f"producer sidecar differs from its cache/milestone pin: {source}")
    payload = source.read_bytes()
    if stat_identity(source) != before:
        raise ExportError(f"producer sidecar changed while being copied: {source}")
    return payload, observed_sha


def _write_suffix_file(
    source_path: Path,
    source_record: dict[str, Any],
    output_path: Path,
    deadline: float,
    started: float,
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Write cols 112:117 by sequence from a validated read-only mmap."""
    before = compare_source_identity(source_path, source_record)
    original_header, shape, dtype = read_npy_header(source_path, source_record)
    if (
        shape[1:] != (SEQUENCE_STEPS, SOURCE_FEATURES)
        or dtype != "<f4"
        or int(source_record["identity"]["bytes"])
        != 128 + shape[0] * SEQUENCE_STEPS * SOURCE_FEATURES * FLOAT_BYTES
    ):
        raise ExportError(f"fit cache source shape or byte arithmetic differs: {source_path}")
    output_shape = (shape[0], SEQUENCE_STEPS, SUFFIX_FEATURES)
    header_stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header_stream,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype("<f4")),
            "fortran_order": False,
            "shape": output_shape,
        },
    )
    output_header = header_stream.getvalue()
    row_bytes = SEQUENCE_STEPS * SOURCE_FEATURES * FLOAT_BYTES
    output_digest = hashlib.sha256(output_header)
    madvise_calls = 0
    page_size = os.sysconf("SC_PAGE_SIZE")
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(source, np.memmap) or source.shape != shape or source.dtype.str != "<f4":
        raise ExportError(
            f"fit sequence source did not open as the pinned little-endian mmap: {source_path}"
        )
    mapping = source._mmap
    if not hasattr(mapping, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
        raise ExportError("platform lacks the required mmap MADV_DONTNEED operation")
    mapping.madvise(mmap.MADV_SEQUENTIAL)
    array_offset = int(source.offset)
    mapping_file_offset = array_offset - (array_offset % mmap.ALLOCATIONGRANULARITY)
    last_sample: dict[str, int | float] = {}
    try:
        with output_path.open("xb", buffering=CHUNK_BYTES) as output:
            output.write(output_header)
            for sequence in range(shape[0]):
                last_sample = _resource_sample(deadline, started)
                row = np.ascontiguousarray(
                    source[sequence, :, FIT_SUFFIX_COLUMNS[0] : FIT_SUFFIX_COLUMNS[1]]
                )
                if row.shape != (SEQUENCE_STEPS, SUFFIX_FEATURES) or row.dtype.str != "<f4":
                    raise ExportError("fit suffix row shape/dtype differs")
                block = memoryview(row).cast("B")
                output.write(block)
                output_digest.update(block)
                del block, row
                _aligned_madvise(
                    mapping,
                    array_offset + sequence * row_bytes - mapping_file_offset,
                    row_bytes,
                    page_size,
                )
                madvise_calls += 1
            output.flush()
            os.fsync(output.fileno())
    finally:
        del source
        gc.collect()
    after = stat_identity(source_path)
    if after != before:
        raise ExportError(f"source cache identity changed during suffix export: {source_path}")
    expected_output_bytes = (
        len(output_header) + shape[0] * SEQUENCE_STEPS * SUFFIX_FEATURES * FLOAT_BYTES
    )
    if output_path.stat().st_size != expected_output_bytes:
        raise ExportError(f"suffix output byte length differs: {output_path}")
    return (
        {
            "path": output_path.relative_to(output_path.parents[1]).as_posix(),
            "sha256": output_digest.hexdigest(),
            "bytes": expected_output_bytes,
            "shape": list(output_shape),
            "dtype": "<f4",
            "header_bytes": len(output_header),
        },
        {
            "base64": base64.b64encode(original_header).decode("ascii"),
            "sha256": hashlib.sha256(original_header).hexdigest(),
            "bytes": len(original_header),
        },
        madvise_calls,
        int(last_sample.get("peak_rss_bytes", 0)),
    )


def _copy_parent_predictions(
    source_path: Path,
    source_record: dict[str, Any],
    output_path: Path,
    deadline: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Stream-copy one original parent prediction NPY while verifying its full SHA."""
    before = compare_source_identity(source_path, source_record)
    source_header, shape, dtype = read_npy_header(source_path, source_record)
    if (
        shape[1:] != (SEQUENCE_STEPS, 2)
        or dtype != "<f4"
        or int(source_record["identity"]["bytes"])
        != 128 + shape[0] * SEQUENCE_STEPS * 2 * FLOAT_BYTES
    ):
        raise ExportError(f"parent prediction source header differs: {source_path}")
    digest = hashlib.sha256()
    total = 0
    with source_path.open("rb") as source, output_path.open("xb", buffering=CHUNK_BYTES) as output:
        while True:
            check_deadline(deadline)
            block = source.read(CHUNK_BYTES)
            if not block:
                break
            output.write(block)
            digest.update(block)
            total += len(block)
        output.flush()
        os.fsync(output.fileno())
    after = stat_identity(source_path)
    if after != before or total != int(source_record["identity"]["bytes"]):
        raise ExportError(f"parent prediction changed while copying: {source_path}")
    output_sha = digest.hexdigest()
    if output_sha != source_record["sha256"]:
        raise ExportError(f"parent prediction copy differs from the full cache hash: {source_path}")
    return (
        {
            "path": output_path.relative_to(output_path.parents[1]).as_posix(),
            "sha256": output_sha,
            "bytes": total,
            "shape": list(shape),
            "dtype": dtype,
            "header_bytes": len(source_header),
        },
        {
            "base64": base64.b64encode(source_header).decode("ascii"),
            "sha256": hashlib.sha256(source_header).hexdigest(),
            "bytes": len(source_header),
        },
        after,
    )


def _copy_verified_small(
    source: Path,
    destination: Path,
    bundle_root: Path,
    expected_sha256: str,
    deadline: float,
) -> dict[str, Any]:
    """Copy one admitted small recipe, sidecar, or attestation byte-for-byte."""
    before = stat_identity(source)
    payload, observed_sha = _check_sidecar(source, expected_sha256, deadline)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    after = stat_identity(source)
    if before != after:
        raise ExportError(f"metadata source changed while being staged: {source}")
    return {
        "path": destination.relative_to(bundle_root).as_posix(),
        "sha256": observed_sha,
        "bytes": len(payload),
    }


def export(args: argparse.Namespace) -> int:
    """Verify root admission, stage the exact suffix closure, and publish atomically."""
    started = time.monotonic()
    repo_root = args.repo_root.resolve(strict=True)
    admission = read_json(args.admission.resolve(strict=True), maximum_bytes=2_000_000)
    output_root, admission_sha = validate_admission(
        admission, args.admission, args.admission_sha256, repo_root, started + 300
    )
    wall_cap = int(admission["limits"]["wall_seconds"])
    deadline = started + wall_cap
    partial_root = output_root.with_name(output_root.name + ".partial")
    if partial_root.exists() or partial_root.is_symlink() or not partial_root.parent.is_dir():
        raise ExportError("partial export destination must be a fresh sibling on the same disk")

    cache_path = repo_root / CACHE_RECEIPT
    cache = read_json(cache_path)
    if sha256_file(cache_path, deadline, maximum_bytes=32_000_000) != LOCAL_CACHE_SHA256:
        raise ExportError("local cache verification receipt changed")
    records = record_index(cache)
    milestone_path = repo_root / PARENT_MILESTONE
    milestone = read_json(milestone_path)
    if sha256_file(milestone_path, deadline, maximum_bytes=32_000_000) != PARENT_MILESTONE_SHA256:
        raise ExportError("parent cache milestone changed")
    check_path = ACTUAL_PARENT_CHECK
    actual_check = read_json(check_path)
    if sha256_file(check_path, deadline, maximum_bytes=2_000_000) != ACTUAL_PARENT_CHECK_SHA256:
        raise ExportError("actual parent metadata check changed")
    if (
        actual_check.get("status") != "PASS_METADATA_AND_PARENT_PREDICTION_CHECKS"
        or actual_check.get("recipe_sha256") != MACHINE_RECIPE_SHA256
        or actual_check.get("cache_verification_sha256") != LOCAL_CACHE_SHA256
        or actual_check.get("parent_cache_milestone_sha256") != PARENT_MILESTONE_SHA256
        or actual_check.get("checks", {}).get("fit_and_eval_sequence_payloads_mapped_or_read")
        is not False
        or actual_check.get("checks", {}).get("target_cache_mapped_or_opened") is not False
        or actual_check.get("checks", {}).get("score_computed") is not False
    ):
        raise ExportError("actual parent check does not attest the no-label metadata-only scope")
    machine_recipe_path = repo_root / MACHINE_RECIPE
    source_recipe_path = repo_root / SOURCE_RECIPE
    if sha256_file(machine_recipe_path, deadline, maximum_bytes=4_000_000) != MACHINE_RECIPE_SHA256:
        raise ExportError("machine recipe bytes differ from the frozen source recipe")
    if sha256_file(source_recipe_path, deadline, maximum_bytes=16_000_000) != SOURCE_RECIPE_SHA256:
        raise ExportError("source cache recipe bytes differ from the full-cache attestation")

    all_records: dict[str, dict[str, Any]] = {}
    for fold in (0, 1):
        for kind in ("fit_sequences", "evaluation_sequences", "evaluation_predictions"):
            record = records[(fold, kind)]
            path = Path(record["source_path"])
            source_before = compare_source_identity(path, record)
            sidecar_path = Path(record["sidecar_path"])
            sidecar_before = stat_identity(sidecar_path)
            sidecar_sha = sha256_file(sidecar_path, deadline, maximum_bytes=2_000_000)
            if (
                sidecar_sha != record["sidecar_sha256"]
                or stat_identity(sidecar_path) != sidecar_before
            ):
                raise ExportError(
                    f"cache sidecar source differs from the full verification receipt: {sidecar_path}"
                )
            milestone_pin = milestone["parent_caches"][str(fold)]["sidecars"].get(sidecar_path.name)
            if not isinstance(milestone_pin, dict) or milestone_pin.get("sha256") != sidecar_sha:
                raise ExportError(
                    f"cache sidecar differs from the parent milestone: {sidecar_path}"
                )
            all_records[f"f{fold}_{kind}"] = {
                "record": record,
                "path": path,
                "sidecar_path": sidecar_path,
                "sidecar_sha256": sidecar_sha,
                "source_identity_before": source_before,
                "sidecar_identity_before": sidecar_before,
            }
    if args.preflight_only:
        _resource_sample(deadline, started)
        print(
            json.dumps(
                {"status": "preflight_passed_no_payload_read_or_write", "source_files": 6},
                sort_keys=True,
            )
        )
        return 0

    partial_root.mkdir(parents=False, exist_ok=False)
    source_identities: dict[str, Any] = {}
    output_entries: dict[str, Any] = {}
    maximum_rss_bytes = 0
    madvise_total = 0
    try:
        attestation_copy_map = (
            (
                cache_path,
                partial_root / "attestations/LOCAL_CACHE_VERIFICATION_R1.json",
                LOCAL_CACHE_SHA256,
            ),
            (
                milestone_path,
                partial_root / "attestations/PARENT_CACHE_MILESTONE_R1.json",
                PARENT_MILESTONE_SHA256,
            ),
            (
                check_path,
                partial_root / "attestations/ACTUAL_PARENT_METADATA_CHECK_R1.json",
                ACTUAL_PARENT_CHECK_SHA256,
            ),
            (machine_recipe_path, partial_root / MACHINE_RECIPE, MACHINE_RECIPE_SHA256),
            (source_recipe_path, partial_root / SOURCE_RECIPE, SOURCE_RECIPE_SHA256),
        )
        for source, destination, expected in attestation_copy_map:
            output_entries[destination.relative_to(partial_root).as_posix()] = _copy_verified_small(
                source, destination, partial_root, expected, deadline
            )

        fold_transport: dict[str, Any] = {}
        for fold in (0, 1):
            folder = partial_root / f"fold{fold}"
            folder.mkdir()
            fit = all_records[f"f{fold}_fit_sequences"]
            evaluation = all_records[f"f{fold}_evaluation_sequences"]
            parent = all_records[f"f{fold}_evaluation_predictions"]
            for kind, item in (
                ("fit_sequences", fit),
                ("evaluation_sequences", evaluation),
                ("evaluation_predictions", parent),
            ):
                destination = folder / f"{kind}.json"
                output_entries[destination.relative_to(partial_root).as_posix()] = (
                    _copy_verified_small(
                        item["sidecar_path"],
                        destination,
                        partial_root,
                        item["sidecar_sha256"],
                        deadline,
                    )
                )

            fit_header, fit_shape, fit_dtype = read_npy_header(fit["path"], fit["record"])
            eval_header, eval_shape, eval_dtype = read_npy_header(
                evaluation["path"], evaluation["record"]
            )
            if (
                fit_shape[1:] != (SEQUENCE_STEPS, SOURCE_FEATURES)
                or eval_shape[1:] != (SEQUENCE_STEPS, SOURCE_FEATURES)
                or fit_dtype != "<f4"
                or eval_dtype != "<f4"
            ):
                raise ExportError(
                    f"F{fold} full-cache headers differ from the frozen 117-column contract"
                )

            fit_suffix_path = folder / "fit_suffix5.npy"
            suffix, original_fit_header, madvise_calls, peak_rss = _write_suffix_file(
                fit["path"], fit["record"], fit_suffix_path, deadline, started
            )
            madvise_total += madvise_calls
            maximum_rss_bytes = max(maximum_rss_bytes, peak_rss)
            suffix_key = fit_suffix_path.relative_to(partial_root).as_posix()
            output_entries[suffix_key] = {
                "path": suffix_key,
                "sha256": suffix["sha256"],
                "bytes": fit_suffix_path.stat().st_size,
            }

            parent_out = folder / "evaluation_predictions.npy"
            parent_meta, parent_header, parent_identity_after = _copy_parent_predictions(
                parent["path"], parent["record"], parent_out, deadline
            )
            parent_key = parent_out.relative_to(partial_root).as_posix()
            output_entries[parent_key] = {
                "path": parent_key,
                "sha256": parent_meta["sha256"],
                "bytes": parent_out.stat().st_size,
            }

            for label, item in (
                ("fit_sequences", fit),
                ("evaluation_sequences", evaluation),
                ("evaluation_predictions", parent),
            ):
                source_identities[f"f{fold}_{label}"] = {
                    "source_path": str(item["path"]),
                    "source_sha256": item["record"]["sha256"],
                    "source_sidecar_path": str(item["sidecar_path"]),
                    "sidecar_sha256": item["sidecar_sha256"],
                    "source_identity_before": item["source_identity_before"],
                    "source_identity_after": compare_source_identity(item["path"], item["record"]),
                    "sidecar_identity_before": item["sidecar_identity_before"],
                    "sidecar_identity_after": stat_identity(item["sidecar_path"]),
                }
                if (
                    source_identities[f"f{fold}_{label}"]["source_identity_after"]
                    != item["source_identity_before"]
                    or source_identities[f"f{fold}_{label}"]["sidecar_identity_after"]
                    != item["sidecar_identity_before"]
                ):
                    raise ExportError(
                        "cache payload or sidecar identity changed before publication"
                    )

            fold_transport[str(fold)] = {
                "fit_suffix5": suffix,
                "evaluation_predictions": parent_meta,
                "sidecars": {
                    "fit_sequences": {
                        "path": f"fold{fold}/fit_sequences.json",
                        "sha256": fit["sidecar_sha256"],
                    },
                    "evaluation_sequences": {
                        "path": f"fold{fold}/evaluation_sequences.json",
                        "sha256": evaluation["sidecar_sha256"],
                    },
                    "evaluation_predictions": {
                        "path": f"fold{fold}/evaluation_predictions.json",
                        "sha256": parent["sidecar_sha256"],
                    },
                },
                "original_cache_headers": {
                    "fit_sequences": original_fit_header,
                    "evaluation_sequences": {
                        "base64": base64.b64encode(eval_header).decode("ascii"),
                        "sha256": hashlib.sha256(eval_header).hexdigest(),
                        "bytes": len(eval_header),
                    },
                },
            }
            if stat_identity(parent["path"]) != parent_identity_after:
                raise ExportError(f"parent prediction changed after copy: {parent['path']}")

        transport = {
            "schema": "wnn.age-readout-portable-transport.v1",
            "machine_recipe_sha256": MACHINE_RECIPE_SHA256,
            "source_cache_recipe_sha256": SOURCE_RECIPE_SHA256,
            "cache_verification_sha256": LOCAL_CACHE_SHA256,
            "parent_cache_milestone_sha256": PARENT_MILESTONE_SHA256,
            "actual_parent_metadata_check_sha256": ACTUAL_PARENT_CHECK_SHA256,
            "attestations": {
                "cache_verification": "attestations/LOCAL_CACHE_VERIFICATION_R1.json",
                "parent_cache_milestone": "attestations/PARENT_CACHE_MILESTONE_R1.json",
                "actual_parent_metadata_check": "attestations/ACTUAL_PARENT_METADATA_CHECK_R1.json",
            },
            "folds": fold_transport,
        }
        transport_path = partial_root / "transport.json"
        transport_bytes = (json.dumps(transport, indent=2, sort_keys=True) + "\n").encode()
        with transport_path.open("xb") as stream:
            stream.write(transport_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        output_entries["transport.json"] = {
            "path": "transport.json",
            "sha256": hashlib.sha256(transport_bytes).hexdigest(),
            "bytes": len(transport_bytes),
        }
        final_sample = _resource_sample(deadline, started)
        maximum_rss_bytes = max(maximum_rss_bytes, int(final_sample["peak_rss_bytes"]))
        bytes_before_receipt = sum(
            path.stat().st_size for path in partial_root.rglob("*") if path.is_file()
        )
        if bytes_before_receipt > MAX_BUNDLE_BYTES:
            raise ExportError(
                f"suffix transport exceeds the admitted 1.6 GB gate: {bytes_before_receipt}"
            )
        receipt = {
            "schema": "wnn.age-readout-suffix-export-receipt.v1",
            "status": "complete",
            "mode": "readonly_original_cache_suffix_copy",
            "admission_sha256": admission_sha,
            "operational_amendment_sha256": admission["operational_amendment_sha256"],
            "machine_recipe_sha256": MACHINE_RECIPE_SHA256,
            "source_cache_recipe_sha256": SOURCE_RECIPE_SHA256,
            "cache_verification_sha256": LOCAL_CACHE_SHA256,
            "parent_cache_milestone_sha256": PARENT_MILESTONE_SHA256,
            "actual_parent_metadata_check_sha256": ACTUAL_PARENT_CHECK_SHA256,
            "fit_suffix_columns_zero_based": list(FIT_SUFFIX_COLUMNS),
            "source_identities_before_after": source_identities,
            "outputs": output_entries,
            "madvise_dontneed_sequences": madvise_total,
            "peak_rss_bytes": maximum_rss_bytes,
            "task_swap_bytes": int(final_sample["task_swap_bytes"]),
            "final_cpu_package_temperature_celsius": final_sample[
                "cpu_package_temperature_celsius"
            ],
            "payload_and_metadata_bytes_excluding_this_receipt": bytes_before_receipt,
            "target_cache_read": False,
            "evaluation_sequence_payload_read": False,
            "raw_112_exported": False,
            "parent_inference_performed": False,
            "new_scientific_fits": 0,
            "provider_mutations": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
        receipt_path = partial_root / "EXPORT_RECEIPT_R1.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        total_bytes = sum(path.stat().st_size for path in partial_root.rglob("*") if path.is_file())
        if total_bytes > MAX_BUNDLE_BYTES:
            raise ExportError(
                f"suffix transport plus receipt exceeds the admitted 1.6 GB gate: {total_bytes}"
            )
        for file_path in partial_root.rglob("*"):
            if file_path.is_file():
                file_path.chmod(0o444)
        os.replace(partial_root, output_root)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output_root": str(output_root),
                    "transport_sha256": hashlib.sha256(transport_bytes).hexdigest(),
                    "bytes": total_bytes,
                    "elapsed_seconds": receipt["elapsed_seconds"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        failure = {
            "schema": "wnn.age-readout-suffix-export-receipt.v1",
            "status": "failed_unpublished",
            "admission_sha256": admission_sha,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.monotonic() - started,
            "outputs_published": False,
        }
        if partial_root.exists() and partial_root.is_dir():
            failure_path = partial_root / "EXPORT_FAILURE_R1.json"
            if not failure_path.exists():
                failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise


def parser() -> argparse.ArgumentParser:
    """Build the source-export CLI; no Kaggle API operation is available."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path.cwd())
    result.add_argument("--admission", type=Path, required=True)
    result.add_argument("--admission-sha256", required=True)
    result.add_argument("--preflight-only", action="store_true")
    return result


def main() -> int:
    """Run a root-admitted metadata preflight or bounded source-suffix export."""
    args = parser().parse_args()
    if SHA256_RE.fullmatch(args.admission_sha256) is None:
        raise ValueError("admission SHA256 must be lowercase hexadecimal")
    return export(args)


if __name__ == "__main__":
    raise SystemExit(main())
