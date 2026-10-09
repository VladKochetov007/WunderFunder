"""Independently verify H230 cache artifacts using bounded-memory streaming."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any  # JSON records are dynamically shaped external metadata.

import numpy as np

EXPECTED_FAMILIES = ["strong", "full_finetuned", "raw_wmse", "samefold_baseline"]
EXPECTED_SOURCES = [
    ("strong", "c20261003_combo_s3_f0", 3, 0.15),
    ("strong", "c20261003_combo_s4_f0", 3, 0.15),
    ("full_finetuned", "c20260929_ft_since_s3_f0", 0, 0.10),
    ("full_finetuned", "c20260929_ft_since_s4_f0", 0, 0.10),
    ("raw_wmse", "c20260929_since_s3_f0", 3, 0.125),
    ("raw_wmse", "c20260929_since_s4_f0", 3, 0.125),
    ("samefold_baseline", "bd26_w384raw_s3_f0", 3, 0.03125),
    ("samefold_baseline", "bd26_w384raw_s4_f0", 3, 0.03125),
]
ARRAY_WIDTHS = {
    "main_source_train": ("train", 2),
    "main_normalized_train": ("train", 2),
    "main_homogeneous_train": ("train", 2),
    "families_train": ("train", 8),
    "main_source_valid": ("valid", 2),
    "main_normalized_valid": ("valid", 2),
    "main_homogeneous_valid": ("valid", 2),
    "families_valid": ("valid", 8),
}
READ_BYTES = 8 << 20


def file_sha256(path: Path) -> str:
    """Return a file digest without retaining the file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(READ_BYTES):
            digest.update(block)
    return digest.hexdigest()


def row_groups_sha256(groups: list[int]) -> str:
    """Hash ordered row-group identities in the cache's little-endian format."""
    return hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()


def _fadvise_range(file_descriptor: int, start: int, size: int) -> None:
    """Ask Linux to discard the clean pages just read; failures are fatal."""
    if not hasattr(os, "posix_fadvise"):
        raise OSError("POSIX_FADV_DONTNEED is unavailable")
    page_size = os.sysconf("SC_PAGE_SIZE")
    aligned_start = start - (start % page_size)
    end = start + size
    aligned_end = ((end + page_size - 1) // page_size) * page_size
    os.posix_fadvise(
        file_descriptor,
        aligned_start,
        aligned_end - aligned_start,
        os.POSIX_FADV_DONTNEED,
    )


def inspect_npy_file(path: Path, expected_shape: tuple[int, ...]) -> dict[str, Any]:
    """Verify NPY header, exact payload digest and finite float32 values by stream."""
    started = time.monotonic()
    digest = hashlib.sha256()
    finite_values = 0
    advice_calls = 0
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
        payload_offset = stream.tell()
        expected_payload_bytes = int(np.prod(expected_shape, dtype=np.int64)) * 4
        file_bytes = path.stat().st_size
        if tuple(shape) != expected_shape:
            raise ValueError(f"array shape mismatch: {path.name}: {shape}")
        if dtype != np.dtype("float32") or fortran_order:
            raise ValueError(f"array dtype/order mismatch: {path.name}: {dtype}")
        if file_bytes - payload_offset != expected_payload_bytes:
            raise ValueError(f"array payload length mismatch: {path.name}")

        stream.seek(0)
        header = stream.read(payload_offset)
        if len(header) != payload_offset:
            raise OSError(f"short NPY header read: {path.name}")
        digest.update(header)
        _fadvise_range(stream.fileno(), 0, len(header))
        advice_calls += 1
        remaining = expected_payload_bytes
        file_offset = payload_offset
        while remaining:
            block = stream.read(min(READ_BYTES, remaining))
            if not block:
                raise OSError(f"short NPY payload read: {path.name}")
            if len(block) % dtype.itemsize:
                raise ValueError(f"unaligned float32 payload chunk: {path.name}")
            digest.update(block)
            values = np.frombuffer(block, dtype=dtype)
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite cache values: {path.name}")
            finite_values += int(values.size)
            remaining -= len(block)
            _fadvise_range(stream.fileno(), file_offset, len(block))
            advice_calls += 1
            file_offset += len(block)

    return {
        "path": str(path),
        "bytes": file_bytes,
        "sha256": digest.hexdigest(),
        "shape": list(shape),
        "dtype": str(dtype),
        "finite_values": finite_values,
        "fadvise_calls": advice_calls,
        "elapsed_seconds": time.monotonic() - started,
    }


def _ordered_fit_groups(data_root: Path, split_name: str, fit_groups: list[int]) -> list[int]:
    """Filter frozen target row order to the campaign fitting membership."""
    target_path = data_root / "cache" / f"{split_name}_targets.npz"
    with np.load(target_path, allow_pickle=False) as target_index:
        target_order = np.asarray(target_index["row_group"], dtype=np.int64)
    if len(np.unique(target_order)) != len(target_order):
        raise ValueError(f"target index repeats row groups: {target_path}")
    fitting = set(fit_groups)
    ordered = [int(group) for group in target_order if int(group) in fitting]
    if set(ordered) != fitting:
        raise ValueError(f"target index omits fitting groups: {split_name}")
    return ordered


def verify_cache(
    cache_dir: Path,
    data_root: Path,
    expected_metadata_sha256: str,
    expected_split_sha256: str,
    receipt_path: Path,
) -> dict[str, Any]:
    """Verify H230 cache recipe, membership and all eight outputs."""
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": "h230-cache-output-verification-v1",
        "status": "running",
        "cache_dir": str(cache_dir.resolve()),
        "data_root": str(data_root.resolve()),
        "verifier_path": str(Path(__file__).resolve()),
        "verifier_sha256": file_sha256(Path(__file__).resolve()),
        "expected_metadata_sha256": expected_metadata_sha256,
        "expected_split_sha256": expected_split_sha256,
        "observed_metadata_sha256": None,
        "observed_split_sha256": None,
        "files": {},
        "failure": None,
        "started_unix": time.time(),
    }

    def write_receipt() -> None:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt_path.with_suffix(receipt_path.suffix + ".partial")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(temporary, receipt_path)

    write_receipt()
    try:
        metadata_path = cache_dir / "metadata.json"
        receipt["observed_metadata_sha256"] = file_sha256(metadata_path)
        split_path = data_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
        receipt["observed_split_sha256"] = file_sha256(split_path)
        write_receipt()
        if receipt["observed_metadata_sha256"] != expected_metadata_sha256:
            raise ValueError("cache metadata digest differs from frozen build output")
        if receipt["observed_split_sha256"] != expected_split_sha256:
            raise ValueError("fold manifest digest differs from frozen recipe")

        metadata = json.loads(metadata_path.read_text())
        split = json.loads(split_path.read_text())
        if metadata.get("format") != "wnn.heterogeneous-distillation-cache":
            raise ValueError("unexpected cache format")
        if metadata.get("version") != 1 or metadata.get("fold") != 0:
            raise ValueError("unexpected cache version/fold")
        if metadata.get("split_manifest_sha256") != expected_split_sha256:
            raise ValueError("cache split identity differs")
        if metadata.get("family_order") != EXPECTED_FAMILIES:
            raise ValueError("cache family order differs")
        if metadata.get("coefficient_sum") != 0.8125:
            raise ValueError("cache coefficient sum differs")
        if metadata.get("homogeneous_family") != "strong":
            raise ValueError("homogeneous control family differs")
        if metadata.get("source_coefficients") != [
            {"family": family, "run": run, "epoch": epoch, "coefficient": coefficient}
            for family, run, epoch, coefficient in EXPECTED_SOURCES
        ]:
            raise ValueError("cache source recipe differs")

        fit_groups = metadata.get("fit_groups")
        if not isinstance(fit_groups, dict):
            raise ValueError("cache fit-group record is missing")
        expected_fit = split.get("fit")
        if fit_groups != expected_fit:
            raise ValueError("cache fit membership differs from frozen fold manifest")
        evaluation_groups = split.get("evaluation", {}).get("valid", [])
        if metadata.get("evaluation_valid_groups") != evaluation_groups:
            raise ValueError("cache evaluation membership differs from frozen fold")
        if (
            len(fit_groups.get("train", [])),
            len(fit_groups.get("valid", [])),
            len(evaluation_groups),
        ) != (
            10535,
            1405,
            468,
        ):
            raise ValueError("cache group counts differ from the frozen H230 contract")

        expected_rows = {
            split_name: _ordered_fit_groups(data_root, split_name, fit_groups[split_name])
            for split_name in ("train", "valid")
        }
        file_manifest = metadata.get("files")
        if not isinstance(file_manifest, dict) or set(file_manifest) != set(ARRAY_WIDTHS):
            raise ValueError("cache file manifest does not list the exact eight outputs")

        for name, (split_name, width) in ARRAY_WIDTHS.items():
            entry = file_manifest[name]
            row_groups = expected_rows[split_name]
            if entry.get("row_groups") != row_groups:
                raise ValueError(f"cache row order differs: {name}")
            if entry.get("row_groups_sha256") != row_groups_sha256(row_groups):
                raise ValueError(f"cache row-order digest differs: {name}")
            shape = (len(row_groups), 20_000, width)
            if entry.get("shape") != list(shape) or entry.get("dtype") != "float32":
                raise ValueError(f"cache declared shape/dtype differs: {name}")
            relative_path = Path(str(entry.get("path", "")))
            if relative_path.is_absolute() or len(relative_path.parts) != 1:
                raise ValueError(f"cache path is not a single relative name: {name}")
            path = cache_dir / relative_path
            receipt["files"][name] = {
                "status": "checking",
                "path": str(path),
                "expected_sha256": entry.get("sha256"),
                "expected_bytes": entry.get("bytes"),
                "row_groups_sha256": entry.get("row_groups_sha256"),
            }
            write_receipt()
            observed = inspect_npy_file(path, shape)
            receipt["files"][name].update({**observed, "status": "verified"})
            write_receipt()
            if observed["sha256"] != entry.get("sha256"):
                raise ValueError(f"cache content digest differs: {name}")
            if observed["bytes"] != entry.get("bytes"):
                raise ValueError(f"cache byte count differs: {name}")

        receipt["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        receipt["status"] = "failed"
        receipt["failure"] = f"{type(error).__name__}: {error}"
        write_receipt()
        raise
    write_receipt()
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse explicit cache verification identities and output receipt path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run verification and return a process status suitable for admission gates."""
    args = parse_args(argv)
    try:
        receipt = verify_cache(
            args.cache_dir,
            args.data_root,
            args.metadata_sha256,
            args.split_sha256,
            args.receipt,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return 1
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(args.receipt),
                "files_verified": len(receipt["files"]),
                "elapsed_seconds": receipt["elapsed_seconds"],
                "max_rss_kib": receipt["max_rss_kib"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
