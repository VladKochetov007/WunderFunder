#!/usr/bin/env python3
"""Compare exact ZIP callback outputs with a pinned native prediction vector.

The checker reads only the first eight ordered F0 evaluation row groups, one
complete group at a time. It checks every callback row, resets between groups,
and replays the first group around the second (A-B-A). It computes no metric.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import mmap
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __package__:
    from .validate_submission_callback_panel import safe_extract, sha256_file
else:
    from validate_submission_callback_panel import safe_extract, sha256_file

from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, DataPoint, validate_sequence  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402

PANEL_SIZE = 8
MAX_ZIP_MEMBERS = 256
MAX_ZIP_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ZIP_UNPACKED_BYTES = 256 * 1024 * 1024
HASH_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


def validate_prediction(value: Any, *, need_prediction: bool, row: int) -> np.ndarray | None:
    """Enforce the competition callback contract without dtype coercion."""
    if not need_prediction:
        if value is not None:
            raise ValueError(f"warmup row {row} must return None")
        return None
    if not isinstance(value, np.ndarray):
        raise TypeError(f"required row {row} must return a NumPy array")
    if value.dtype != np.dtype(np.float32):
        raise TypeError(f"required row {row} dtype must be float32, got {value.dtype}")
    if value.shape != (2,):
        raise ValueError(f"required row {row} shape must be (2,), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"required row {row} contains a non-finite prediction")
    return value


def compare_required_rows(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    """Report exact float32 bit differences, including signed-zero changes."""
    if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
        raise TypeError("actual and expected predictions must be NumPy arrays")
    if actual.dtype != np.dtype(np.float32) or expected.dtype != np.dtype(np.float32):
        raise TypeError("actual and expected predictions must both be float32")
    if actual.ndim != 2 or actual.shape[1:] != (2,) or actual.shape != expected.shape:
        raise ValueError("actual and expected predictions must have equal shape (rows, 2)")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("actual and expected required predictions must be finite")

    actual_contiguous = np.ascontiguousarray(actual)
    expected_contiguous = np.ascontiguousarray(expected)
    differing_values = actual_contiguous.view(np.uint32) != expected_contiguous.view(np.uint32)
    differing_rows = np.any(differing_values, axis=1)
    difference = np.abs(
        actual_contiguous.astype(np.float64) - expected_contiguous.astype(np.float64)
    )
    return {
        "exact_bitwise": not bool(np.any(differing_values)),
        "different_rows": int(np.count_nonzero(differing_rows)),
        "different_values": int(np.count_nonzero(differing_values)),
        "max_abs_difference": float(np.max(difference)) if difference.size else 0.0,
        "actual_sha256": hashlib.sha256(actual_contiguous.tobytes(order="C")).hexdigest(),
        "expected_sha256": hashlib.sha256(expected_contiguous.tobytes(order="C")).hexdigest(),
    }


def same_float32_bits(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Compare equal-shaped float32 arrays without treating signed zero as equal."""
    if actual.dtype != np.dtype(np.float32) or expected.dtype != np.dtype(np.float32):
        raise TypeError("A-B-A replay arrays must both be float32")
    if actual.shape != expected.shape:
        return False
    return bool(
        np.array_equal(
            np.ascontiguousarray(actual).view(np.uint32),
            np.ascontiguousarray(expected).view(np.uint32),
        )
    )


def validate_archive(archive: zipfile.ZipFile) -> None:
    """Reject unsafe, special-file, duplicate, or oversized ZIP entries."""
    members = archive.infolist()
    if not members or len(members) > MAX_ZIP_MEMBERS:
        raise ValueError(f"ZIP must contain 1..{MAX_ZIP_MEMBERS} members")

    names: set[str] = set()
    unpacked_bytes = 0
    for item in members:
        name = PurePosixPath(item.filename)
        if (
            not item.filename
            or "\\" in item.filename
            or name.is_absolute()
            or ".." in name.parts
            or not name.parts
        ):
            raise ValueError(f"unsafe ZIP entry path: {item.filename!r}")
        normalized = name.as_posix().rstrip("/")
        if normalized in names:
            raise ValueError(f"duplicate ZIP entry path: {item.filename!r}")
        names.add(normalized)

        mode = item.external_attr >> 16
        kind = stat.S_IFMT(mode)
        allowed_kinds = {0, stat.S_IFREG, stat.S_IFDIR}
        if kind not in allowed_kinds:
            raise ValueError(f"ZIP entry is not a regular file or directory: {item.filename!r}")
        if item.is_dir():
            continue
        if item.file_size > MAX_ZIP_MEMBER_BYTES:
            raise ValueError(f"ZIP member exceeds size limit: {item.filename!r}")
        unpacked_bytes += item.file_size
        if unpacked_bytes > MAX_ZIP_UNPACKED_BYTES:
            raise ValueError("ZIP unpacked size exceeds limit")

    if "solution.py" not in names:
        raise ValueError("submission ZIP does not contain solution.py")


def expected_sha256(value: str, label: str) -> str:
    """Validate and normalize a caller-supplied byte hash pin."""
    if not HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return value.lower()


def verify_hash(path: Path, expected: str, label: str) -> str:
    """Hash a file in bounded memory and require its declared identity."""
    digest = sha256_file(path)
    if digest != expected:
        raise ValueError(f"{label} SHA-256 mismatch: got {digest}, expected {expected}")
    return digest


def physical_filesystem_type(directory: Path) -> str:
    """Identify and reject RAM-backed extraction locations."""
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(directory)],
        check=True,
        capture_output=True,
        text=True,
    )
    filesystem = result.stdout.strip().splitlines()[-1].strip()
    if filesystem in {"tmpfs", "ramfs", "devtmpfs"}:
        raise ValueError(f"refusing RAM-backed ZIP extraction filesystem: {filesystem}")
    return filesystem


def advise_parquet(fd: int, advice_name: str) -> None:
    """Apply an advisory cache hint to the data file for each row-group pass."""
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("POSIX fadvise is required for this bounded reader")
    advice = getattr(os, advice_name, None)
    if advice is None:
        raise RuntimeError(f"platform does not expose {advice_name}")
    os.posix_fadvise(fd, 0, 0, advice)


def load_row_group(
    parquet: pq.ParquetFile,
    fd: int,
    row_group: int,
    expected_sequence_id: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Read and validate only features for one complete sequence group."""
    advise_parquet(fd, "POSIX_FADV_SEQUENTIAL")
    table = parquet.read_row_group(
        row_group,
        columns=["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS],
        use_threads=False,
    )
    sequence_id, need = validate_sequence(table)
    if sequence_id != expected_sequence_id:
        raise ValueError(
            f"row group {row_group} has sequence ID {sequence_id}, "
            f"split manifest says {expected_sequence_id}"
        )
    features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(
        np.float32, copy=False
    )
    if features.shape != (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)):
        raise ValueError(f"unexpected feature matrix shape for row group {row_group}: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite feature in row group {row_group}")
    del table
    return sequence_id, features, need


def run_group(
    model: Any,
    parquet: pq.ParquetFile,
    fd: int,
    reference: np.ndarray,
    panel_index: int,
    row_group: int,
    sequence_id: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Reset and replay one full group, comparing each required callback row."""
    reset = getattr(getattr(model, "model", None), "reset", None)
    if not callable(reset):
        raise TypeError("submission PredictionModel.model.reset() is required for group isolation")

    features: np.ndarray | None = None
    need: np.ndarray | None = None
    try:
        actual_sequence, features, need = load_row_group(
            parquet, fd, row_group, sequence_id
        )
        reset()
        actual = np.empty((int(np.count_nonzero(need)), 2), dtype=np.float32)
        write_index = 0
        for step, row in enumerate(features):
            result = model.predict(DataPoint(actual_sequence, step, bool(need[step]), row))
            checked = validate_prediction(
                result, need_prediction=bool(need[step]), row=step
            )
            if checked is not None:
                actual[write_index] = checked
                write_index += 1
        if write_index != len(actual):
            raise RuntimeError("callback output count differs from the required-row mask")

        expected = reference[panel_index, :, :][need]
        comparison = compare_required_rows(actual, expected)
        report = {
            "row_group": int(row_group),
            "sequence_id": int(actual_sequence),
            "reference_panel_index": int(panel_index),
            "required_rows": int(len(actual)),
            **comparison,
        }
        return report, actual
    finally:
        if features is not None:
            del features
        if need is not None:
            del need
        advise_parquet(fd, "POSIX_FADV_DONTNEED")


def load_sequence_map(split_path: Path) -> dict[int, int]:
    """Bind row-group ordering to the split's frozen sequence identities."""
    manifest = json.loads(split_path.read_text())
    sequences = manifest["sequences"]["valid"]
    result = {int(node["row_group"]): int(node["seq_ix"]) for node in sequences}
    if len(result) != len(sequences):
        raise ValueError("duplicate valid row-group identity in the split manifest")
    return result


def load_submission(package: Path, level: str) -> tuple[Any, Any, str]:
    """Import one package and verify that its selected native level was loaded."""
    if "wnn_native" in sys.modules:
        raise RuntimeError("wnn_native is already imported; invoke each ISA level in a fresh process")
    os.environ["WNN_NATIVE_LEVEL"] = level
    module = import_solution_module(package)
    constructor = getattr(module, "PredictionModel", None)
    if not callable(constructor):
        raise TypeError("submission solution.py lacks PredictionModel")
    model = constructor()

    native_module = sys.modules.get("wnn_native")
    if native_module is None:
        raise ImportError("submission did not import its wnn_native extension")
    native_path = Path(native_module.__file__).resolve()
    expected_parent = (package / level).resolve()
    if native_path.parent != expected_parent:
        raise RuntimeError(f"loaded native library from {native_path}, expected {expected_parent}")
    if native_path.name != "wnn_native.abi3.so":
        raise RuntimeError(f"submission loaded an unexpected native ABI member: {native_path.name}")
    if not callable(getattr(model, "predict", None)):
        raise TypeError("submission PredictionModel.predict is not callable")
    return model, native_path, str(getattr(getattr(model, "model", None), "precision", "unknown"))


def import_solution_module(package: Path) -> Any:
    """Import solution.py with its extracted directory available for siblings.

    Python caches top-level imports such as ``qat_archive`` in ``sys.modules``;
    only the temporary search-path entry is removed after module execution.
    """
    solution = package / "solution.py"
    helper_path = package / "qat_archive.py"
    cached_helper = sys.modules.get("qat_archive")
    if helper_path.is_file() and cached_helper is not None:
        cached_path = getattr(cached_helper, "__file__", None)
        if cached_path is None or Path(cached_path).resolve() != helper_path.resolve():
            raise RuntimeError("qat_archive is already cached from a different package")

    package_root = str(package.resolve())
    original_root_entries = sys.path.count(package_root)
    added_to_path = package_root not in sys.path
    if added_to_path:
        sys.path.insert(0, package_root)
    try:
        spec = importlib.util.spec_from_file_location("checked_zip_submission_solution", solution)
        if spec is None or spec.loader is None:
            raise ImportError("submission ZIP has no importable solution.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except BaseException:
        if cached_helper is None:
            sys.modules.pop("qat_archive", None)
        raise
    finally:
        while sys.path.count(package_root) > original_root_entries:
            sys.path.remove(package_root)

    imported_helper = sys.modules.get("qat_archive")
    if helper_path.is_file() and imported_helper is not None:
        imported_path = getattr(imported_helper, "__file__", None)
        if imported_path is None or Path(imported_path).resolve() != helper_path.resolve():
            raise RuntimeError("submission imported qat_archive from outside its extracted package")
    return module


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write an immutable JSON receipt atomically."""
    if path.exists():
        raise FileExistsError(f"refusing to replace parity receipt: {path}")
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to replace parity receipt: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    """Verify pinned ZIP-to-vector equivalence for eight F0 evaluation groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="descriptive immutable case label")
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="native blend .npy vector")
    parser.add_argument("--split", type=Path, required=True, help="frozen F0 campaign split")
    parser.add_argument("--data", type=Path, required=True, help="exact valid.parquet bound by split")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", choices=("v2", "avx2", "v3"), required=True)
    parser.add_argument(
        "--require-python311",
        action="store_true",
        help="require the current interpreter to be Python 3.11 for an abi3-py311 import check",
    )
    parser.add_argument("--expected-zip-sha256", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    args = parser.parse_args()

    if args.require_python311 and sys.version_info[:2] != (3, 11):
        raise RuntimeError("--require-python311 needs an actual Python 3.11 interpreter")
    if args.require_python311 and (
        os.environ.get("LD_LIBRARY_PATH") or os.environ.get("PYTHONPATH")
    ):
        raise RuntimeError("Python 3.11 import check requires clean LD_LIBRARY_PATH and PYTHONPATH")

    submission = args.zip.resolve(strict=True)
    vector_path = args.reference.resolve(strict=True)
    split_path = args.split.resolve(strict=True)
    data_path = args.data.resolve(strict=True)
    repo_root = args.repo_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace parity receipt: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"parity receipt directory does not exist: {output.parent}")
    if data_path != (repo_root / "datasets/valid.parquet").resolve(strict=True):
        raise ValueError("--data must be the exact datasets/valid.parquet source under --repo-root")
    if submission.stat().st_size > MAX_ZIP_UNPACKED_BYTES:
        raise ValueError("submission ZIP exceeds the archive-size limit")

    expected_hashes = {
        "zip": expected_sha256(args.expected_zip_sha256, "ZIP hash pin"),
        "reference": expected_sha256(args.expected_reference_sha256, "reference hash pin"),
        "split": expected_sha256(args.expected_split_sha256, "split hash pin"),
        "data": expected_sha256(args.expected_data_sha256, "data hash pin"),
    }
    actual_hashes = {
        "zip": verify_hash(submission, expected_hashes["zip"], "ZIP"),
        "reference": verify_hash(vector_path, expected_hashes["reference"], "reference vector"),
        "split": verify_hash(split_path, expected_hashes["split"], "campaign split"),
        "data": verify_hash(data_path, expected_hashes["data"], "validation data"),
    }

    split = load_campaign_split(split_path, repo_root)
    if split.fold != 0:
        raise ValueError(f"this checker is scoped to fold 0, got fold {split.fold}")
    if split.sha256 != actual_hashes["split"]:
        raise ValueError("split hash changed between pin verification and manifest loading")
    ordered_groups = tuple(map(int, split.evaluation_groups[:PANEL_SIZE]))
    if len(ordered_groups) != PANEL_SIZE:
        raise ValueError(f"F0 evaluation split has fewer than {PANEL_SIZE} groups")
    sequence_by_group = load_sequence_map(split_path)
    if any(group not in sequence_by_group for group in ordered_groups):
        raise ValueError("ordered F0 evaluation groups are absent from valid sequence identities")

    reference = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(reference, np.memmap):
        raise ValueError("reference must be a memory-mappable .npy array")
    if reference.dtype != np.dtype(np.float32):
        raise TypeError(f"reference must be float32, got {reference.dtype}")
    if reference.shape != (len(split.evaluation_groups), SEQUENCE_LENGTH, 2):
        raise ValueError(
            "reference shape must be "
            f"({len(split.evaluation_groups)}, {SEQUENCE_LENGTH}, 2), got {reference.shape}"
        )
    if hasattr(reference, "_mmap") and hasattr(reference._mmap, "madvise"):
        reference._mmap.madvise(mmap.MADV_SEQUENTIAL)

    scratch = output.parent.resolve(strict=True)
    filesystem = physical_filesystem_type(scratch)
    parquet = pq.ParquetFile(data_path)
    if max(ordered_groups) >= parquet.num_row_groups:
        raise ValueError("split evaluation group index exceeds validation parquet row-group count")

    data_stat = data_path.stat()
    group_results: list[dict[str, Any]] = []
    replay_result: dict[str, Any] | None = None
    replay_matches_first_a = False
    first_a_predictions: np.ndarray | None = None
    native_relative_path: str | None = None
    native_sha256: str | None = None
    precision = "unknown"
    advise_fd = os.open(data_path, os.O_RDONLY)
    try:
        with zipfile.ZipFile(submission) as archive:
            validate_archive(archive)
            with tempfile.TemporaryDirectory(prefix="wnn-zip-parity-", dir=scratch) as temporary:
                package = Path(temporary)
                safe_extract(archive, package)
                model, native_path, precision = load_submission(package, args.level)
                native_relative_path = native_path.relative_to(package).as_posix()
                native_sha256 = sha256_file(native_path)

                def run(index: int) -> tuple[dict[str, Any], np.ndarray]:
                    group = ordered_groups[index]
                    return run_group(
                        model,
                        parquet,
                        advise_fd,
                        reference,
                        index,
                        group,
                        sequence_by_group[group],
                    )

                for panel_index in range(PANEL_SIZE):
                    group_result, predictions = run(panel_index)
                    group_results.append(group_result)
                    if panel_index == 0:
                        first_a_predictions = predictions.copy()
                    del predictions
                    if panel_index == 1:
                        replay_result, replay_predictions = run(0)
                        if first_a_predictions is None:
                            raise RuntimeError("A-B-A replay ran before its first A sequence")
                        replay_matches_first_a = same_float32_bits(
                            first_a_predictions, replay_predictions
                        )
                        del replay_predictions
    finally:
        os.close(advise_fd)

    data_after = data_path.stat()
    stable_identity_before = (
        data_stat.st_dev,
        data_stat.st_ino,
        data_stat.st_size,
        data_stat.st_mtime_ns,
    )
    stable_identity_after = (
        data_after.st_dev,
        data_after.st_ino,
        data_after.st_size,
        data_after.st_mtime_ns,
    )
    if stable_identity_before != stable_identity_after:
        raise RuntimeError("validation data file identity changed during ZIP parity checking")
    all_comparisons = [*group_results]
    if replay_result is not None:
        all_comparisons.append(replay_result)
    total_rows = sum(int(item["required_rows"]) for item in all_comparisons)
    total_different_rows = sum(int(item["different_rows"]) for item in all_comparisons)
    total_different_values = sum(int(item["different_values"]) for item in all_comparisons)
    max_abs = max(float(item["max_abs_difference"]) for item in all_comparisons)
    exact = all(bool(item["exact_bitwise"]) for item in all_comparisons)
    replay_matches = bool(replay_result and replay_matches_first_a)
    report = {
        "version": 1,
        "label": args.label,
        "passed": exact and replay_matches,
        "exact_bitwise_match": exact,
        "a_b_a_first_sequence_replay_exact": replay_matches,
        "a_b_a_first_and_replay_sha256_equal": bool(
            replay_result
            and group_results[0]["actual_sha256"] == replay_result["actual_sha256"]
        ),
        "a_b_a_replay_matches_reference": bool(replay_result and replay_result["exact_bitwise"]),
        "a_b_a_group_order": [ordered_groups[0], ordered_groups[1], ordered_groups[0]],
        "native_level": args.level,
        "native_relative_path": native_relative_path,
        "native_sha256": native_sha256,
        "runtime_precision": precision,
        "python": sys.version,
        "python311_required": bool(args.require_python311),
        "external_python311_import": bool(args.require_python311),
        "platform": platform.platform(),
        "reference_shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "ordered_evaluation_row_groups": list(ordered_groups),
        "groups": group_results,
        "a_b_a_replay": replay_result,
        "required_rows_compared_including_replay": total_rows,
        "different_rows_including_replay": total_different_rows,
        "different_values_including_replay": total_different_values,
        "max_abs_difference_including_replay": max_abs,
        "warmup": "each callback was required to return None; numeric comparison covers required rows only",
        "hashes": {
            "zip_sha256": actual_hashes["zip"],
            "reference_sha256": actual_hashes["reference"],
            "split_sha256": actual_hashes["split"],
            "data_sha256": actual_hashes["data"],
            "expected_sha256": expected_hashes,
        },
        "split_fold": split.fold,
        "split_panel_order": "CampaignSplit.evaluation_groups[:8], without sorting",
        "reader": {
            "data_row_groups_read": PANEL_SIZE + 1,
            "labels_read": False,
            "row_group_at_a_time": True,
            "reference_memory_mapped": True,
            "fadvise_each_group_pass": True,
            "extraction_filesystem": filesystem,
            "extraction_location": "temporary directory under output parent, removed after import/replay",
        },
        "scope": "ZIP callback/import/parity/reset evidence only; no metric, quality, runtime, or public-score approval",
    }
    write_report(output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
