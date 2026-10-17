"""Replay a fixed geometry ZIP on source-qualified F0 groups and score it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from utils import (  # noqa: E402
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    DataPoint,
    validate_sequence,
)
from wnn.gmetric import GlobalAccumulator  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def filesystem_type(path: Path) -> str:
    """Return the filesystem type and reject RAM-backed scratch."""
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    filesystem = result.stdout.strip().splitlines()[-1].strip()
    if filesystem in {"tmpfs", "ramfs", "devtmpfs"}:
        raise ValueError(f"RAM-backed scratch is not permitted: {filesystem}")
    return filesystem


def extract_package(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract only regular, safe relative ZIP entries."""
    for item in archive.infolist():
        name = PurePosixPath(item.filename)
        if name.is_absolute() or ".." in name.parts or not name.parts or "\\" in item.filename:
            raise ValueError(f"unsafe ZIP entry: {item.filename!r}")
        if item.is_dir():
            continue
        target = destination.joinpath(*name.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(item))


def import_solution(package: Path) -> tuple[ModuleType, Any, Path]:
    """Load the exact callback from the extracted package and pin its native library."""
    if "wnn_native" in sys.modules:
        raise RuntimeError("run one precision per process; wnn_native is already imported")
    os.environ.pop("WNN_NATIVE_LEVEL", None)
    spec = importlib.util.spec_from_file_location(
        "geometry_served_f0_solution", package / "solution.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("ZIP does not contain an importable solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.PredictionModel()
    native = sys.modules.get("wnn_native")
    if native is None:
        raise ImportError("solution.py did not import wnn_native")
    native_path = Path(native.__file__).resolve()
    if native_path.parent.name not in {"v2", "avx2", "v3"}:
        raise RuntimeError(f"unexpected native library path: {native_path}")
    return module, model, native_path


def cpu_quota_fraction() -> float | None:
    """Read the one-core CPU quota supplied to `scripts/limited`."""
    value = os.environ.get("WNN_CPU_QUOTA")
    if value is None or not value.endswith("%"):
        return None
    try:
        fraction = float(value[:-1]) / 100.0
    except ValueError:
        return None
    return fraction if 0.0 < fraction <= 1.0 else None


def load_reference(
    path: Path | None, split_sha256: str, groups: tuple[int, ...]
) -> tuple[Any, dict[int, int]] | None:
    """Load a pinned FP32 prediction cache using its source-qualified sidecar."""
    if path is None:
        return None
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("split_manifest_sha256") != split_sha256:
        raise ValueError("FP32 reference prediction belongs to a different split manifest")
    declared_groups = tuple(map(int, sidecar.get("row_groups", [])))
    if len(set(declared_groups)) != len(declared_groups) or not set(groups) <= set(declared_groups):
        raise ValueError("FP32 reference sidecar lacks one or more F0 evaluation groups")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.dtype(np.float32) or array.shape != (
        len(declared_groups),
        SEQUENCE_LENGTH,
        2,
    ):
        raise ValueError(
            f"unexpected FP32 prediction cache shape/dtype: {array.shape}/{array.dtype}"
        )
    return array, {group: index for index, group in enumerate(declared_groups)}


def write_create_only(path: Path, value: dict[str, Any]) -> None:
    """Write a JSON receipt without replacing prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    """Score every source-qualified F0 group with the exact ZIP callback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument(
        "--scratch", type=Path, default=Path("/home/vlad/development/wunder/research_tmp")
    )
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument(
        "--max-groups",
        type=int,
        help="replay only this many leading split-qualified F0 evaluation groups",
    )
    parser.add_argument("--total-rows", type=int, default=39_400_000)
    args = parser.parse_args()

    submission = args.zip.resolve(strict=True)
    build_path = args.build_receipt.resolve(strict=True)
    repo_root = args.repo_root.resolve(strict=True)
    split_path = args.split.resolve(strict=True)
    predictions_path = args.output_predictions.resolve()
    sidecar_path = args.output_sidecar.resolve()
    receipt_path = args.output_receipt.resolve()
    scratch = args.scratch.resolve(strict=True)
    if any(path.exists() for path in (predictions_path, sidecar_path, receipt_path)):
        raise FileExistsError("refusing to replace served predictions or receipt")
    if args.total_rows <= 0:
        raise ValueError("--total-rows must be positive")
    filesystem = filesystem_type(scratch)
    build = json.loads(build_path.read_text(encoding="utf-8"))
    zip_sha256 = sha256_file(submission)
    if build["artifact"]["sha256"] != zip_sha256:
        raise ValueError("build receipt does not bind the requested ZIP")
    split = load_campaign_split(split_path, repo_root)
    if split.fold != 0:
        raise ValueError(f"expected F0, got fold {split.fold}")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    sequence_by_group = {
        int(item["row_group"]): int(item["seq_ix"]) for item in split_manifest["sequences"]["valid"]
    }
    evaluation_groups = tuple(map(int, split.evaluation_groups))
    if args.max_groups is not None and args.max_groups <= 0:
        raise ValueError("--max-groups must be positive when provided")
    groups = (
        evaluation_groups[: args.max_groups] if args.max_groups is not None else evaluation_groups
    )
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("F0 evaluation row groups must be nonempty and unique")
    reference = load_reference(args.reference_predictions, split.sha256, groups)
    data_path = repo_root / "datasets" / "valid.parquet"
    data_stat = data_path.stat()
    parquet = pq.ParquetFile(data_path)
    if max(groups) >= parquet.num_row_groups:
        raise ValueError("F0 group index exceeds valid.parquet row-group count")

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{predictions_path.name}.", suffix=".npy.tmp", dir=predictions_path.parent
    )
    os.close(fd)
    temporary_predictions = Path(temporary_name)
    temporary_predictions.unlink()
    feature_columns = ["seq_ix", "step_in_seq", "need_prediction", "is_scored", *FEATURE_COLUMNS]
    columns = [*feature_columns, *TARGET_COLUMNS]
    score = GlobalAccumulator()
    reference_score = GlobalAccumulator() if reference is not None else None
    max_abs_difference = 0.0
    absolute_difference_sum = 0.0
    squared_difference_sum = 0.0
    different_values = 0
    compared_values = 0
    callback_wall_ns = 0
    callback_cpu_ns = 0
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    native_path: Path | None = None
    try:
        with zipfile.ZipFile(submission) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
            with tempfile.TemporaryDirectory(prefix="geometry-f0-", dir=scratch) as directory:
                package = Path(directory)
                extract_package(archive, package)
                _module, model, native_path = import_solution(package)
                package_precision = (package / "precision.txt").read_text(encoding="utf-8").strip()
                if package_precision != build["precision"]:
                    raise ValueError("package precision differs from the build receipt")
                level = native_path.parent.name
                expected_library_hash = build["native_runtime"]["libraries"][
                    f"{level}/wnn_native.abi3.so"
                ]
                if sha256_file(native_path) != expected_library_hash:
                    raise ValueError("imported native library differs from build receipt")
                output = np.lib.format.open_memmap(
                    temporary_predictions,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(groups), SEQUENCE_LENGTH, 2),
                )
                group_receipts = []
                for output_index, row_group in enumerate(groups):
                    table = parquet.read_row_group(row_group, columns=columns, use_threads=False)
                    sequence_id, need = validate_sequence(table)
                    if sequence_id != sequence_by_group.get(row_group):
                        raise ValueError(f"row-group identity changed for group {row_group}")
                    features = np.column_stack(
                        [table[name].to_numpy() for name in FEATURE_COLUMNS]
                    ).astype(np.float32, copy=False)
                    if features.shape != (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)):
                        raise ValueError(f"unexpected raw feature shape in group {row_group}")
                    if not np.isfinite(features).all():
                        raise ValueError(f"non-finite raw input in group {row_group}")
                    targets = np.column_stack(
                        [table[name].to_numpy() for name in TARGET_COLUMNS]
                    ).astype(np.float32, copy=False)
                    selected = table["is_scored"].to_numpy().astype(bool) & need
                    values = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
                    reset = getattr(model.model, "reset", None)
                    if not callable(reset):
                        raise TypeError("geometry callback must expose model.model.reset()")
                    reset()
                    model.seq_ix = None
                    callback_cpu_start = time.process_time_ns()
                    for step, row in enumerate(features):
                        point = DataPoint(sequence_id, step, bool(need[step]), row)
                        before = time.perf_counter_ns()
                        value = model.predict(point)
                        callback_wall_ns += time.perf_counter_ns() - before
                        if not need[step]:
                            if value is not None:
                                raise ValueError(
                                    f"warm-up callback returned a value in group {row_group}"
                                )
                            continue
                        if value is None:
                            raise ValueError(
                                f"required prediction is missing in group {row_group}, step {step}"
                            )
                        prediction = np.asarray(value, dtype=np.float32)
                        if prediction.shape != (2,) or not np.isfinite(prediction).all():
                            raise ValueError(
                                f"invalid prediction in group {row_group}, step {step}"
                            )
                        values[step] = prediction
                    callback_cpu_ns += time.process_time_ns() - callback_cpu_start
                    output[output_index] = values
                    score.add(targets[None], values[None], selected[None])
                    if reference is not None and reference_score is not None:
                        reference_array, reference_positions = reference
                        expected = np.asarray(reference_array[reference_positions[row_group]])
                        reference_score.add(targets[None], expected[None], selected[None])
                        difference = np.abs(
                            values[need].astype(np.float64) - expected[need].astype(np.float64)
                        )
                        max_abs_difference = max(
                            max_abs_difference, float(np.max(difference, initial=0.0))
                        )
                        absolute_difference_sum += float(difference.sum())
                        squared_difference_sum += float(np.square(difference).sum())
                        different_values += int(np.count_nonzero(difference))
                        compared_values += int(difference.size)
                    group_receipts.append(
                        {
                            "row_group": row_group,
                            "sequence_id": sequence_id,
                            "rows": SEQUENCE_LENGTH,
                            "need_prediction_rows": int(np.count_nonzero(need)),
                            "scored_rows": int(np.count_nonzero(selected)),
                        }
                    )
                    del table, features, targets, values
                output.flush()
                del output
                os.link(temporary_predictions, predictions_path)
                prediction_sha256 = sha256_file(predictions_path)
                sidecar = {
                    "candidate": build["candidate"],
                    "precision": build["precision"],
                    "split_manifest_sha256": split.sha256,
                    "row_groups": list(groups),
                    "sequence_ids": [int(sequence_by_group[group]) for group in groups],
                    "prediction_sha256": prediction_sha256,
                    "zip_sha256": zip_sha256,
                    "scope": "source-qualified exposed F0 callback replay; labels used only for official metric",
                }
                write_create_only(sidecar_path, sidecar)
                native_relative_path = native_path.relative_to(package).as_posix()
        temporary_predictions.unlink(missing_ok=True)
    except BaseException:
        temporary_predictions.unlink(missing_ok=True)
        predictions_path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        raise

    after = data_path.stat()
    if (data_stat.st_dev, data_stat.st_ino, data_stat.st_size, data_stat.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("valid.parquet identity changed during F0 replay")
    elapsed_wall = time.perf_counter() - started_wall
    elapsed_cpu = time.process_time() - started_cpu
    callback_seconds = callback_wall_ns / 1e9
    callback_cpu_seconds = callback_cpu_ns / 1e9
    quota_fraction = cpu_quota_fraction()
    rows = len(groups) * SEQUENCE_LENGTH
    callback_seconds_per_row = callback_seconds / rows
    report = {
        "schema": "wnn.geometry-served-f0.v1",
        "passed": True,
        "candidate": build["candidate"],
        "precision": build["precision"],
        "metric_id": score.result()["metric_id"],
        "official_f0_score": score.result(),
        "fp32_reference_f0_score": reference_score.result()
        if reference_score is not None
        else None,
        "served_minus_fp32_f0_wp": (
            float(score.result()["wp"] - reference_score.result()["wp"])
            if reference_score is not None
            else None
        ),
        "reference_prediction_difference": (
            {
                "max_abs": max_abs_difference,
                "mean_abs": absolute_difference_sum / compared_values if compared_values else 0.0,
                "rms": math.sqrt(squared_difference_sum / compared_values)
                if compared_values
                else 0.0,
                "different_values": different_values,
                "compared_values": compared_values,
            }
            if reference is not None
            else None
        ),
        "groups": group_receipts,
        "row_group_order": "CampaignSplit.evaluation_groups; each group bound to split sequence identity",
        "rows_total": rows,
        "evaluation_groups_total": len(evaluation_groups),
        "evaluation_groups_replayed": len(groups),
        "scored_rows": int(score.selected_rows),
        "zip_path": str(submission),
        "zip_sha256": zip_sha256,
        "build_receipt_path": str(build_path),
        "build_receipt_sha256": sha256_file(build_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": prediction_sha256,
        "prediction_sidecar": str(sidecar_path),
        "native_library": native_relative_path,
        "native_library_sha256": expected_library_hash,
        "split_path": str(split_path),
        "split_sha256": split.sha256,
        "data_path": str(data_path),
        "data_identity": {
            "bytes": data_stat.st_size,
            "mtime_ns": data_stat.st_mtime_ns,
            "footer_sha256_from_split": split_manifest["sources"]["datasets/valid.parquet"][
                "footer_sha256"
            ],
        },
        "filesystem": filesystem,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "memory_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "timing": {
            "elapsed_wall_seconds_including_parquet_and_score": elapsed_wall,
            "elapsed_process_cpu_seconds_including_parquet_and_score": elapsed_cpu,
            "callback_only_seconds": callback_seconds,
            "callback_only_wall_us_per_row": callback_seconds_per_row * 1e6,
            "callback_only_cpu_seconds": callback_cpu_seconds,
            "callback_only_cpu_seconds_per_row": callback_cpu_seconds / rows,
            "cpu_quota_fraction": quota_fraction,
            "rows_per_second_callback_only": rows / callback_seconds,
            "full_dataset_rows_for_projection": args.total_rows,
            "callback_only_same_quota_full_dataset_seconds_proxy": callback_seconds_per_row
            * args.total_rows,
            "callback_only_one_core_full_dataset_seconds_proxy": (
                callback_seconds_per_row * quota_fraction * args.total_rows
                if quota_fraction is not None
                else None
            ),
            "projection_label": "measured actual F0 row-cost extrapolations; complete competition runtime not measured",
        },
        "scope": (
            "source-qualified exposed F0 panel plus exact ZIP callback and reset handling; panel score is descriptive only"
            if len(groups) != len(evaluation_groups)
            else "official exposed F0 quality plus exact ZIP callback and source/reset handling; no public score forecast"
        ),
    }
    write_create_only(receipt_path, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
