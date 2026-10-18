"""Replay a packaged native submission on a small, complete validation panel.

The checker imports the delivered ZIP itself, invokes every callback row,
validates warm-up behavior and reports the official pooled clipped weighted
Pearson score. It is a CPU panel check, not a full-dataset runtime or quality
claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow.parquet as pq

SEQUENCE_LENGTH = 20_000
WARMUP = 99
N_FEATURES = 112
CLIP = 2.0
EPSILON = 1e-8
ETA_ROWS = 39_400_000
MAX_SEQUENCES = 8
MAX_UNPACKED_BYTES = 512 * 1024 * 1024
FEATURE_COLUMNS = [
    f"i{i}_{group}{j}"
    for i in range(2)
    for group, count in [("p", 22), ("v", 22), ("dp", 4), ("dv", 4)]
    for j in range(count)
] + [f"a{i}" for i in range(8)]
TARGET_COLUMNS = ("t0", "t1")
PANEL_COLUMNS = [
    "seq_ix",
    "step_in_seq",
    "need_prediction",
    "is_scored",
    *FEATURE_COLUMNS,
    *TARGET_COLUMNS,
]


@dataclasses.dataclass(frozen=True)
class DataPoint:
    """Minimal standalone callback input matching the competition API."""

    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def sha256_file(path: Path) -> str:
    """Hash a file incrementally to keep memory bounded."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def official_wp(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | str]:
    """Compute pooled clipped weighted Pearson independently for both targets."""
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    if y.ndim != 3 or y.shape[-1] != 2 or p.shape != y.shape or selected.shape != y.shape[:2]:
        raise ValueError("expected (sequences, rows, 2) targets/predictions and row mask")
    if not selected.any():
        raise ValueError("panel has no rows selected for scoring")
    if not np.isfinite(y[selected]).all() or not np.isfinite(p[selected]).all():
        raise ValueError("selected target or prediction is nonfinite")
    per_target: list[float] = []
    for target_index in range(2):
        yt = np.clip(y[..., target_index][selected], -CLIP, CLIP).astype(np.float64)
        pt = np.clip(p[..., target_index][selected], -CLIP, CLIP).astype(np.float64)
        weight = np.abs(yt)
        total = float(weight.sum())
        if total < EPSILON:
            per_target.append(0.0)
            continue
        y_mean = float(np.sum(weight * yt) / total)
        p_mean = float(np.sum(weight * pt) / total)
        centered_y = yt - y_mean
        centered_p = pt - p_mean
        covariance = float(np.sum(weight * centered_y * centered_p) / total)
        y_std = float(np.sqrt(np.sum(weight * centered_y**2) / total))
        p_std = float(np.sqrt(np.sum(weight * centered_p**2) / total))
        score = 0.0 if y_std <= EPSILON or p_std <= EPSILON else covariance / (y_std * p_std)
        per_target.append(float(np.clip(score, -1.0, 1.0)))
    return {
        "metric_id": "global_wp_clip2_v1",
        "t0": per_target[0],
        "t1": per_target[1],
        "wp": 0.5 * (per_target[0] + per_target[1]),
        "selected_rows": int(selected.sum()),
    }


class OfficialWPAccumulator:
    """Stream pooled official metric moments across complete sequences."""

    def __init__(self) -> None:
        self.moments = np.zeros((2, 6), dtype=np.float64)
        self.selected_rows = 0

    def add(self, targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> None:
        """Merge one sequence's centered weighted moments into the global score."""
        y = np.asarray(targets, dtype=np.float32)
        p = np.asarray(predictions, dtype=np.float32)
        selected = np.asarray(mask, dtype=bool)
        if (
            y.shape != (SEQUENCE_LENGTH, 2)
            or p.shape != y.shape
            or selected.shape != (SEQUENCE_LENGTH,)
        ):
            raise ValueError("metric input does not match one complete sequence")
        if not np.isfinite(y[selected]).all() or not np.isfinite(p[selected]).all():
            raise ValueError("selected target or prediction is nonfinite")
        self.selected_rows += int(selected.sum())
        for target_index in range(2):
            yt = np.clip(y[selected, target_index], -CLIP, CLIP).astype(np.float64)
            pt = np.clip(p[selected, target_index], -CLIP, CLIP).astype(np.float64)
            weight = np.abs(yt)
            keep = weight > 0
            if not keep.any():
                continue
            yt, pt, weight = yt[keep], pt[keep], weight[keep]
            total = float(weight.sum())
            yshift, pshift = yt - yt[0], pt - pt[0]
            ymean = float(np.sum(weight * yshift) / total)
            pmean = float(np.sum(weight * pshift) / total)
            centered = np.array(
                [
                    total,
                    float(yt[0] + ymean),
                    float(pt[0] + pmean),
                    float(np.sum(weight * (yshift - ymean) ** 2)),
                    float(np.sum(weight * (pshift - pmean) ** 2)),
                    float(np.sum(weight * (yshift - ymean) * (pshift - pmean))),
                ],
                dtype=np.float64,
            )
            previous = self.moments[target_index]
            if previous[0] == 0:
                self.moments[target_index] = centered
                continue
            combined = previous[0] + total
            ratio = total / combined
            dy, dp = centered[1] - previous[1], centered[2] - previous[2]
            cross_weight = previous[0] * ratio
            self.moments[target_index] = [
                combined,
                previous[1] + dy * ratio,
                previous[2] + dp * ratio,
                previous[3] + centered[3] + dy * dy * cross_weight,
                previous[4] + centered[4] + dp * dp * cross_weight,
                previous[5] + centered[5] + dy * dp * cross_weight,
            ]

    def result(self) -> dict[str, float | int | str]:
        """Return both target correlations after pooling every selected row."""
        if self.selected_rows == 0:
            raise ValueError("panel has no rows selected for scoring")
        scores: list[float] = []
        for total, _mean_y, _mean_p, variance_y, variance_p, covariance in self.moments:
            std_y, std_p = np.sqrt(np.maximum([variance_y, variance_p], 0.0) / max(total, 1.0))
            score = (
                0.0
                if total < EPSILON or std_y <= EPSILON or std_p <= EPSILON
                else covariance / total / (std_y * std_p)
            )
            scores.append(float(np.clip(score, -1.0, 1.0)))
        return {
            "metric_id": "global_wp_clip2_v1",
            "t0": scores[0],
            "t1": scores[1],
            "wp": 0.5 * (scores[0] + scores[1]),
            "selected_rows": self.selected_rows,
        }


def validate_group(parquet: pq.ParquetFile, group: int) -> int:
    """Validate one selected complete row group and return its sequence ID."""
    table = parquet.read_row_group(
        group, columns=["seq_ix", "step_in_seq", "need_prediction"], use_threads=False
    )
    if table.num_rows != SEQUENCE_LENGTH:
        raise ValueError(f"row group {group} is not a complete 20,000-row sequence")
    seq_ids = table["seq_ix"].to_numpy()
    steps = table["step_in_seq"].to_numpy()
    needs = table["need_prediction"].to_numpy().astype(bool)
    if not np.all(seq_ids == seq_ids[0]) or not np.array_equal(steps, np.arange(SEQUENCE_LENGTH)):
        raise ValueError(f"row group {group} has mixed IDs or noncanonical row order")
    if not np.array_equal(needs, np.arange(SEQUENCE_LENGTH) >= WARMUP):
        raise ValueError(f"row group {group} violates the first-99-row warm-up contract")
    return int(seq_ids[0])


def select_groups(
    data_path: Path, fold_path: Path | None, max_groups: int | None
) -> tuple[pq.ParquetFile, list[int], list[int], int]:
    """Select a bounded panel or the fold's ordered evaluation-valid groups."""
    parquet = pq.ParquetFile(data_path)
    if parquet.schema_arrow.names != PANEL_COLUMNS:
        raise ValueError(
            "parquet schema or feature-column order does not match the official contract"
        )
    if fold_path is None:
        if parquet.num_row_groups > MAX_SEQUENCES:
            raise ValueError("standalone panel mode accepts at most eight row groups")
        source_groups = list(range(parquet.num_row_groups))
    else:
        fold = json.loads(fold_path.read_text(encoding="utf-8"))
        evaluation = fold.get("evaluation")
        valid = evaluation.get("valid") if isinstance(evaluation, dict) else None
        if not isinstance(valid, list) or not 1 <= len(valid) <= 468:
            raise ValueError("fold JSON must contain 1..468 ordered evaluation.valid row groups")
        source_groups = [int(value) for value in valid]
        if (
            len(set(source_groups)) != len(source_groups)
            or min(source_groups) < 0
            or max(source_groups) >= parquet.num_row_groups
        ):
            raise ValueError("fold evaluation-valid groups are duplicated or outside the parquet")
    count = len(source_groups) if max_groups is None else max_groups
    if count < 1 or count > len(source_groups):
        raise ValueError("--max-groups must be between one and the available selected groups")
    selected_groups = source_groups[:count]
    sequence_ids = [validate_group(parquet, group) for group in selected_groups]
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("selected panel contains duplicate sequence IDs")
    return parquet, selected_groups, sequence_ids, len(source_groups)


def extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract regular ZIP members safely and enforce an unpacked-size bound."""
    total = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        if "solution.py" not in names:
            raise ValueError("ZIP must contain solution.py at its root")
        for member in members:
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or (mode & 0o170000) == 0o120000:
                raise ValueError(f"unsafe ZIP member: {member.filename}")
            total += member.file_size
            if total > MAX_UNPACKED_BYTES:
                raise ValueError("ZIP exceeds the 512 MiB unpacked-size limit")
            target = destination.joinpath(*path.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def import_submission(
    package: Path, level: str | None
) -> tuple[ModuleType, Any, Path | None, str | None]:
    """Import the actual extracted callback and report its loaded native library."""
    if "solution" in sys.modules or "wnn_native" in sys.modules:
        raise RuntimeError(
            "checker must run in a fresh process with no cached solution/native module"
        )
    if level is not None:
        os.environ["WNN_NATIVE_LEVEL"] = level
    sys.path.insert(0, str(package))
    try:
        spec = importlib.util.spec_from_file_location("solution", package / "solution.py")
        if spec is None or spec.loader is None:
            raise ImportError("could not load ZIP solution.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["solution"] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "PredictionModel"):
            raise TypeError("solution.py does not define PredictionModel")
        model = module.PredictionModel()
        native = sys.modules.get("wnn_native")
        native_path = (
            Path(native.__file__).resolve()
            if native is not None and getattr(native, "__file__", None)
            else None
        )
        native_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
        if (
            native_path is None
            or not native_path.is_relative_to(package.resolve())
            or not str(native_path).endswith(native_suffixes)
        ):
            raise RuntimeError("callback did not load wnn_native from the delivered ZIP")
        observed_level = getattr(module, "_LEVEL", None)
        if level is not None and observed_level not in (None, level):
            raise RuntimeError(f"requested native level {level}, package selected {observed_level}")
        return module, model, native_path, observed_level
    except BaseException:
        sys.path.remove(str(package))
        sys.modules.pop("solution", None)
        sys.modules.pop("wnn_native", None)
        raise


def callback_prediction(
    model: Any, sequence: int, step: int, need: bool, features: np.ndarray
) -> np.ndarray | None:
    """Call one row and validate requested-output and warm-up return behavior."""
    value = model.predict(DataPoint(sequence, step, need, features))
    if not need:
        if value is not None:
            raise ValueError("warm-up callback rows must update state and return None")
        return None
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("each required callback row must return two finite values")
    return result


def main() -> None:
    """Run the package on a panel or ordered fold and write evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--fold", type=Path, help="optional fold JSON; select evaluation.valid from full parquet"
    )
    parser.add_argument("--max-groups", type=int, help="run only the first N selected groups")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--level", choices=("v2", "avx2", "v3"))
    args = parser.parse_args()
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "run this native panel checker with the Linux Python 3.11 submission runtime"
        )
    if args.output.exists() or args.predictions.exists():
        raise FileExistsError("output and predictions paths must not already exist")
    if args.predictions.suffix != ".npy":
        raise ValueError("--predictions must be a .npy path")
    if args.reference is not None and not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if not args.zip_path.is_file() or not args.panel.is_file():
        raise FileNotFoundError("ZIP and panel/full validation parquet must exist")
    if args.fold is not None and not args.fold.is_file():
        raise FileNotFoundError(args.fold)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parquet, source_groups, sequence_ids, available_groups = select_groups(
        args.panel, args.fold, args.max_groups
    )
    expected_shape = (len(sequence_ids), SEQUENCE_LENGTH, 2)
    reference = None
    if args.reference is not None:
        reference = np.load(args.reference, mmap_mode="r", allow_pickle=False)
        allowed_shapes = {
            expected_shape,
            (available_groups, SEQUENCE_LENGTH, 2),
        }
        if reference.shape not in allowed_shapes or reference.dtype != np.float32:
            raise ValueError(
                f"reference must be float32 with selected or full-fold shape; selected={expected_shape}"
            )

    started_all = time.perf_counter()
    source_stat_before = args.panel.stat()
    full_source_selected = args.fold is None or len(sequence_ids) == available_groups
    panel_sha = sha256_file(args.panel) if full_source_selected else None
    zip_sha = sha256_file(args.zip_path)
    fold_sha = sha256_file(args.fold) if args.fold else None
    reference_sha = sha256_file(args.reference) if args.reference else None
    with tempfile.TemporaryDirectory(
        prefix="native-panel-", dir=args.predictions.parent
    ) as temporary:
        package = Path(temporary) / "zip"
        package.mkdir()
        extract_zip(args.zip_path, package)
        zip_solution_sha = sha256_file(package / "solution.py")
        import_started = time.perf_counter()
        module, model, native_path, observed_level = import_submission(package, args.level)
        load_seconds = time.perf_counter() - import_started
        predictions_tmp = Path(temporary) / "predictions.npy"
        predictions = np.lib.format.open_memmap(
            predictions_tmp, mode="w+", dtype=np.float32, shape=expected_shape
        )
        metric = OfficialWPAccumulator()
        reference_squares = 0.0
        reference_elements = 0
        reference_bitwise_equal = True
        reference_max_abs = 0.0
        first_sequence: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
        callback_seconds = 0.0
        rows = 0
        for output_group, (source_group, sequence) in enumerate(
            zip(source_groups, sequence_ids, strict=True)
        ):
            table = parquet.read_row_group(source_group, use_threads=False)
            features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(
                np.float32, copy=False
            )
            if features.shape != (SEQUENCE_LENGTH, N_FEATURES) or not np.isfinite(features).all():
                raise ValueError(f"row group {source_group} has invalid features")
            target = np.column_stack([table[name].to_numpy() for name in TARGET_COLUMNS]).astype(
                np.float32, copy=False
            )
            need = table["need_prediction"].to_numpy().astype(bool)
            scored = table["is_scored"].to_numpy().astype(bool)
            output = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
            start = time.perf_counter()
            for step in range(SEQUENCE_LENGTH):
                value = callback_prediction(model, sequence, step, bool(need[step]), features[step])
                if value is not None:
                    output[step] = value
            callback_seconds += time.perf_counter() - start
            rows += SEQUENCE_LENGTH
            predictions[output_group] = output
            score_mask = scored & need
            metric.add(target, output, score_mask)
            if reference is not None:
                reference_group = reference[output_group]
                if not np.isfinite(reference_group[need]).all():
                    raise ValueError(
                        f"reference row group {output_group} has nonfinite predictions"
                    )
                delta = output[need].astype(np.float64) - reference_group[need].astype(np.float64)
                reference_max_abs = max(reference_max_abs, float(np.abs(delta).max(initial=0.0)))
                reference_squares += float(np.sum(delta * delta, dtype=np.float64))
                reference_elements += delta.size
                reference_bitwise_equal = reference_bitwise_equal and bool(
                    np.array_equal(output[need], reference_group[need])
                )
            if output_group == 0:
                first_sequence = (
                    sequence,
                    features[:120].copy(),
                    need[:120].copy(),
                    output[:120].copy(),
                )
            if (
                len(sequence_ids) <= MAX_SEQUENCES
                or (output_group + 1) % 20 == 0
                or output_group + 1 == len(sequence_ids)
            ):
                partial_score = metric.result() if metric.selected_rows else None
                print(
                    json.dumps(
                        {
                            "event": "native_panel_progress",
                            "groups_complete": output_group + 1,
                            "groups_total": len(sequence_ids),
                            "rows_complete": rows,
                            "elapsed_seconds": round(time.perf_counter() - started_all, 2),
                            "partial_wp": partial_score["wp"] if partial_score else None,
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )

        if first_sequence is None:
            raise RuntimeError("panel did not yield a first sequence")
        first_id, first_features, first_need, first_expected = first_sequence
        b_id = sequence_ids[1] if len(sequence_ids) > 1 else min(sequence_ids) - 1
        # Force a reset only for the A-B-A diagnostic, through sequence-ID changes.
        callback_prediction(model, b_id, 0, False, first_features[0])
        replay = np.zeros((120, 2), dtype=np.float32)
        for step in range(120):
            value = callback_prediction(
                model, first_id, step, bool(first_need[step]), first_features[step]
            )
            if value is not None:
                replay[step] = value
        aba_equal = bool(np.array_equal(replay, first_expected))
        if not aba_equal:
            raise AssertionError("A-B-A replay of the first 120 rows was not bitwise identical")
        predictions.flush()
        score = metric.result()
        reference_summary = None
        if reference is not None:
            reference_summary = {
                "bitwise_equal": reference_bitwise_equal,
                "max_abs": reference_max_abs,
                "rms": float(np.sqrt(reference_squares / reference_elements)),
            }
        rows_per_second = rows / callback_seconds if callback_seconds else 0.0
        source_stat_after = args.panel.stat()
        if (source_stat_before.st_size, source_stat_before.st_mtime_ns) != (
            source_stat_after.st_size,
            source_stat_after.st_mtime_ns,
        ):
            raise RuntimeError("input parquet changed while the panel was being checked")
        receipt = {
            "status": "passed_panel_check",
            "scope": "CPU panel callback, native ZIP import, official metric and parity diagnostics; not full submission runtime or generalization evidence",
            "zip": str(args.zip_path.resolve()),
            "zip_sha256": zip_sha,
            "zip_solution_sha256": zip_solution_sha,
            "native_library_sha256": sha256_file(native_path),
            "native_library_member": str(native_path.relative_to(package.resolve())),
            "requested_native_level": args.level,
            "observed_native_level": observed_level,
            "panel": str(args.panel.resolve()),
            "panel_sha256": panel_sha,
            "panel_sha256_scope": "full input file"
            if panel_sha
            else "not recomputed for a fold prefix; source file stat recorded",
            "fold": str(args.fold.resolve()) if args.fold else None,
            "fold_sha256": fold_sha,
            "selected_source_row_groups": source_groups,
            "available_selected_groups": available_groups,
            "reference": str(args.reference.resolve()) if args.reference else None,
            "reference_sha256": reference_sha,
            "reference_comparison_mask": "need_prediction only",
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pyarrow": __import__("pyarrow").__version__,
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "WNN_PRECISION",
                    "WNN_NATIVE_LEVEL",
                    "WNN_BUDGET_S",
                    "WNN_EXPECTED_ROWS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "reported_model_precision": getattr(getattr(model, "model", None), "precision", None),
            "sequence_ids": sequence_ids,
            "panel_bytes": source_stat_after.st_size,
            "panel_mtime_ns_before": source_stat_before.st_mtime_ns,
            "panel_mtime_ns_after": source_stat_after.st_mtime_ns,
            "rows": rows,
            "callbacks_including_warmup": rows + 121,
            "callback_seconds": callback_seconds,
            "load_seconds": load_seconds,
            "rows_per_second": rows_per_second,
            "eta_seconds_39_4m_rows": ETA_ROWS / rows_per_second if rows_per_second else None,
            "official_score": score,
            "reference_delta": reference_summary,
            "aba": {
                "first_sequence_id": first_id,
                "b_sequence_id": b_id,
                "rows_checked": 120,
                "bitwise_equal": aba_equal,
            },
            "predictions_path": str(args.predictions.resolve()),
            "predictions_sha256": sha256_file(predictions_tmp),
            "total_seconds": time.perf_counter() - started_all,
        }
        if not np.isfinite(np.asarray([receipt["official_score"]["wp"]], dtype=np.float64)).all():
            raise ValueError("score is nonfinite")
        # Publish the NPY with a create-only hard link; never overwrite prior results.
        os.link(predictions_tmp, args.predictions)
        with args.output.open("x", encoding="utf-8") as output_file:
            json.dump(receipt, output_file, indent=2, allow_nan=False)
            output_file.write("\n")
    sys.path.remove(str(package))
    sys.modules.pop("solution", None)
    sys.modules.pop("wnn_native", None)


if __name__ == "__main__":
    main()
