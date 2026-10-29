"""Execute complete competition streams and compare reset and reference behavior."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from validation_io import read_json, sha256

from utils import (
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    DataPoint,
    PredictionModel,
    validate_sequence,
)

if TYPE_CHECKING:
    import pyarrow.parquet as pq


def load_model(package: Path, level: str) -> tuple[PredictionModel, dict]:
    """Import only the extracted shipped solution and verify the selected native path."""
    os.environ["WNN_NATIVE_LEVEL"] = level
    if "wnn_cpp" in sys.modules:
        raise RuntimeError("each candidate and CPU level requires a fresh Python process")
    spec = importlib.util.spec_from_file_location("candidate_solution", package / "solution.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load solution.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.PredictionModel()
    native = sys.modules.get("wnn_cpp")
    if native is None or Path(native.__file__).resolve().parent != (package / level).resolve():
        raise ImportError("solution did not import the expected extracted wnn_cpp level")
    return model, {
        "level": level,
        "path": str(native.__file__),
        "sha256": sha256(Path(native.__file__)),
    }


def load_sequence(parquet: pq.ParquetFile, group: int) -> dict:
    """Decode one complete official sequence; never skip unscored callback rows."""
    columns = [
        "seq_ix",
        "step_in_seq",
        "need_prediction",
        "is_scored",
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    ]
    table = parquet.read_row_group(group, columns=columns, use_threads=False)
    sequence, need = validate_sequence(table)
    features = np.column_stack([table[key].to_numpy() for key in FEATURE_COLUMNS]).astype(
        np.float32
    )
    targets = np.column_stack([table[key].to_numpy() for key in TARGET_COLUMNS]).astype(np.float32)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("nonfinite data")
    return {
        "seq_ix": sequence,
        "features": features,
        "targets": targets,
        "need": need,
        "score": table["is_scored"].to_numpy().astype(bool) & need,
    }


def replay(
    model: PredictionModel, stream: dict, *, force_all_outputs: bool = False
) -> tuple[np.ndarray, float]:
    """Include every state update and callback in wall time, excluding parquet decode."""
    predictions = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
    start = time.perf_counter()
    for step, row in enumerate(stream["features"]):
        need = bool(stream["need"][step]) or force_all_outputs
        value = model.predict(DataPoint(stream["seq_ix"], step, need, row))
        if not need:
            if value is not None:
                raise ValueError(f"warmup callback returned a prediction at {step}")
            continue
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.shape != (2,)
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"callback requires finite float32[2] at row {step}")
        predictions[step] = value
    return predictions, time.perf_counter() - start


def same_bits(actual: np.ndarray, expected: np.ndarray, mask: np.ndarray) -> bool:
    """Use a bitwise reset check on all predictions required by the official ABI."""
    return bool(np.array_equal(actual[mask].view(np.uint32), expected[mask].view(np.uint32)))


def state_checks(
    model: PredictionModel, first: dict, first_predictions: np.ndarray, second: dict
) -> dict:
    """A-B-A proves sequence resets; alternate need flags prove output-mask independence."""
    if first["seq_ix"] == second["seq_ix"]:
        raise ValueError("A-B-A requires two distinct sequence IDs")
    replay(model, second)
    repeated, _ = replay(model, first)
    if not same_bits(repeated, first_predictions, first["need"]):
        raise ValueError("A-B-A replay changed predictions: state reset failed")
    replay(model, second)
    all_rows, _ = replay(model, first, force_all_outputs=True)
    if not same_bits(all_rows, first_predictions, first["need"]):
        raise ValueError("need_prediction changed recurrent state or feature clocks")
    return {
        "a_b_a_bitwise": True,
        "warmup_mask_state_independent": True,
        "callback_rows_skipped": 0,
        "input_gap_policy": "gaps forbidden; validate_sequence requires every step 0..19999",
    }


def open_reference(
    path: Path | None, metadata_path: Path | None, data_sha: str, groups: list[int], split_sha: str
) -> tuple[np.ndarray | None, dict | None]:
    """Accept only identity-bound ordered float32 reference predictions."""
    if path is None:
        if metadata_path is not None:
            raise ValueError("--reference-metadata requires --reference")
        return None, None
    if metadata_path is None:
        raise ValueError("reference needs its identity metadata")
    metadata = read_json(metadata_path)
    if (
        metadata.get("data_sha256") != data_sha
        or metadata.get("split_sha256") != split_sha
        or metadata.get("row_groups") != groups
        or metadata.get("predictions", {}).get("sha256") != sha256(path)
    ):
        raise ValueError("reference data, split, row order or prediction hash mismatch")
    if metadata.get("reference_kind") not in {
        "torch_chunk4000",
        "callback_f32",
        "callback_dynamic_int8",
    }:
        raise ValueError("reference must declare its predictor semantics")
    predictions = np.load(path, mmap_mode="r", allow_pickle=False)
    if predictions.dtype != np.float32 or predictions.shape != (len(groups), SEQUENCE_LENGTH, 2):
        raise ValueError("reference must be float32[groups,20000,2]")
    return predictions, metadata


def differences(actual: np.ndarray, expected: np.ndarray, need: np.ndarray) -> dict:
    """Probe all served rows and windows around every canonical evaluation boundary."""
    if not np.isfinite(expected[need]).all():
        raise ValueError("nonfinite required reference prediction")
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    boundaries = [
        row
        for start in range(4000, SEQUENCE_LENGTH, 4000)
        for row in range(start - 2, start + 3)
        if need[row]
    ]
    return {
        "max_abs": float(error[need].max()),
        "boundary4000_max_abs": float(error[boundaries].max()),
        "boundary4000_rows": boundaries,
        "bitwise_equal": same_bits(actual, expected, need),
    }
