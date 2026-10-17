"""Native-only scorer for one 271-input causal geometry model."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Protocol, cast

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
_INPUT_WIDTH = 271
_SCHEMA = "causal_geometry_discrete_v1"
_V3_FLAGS = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}
_AVX2_FLAGS = {"avx", "avx2", "fma"}


class _DataPoint(Protocol):
    """Competition callback row contract."""

    seq_ix: int
    state: np.ndarray
    need_prediction: bool


def _native_directory() -> Path:
    """Select the strongest packaged native build supported by this host."""
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8") as source:
            flags = next(
                set(line.split(":", 1)[1].split()) for line in source if line.startswith("flags")
            )
    except (OSError, StopIteration):
        flags = set()
    if flags >= _V3_FLAGS and (HERE / "v3").is_dir():
        return HERE / "v3"
    if flags >= _AVX2_FLAGS and (HERE / "avx2").is_dir():
        return HERE / "avx2"
    return HERE / "v2"


sys.path.insert(0, str(_native_directory()))
import wnn_native  # noqa: E402


def _feature_order() -> list[str]:
    """Return the fixed geometry and discrete feature order."""
    return [
        *(
            f"geometry_{name}_{index}"
            for name in ("mean", "std", "arc_residual", "atom_occupancy", "normalized")
            for index in range(4)
        ),
        *(f"level_current_{field}_{level}" for field in range(4) for level in range(4)),
        *(f"level_occupancy_{field}_{level}" for field in range(4) for level in range(4)),
        *(f"level_dominant_event_{field}" for field in range(4)),
        *(f"level_log_dwell_{field}" for field in range(4)),
    ]


def _load_prior(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the fitting-only constants and schema shipped with the model."""
    value = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if (
        value.get("schema") != _SCHEMA
        or value.get("selected_raw_columns") != [0, 3, 52, 55]
        or value.get("feature_order") != _feature_order()
    ):
        raise ValueError("geometry prior schema, columns, or feature order do not match")
    arrays = tuple(
        np.ascontiguousarray(value[name], dtype=np.float32)
        for name in ("beta", "intercept", "atoms", "vocabulary")
    )
    if tuple(array.shape for array in arrays) != ((4,), (4,), (4,), (4, 3)):
        raise ValueError("geometry prior must have shapes (4,), (4,), (4,), and (4,3)")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("geometry prior values must be finite")
    return cast(tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], arrays)


def _load_weights(path: Path) -> dict[str, np.ndarray]:
    """Load and validate one packaged FP32 member with 271 inputs."""
    with np.load(path, allow_pickle=False) as archive:
        weights = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32) for name in archive.files
        }
    if weights.get("mean", np.empty(0)).shape != (_INPUT_WIDTH,) or weights.get(
        "std", np.empty(0)
    ).shape != (_INPUT_WIDTH,):
        raise ValueError(f"geometry member normalizers must have {_INPUT_WIDTH} inputs")
    if not np.isfinite(weights["mean"]).all() or not np.isfinite(weights["std"]).all():
        raise ValueError("model normalizers must be finite")
    if (weights["std"] <= 0).any() or not all(
        np.isfinite(value).all() for value in weights.values()
    ):
        raise ValueError("model tensors must be finite with positive normalizer scales")
    return weights


class PredictionModel:
    """Single-member native callback; every row advances the stateful scorer."""

    def __init__(self) -> None:
        if not hasattr(wnn_native, "GeometryPredictor"):
            raise RuntimeError(
                "packaged wnn_native lacks GeometryPredictor; rebuild native/src/python_geometry.rs"
            )
        prior = _load_prior(HERE / "geometry_prior.json")
        precision_path = HERE / "precision.txt"
        precision = (
            precision_path.read_text(encoding="utf-8").strip()
            if precision_path.is_file()
            else "f32"
        )
        if precision not in {"f32", "f16", "i16", "q16x", "q8vnni"}:
            raise ValueError(f"unsupported fixed package precision {precision!r}")
        spec = cast(
            list[dict[str, object]], json.loads((HERE / "members.json").read_text(encoding="utf-8"))
        )
        if len(spec) != 1:
            raise ValueError("geometry serving currently requires one fixed model member")
        member = spec[0]
        coefficient = np.float32(member["coef"])
        if not np.isfinite(coefficient):
            raise ValueError("member coefficient must be finite")
        weights = _load_weights(HERE / str(member["weights"]))
        self.stepper = wnn_native.GeometryPredictor(
            weights,
            *prior,
            precision=precision,
        )
        self.coefficient = coefficient
        self.seq_ix: int | None = None

    def predict(self, data_point: _DataPoint) -> np.ndarray | None:
        """Advance native features and recurrence, returning only requested rows."""
        seq_ix = int(data_point.seq_ix)
        if seq_ix != self.seq_ix:
            self.seq_ix = seq_ix
            self.stepper.reset()
        row = np.asarray(data_point.state, dtype=np.float32)
        if row.shape != (112,) or not np.isfinite(row).all():
            raise ValueError("expected one finite raw row with shape (112,)")
        value = self.stepper.step(np.ascontiguousarray(row), bool(data_point.need_prediction))
        if value is None:
            return None
        prediction = np.asarray(value, dtype=np.float32) * self.coefficient
        if not np.isfinite(prediction).all():
            prediction = np.nan_to_num(prediction, nan=0.0, posinf=2.0, neginf=-2.0)
        return prediction
