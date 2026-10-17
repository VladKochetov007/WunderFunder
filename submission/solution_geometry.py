"""Fixed-feature geometry GRU callback using the packaged native stepper."""

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
_SELECTED = np.asarray((0, 3, 52, 55), dtype=np.intp)
_RAW_WIDTH = 112
_EVENT_WIDTH = 99
_GEOMETRY_WIDTH = 60
_INPUT_WIDTH = _RAW_WIDTH + _EVENT_WIDTH + _GEOMETRY_WIDTH
_SCHEMA = "causal_geometry_discrete_v1"
_COUNTER_CAP = 10_000
_I0_PX = slice(0, 22)
_I0_BID_VOL = slice(22, 33)
_I0_ASK_VOL = slice(33, 44)
_I0_TRADE_PX = slice(44, 48)
_I0_TRADE_VOL = slice(48, 52)
_I1_PX = slice(52, 74)
_I1_VOL = slice(74, 96)
_I1_TRADE_VOL = slice(100, 104)
_V3_FLAGS = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}
_AVX2_FLAGS = {"avx", "avx2", "fma"}


class _Stepper(Protocol):
    """Native stateful model API used by this callback."""

    def reset(self) -> None: ...

    def run(self, rows: np.ndarray) -> np.ndarray: ...


class _DataPoint(Protocol):
    """Competition callback row contract."""

    seq_ix: int
    state: np.ndarray
    need_prediction: bool


def _cpu_flags() -> set[str]:
    """Read Linux CPU feature flags for safe native-library selection."""
    try:
        with (Path("/proc/cpuinfo")).open(encoding="utf-8") as source:
            for line in source:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        return set()
    return set()


def _native_directory() -> Path:
    """Select the strongest packaged native build supported by the host CPU."""
    flags = _cpu_flags()
    if flags >= _V3_FLAGS and (HERE / "v3").is_dir():
        return HERE / "v3"
    if flags >= _AVX2_FLAGS and (HERE / "avx2").is_dir():
        return HERE / "avx2"
    return HERE / "v2"


sys.path.insert(0, str(_native_directory()))
import wnn_native  # noqa: E402


def _feature_order() -> tuple[str, ...]:
    """Return the trained 60-channel order, including its discrete suffix."""
    return (
        *(
            f"geometry_{name}_{index}"
            for name in ("mean", "std", "arc_residual", "atom_occupancy", "normalized")
            for index in range(4)
        ),
        *(f"level_current_{field}_{level}" for field in range(4) for level in range(4)),
        *(f"level_occupancy_{field}_{level}" for field in range(4) for level in range(4)),
        *(f"level_dominant_event_{field}" for field in range(4)),
        *(f"level_log_dwell_{field}" for field in range(4)),
    )


def _load_prior(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate fitting-only feature constants from the package root."""
    value = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if value.get("schema") != _SCHEMA:
        raise ValueError(f"geometry prior schema must be {_SCHEMA!r}")
    if value.get("selected_raw_columns") != [0, 3, 52, 55]:
        raise ValueError("geometry prior selected columns do not match the callback")
    if value.get("feature_order") != list(_feature_order()):
        raise ValueError("geometry prior feature order does not match the callback")
    arrays = (
        np.asarray(value.get("beta"), dtype=np.float32),
        np.asarray(value.get("intercept"), dtype=np.float32),
        np.asarray(value.get("atoms"), dtype=np.float32),
        np.asarray(value.get("vocabulary"), dtype=np.float32),
    )
    expected = ((4,), (4,), (4,), (4, 3))
    if any(
        array.shape != shape or not np.isfinite(array).all()
        for array, shape in zip(arrays, expected, strict=True)
    ):
        raise ValueError(
            "geometry prior vectors must be finite with shapes (4,), (4,), (4,), and (4,3)"
        )
    return (
        np.ascontiguousarray(arrays[0], dtype=np.float64),
        np.ascontiguousarray(arrays[1], dtype=np.float64),
        np.ascontiguousarray(arrays[2], dtype=np.float64),
        np.ascontiguousarray(arrays[3], dtype=np.float64),
    )


def _load_weights(path: Path) -> dict[str, np.ndarray]:
    """Load one exported member as finite FP32 tensors with supported inputs."""
    with np.load(path, allow_pickle=False) as archive:
        weights = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32) for name in archive.files
        }
    input_width = weights.get("mean", np.empty(0)).shape
    if (
        input_width not in {(211,), (_INPUT_WIDTH,)}
        or weights.get("std", np.empty(0)).shape != input_width
    ):
        raise ValueError("member normalizers must have either 211 or 271 inputs")
    if (
        not np.isfinite(weights["mean"]).all()
        or not np.isfinite(weights["std"]).all()
        or (weights["std"] <= 0).any()
    ):
        raise ValueError("model normalizers must be finite with positive scales")
    if not all(np.isfinite(value).all() for value in weights.values()):
        raise ValueError("model contains non-finite weights")
    return weights


class _EventStream:
    """Incremental twin of the 99 causal raw-event features."""

    def __init__(self) -> None:
        self.out = np.zeros(_EVENT_WIDTH, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Clear previous-row and event-history state at a sequence boundary."""
        self.previous: np.ndarray | None = None
        self.step = 0
        self.last_event = np.zeros(3, dtype=np.int64)
        self.held = np.zeros(22, dtype=np.float32)

    def update(self, row: np.ndarray) -> np.ndarray:
        """Advance once and return the event vector, even on unscored rows."""
        previous = row if self.previous is None else self.previous
        delta = row - previous
        changed = previous != row
        flags = (
            bool(changed[0] and changed[11]),
            bool(changed[_I0_BID_VOL].any()),
            bool(changed[_I0_ASK_VOL].any()),
            bool(changed[_I0_TRADE_PX].any()),
            bool(changed[_I0_TRADE_VOL].any()),
            bool(changed[52] or changed[63]),
            bool(changed[_I1_VOL].any()),
            bool(changed[_I1_TRADE_VOL].any()),
        )
        for index, flag_index in enumerate((0, 4, 5)):
            if flags[flag_index]:
                self.last_event[index] = self.step
        if flags[0]:
            self.held[:] = delta[_I0_PX]
        self.out[:22] = delta[_I0_PX]
        self.out[22:44] = delta[22:44]
        self.out[44:66] = delta[_I1_PX]
        self.out[66:74] = flags
        self.out[74:77] = np.log1p(np.minimum(self.step - self.last_event, _COUNTER_CAP))
        self.out[77:99] = self.held
        self.previous = row.copy()
        self.step += 1
        return self.out


class _GeometryStream:
    """Reset-local FP64 moments and exact-level counters, emitted as FP32."""

    def __init__(self, prior: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        self.beta, self.intercept, self.atoms, self.vocabulary = prior
        self.out = np.empty(_GEOMETRY_WIDTH, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Clear all causal accumulators at a sequence boundary."""
        self.position = 0
        self.count = 0
        self.previous: np.ndarray | None = None
        self.total = np.zeros(4, dtype=np.float64)
        self.total_squared = np.zeros(4, dtype=np.float64)
        self.atom_count = np.zeros(4, dtype=np.float64)
        self.occupancy = np.zeros((4, 4), dtype=np.float64)
        self.event_count = np.zeros(4, dtype=np.int64)
        self.dominant_count = np.zeros(4, dtype=np.int64)
        self.last_change = np.zeros(4, dtype=np.int64)
        self.category = np.empty((4, 4), dtype=np.bool_)

    def update(self, row: np.ndarray) -> np.ndarray:
        """Update causal state with the current raw row and return 60 features."""
        values = row[_SELECTED].astype(np.float64)
        changed = np.ones(4, dtype=np.bool_) if self.previous is None else values != self.previous
        self.total += values
        self.total_squared += values * values
        self.count += 1
        mean = self.total / self.count
        second = self.total_squared / self.count
        std = np.sqrt(np.maximum(second - mean * mean, 0.0))
        self.atom_count += values == self.atoms
        known = values[:, None] == self.vocabulary
        self.category[:, :3] = known
        self.category[:, 3] = ~known.any(axis=1)
        self.occupancy += self.category
        self.event_count += changed
        self.dominant_count += changed & self.category[:, 0]
        self.last_change[changed] = self.position

        out = self.out
        out[:4] = mean
        out[4:8] = std
        out[8:12] = second - self.beta * mean - self.intercept
        out[12:16] = self.atom_count / self.count
        out[16:20] = np.clip((values - mean) / np.maximum(std, 0.05), -8.0, 8.0)
        np.copyto(out[20:36], self.category.reshape(16), casting="unsafe")
        out[36:52] = (self.occupancy / self.count).reshape(16)
        out[52:56] = self.dominant_count / np.maximum(self.event_count, 1)
        out[56:60] = np.log1p(np.minimum(self.position - self.last_change, _COUNTER_CAP))
        if self.previous is None:
            self.previous = values.copy()
        else:
            self.previous[:] = values
        self.position += 1
        return out


class PredictionModel:
    """No-argument scorer callback with fixed FP32 or package-selected precision."""

    def __init__(self) -> None:
        prior = _load_prior(HERE / "geometry_prior.json")
        self.seq_ix: int | None = None
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
        if not spec:
            raise ValueError("members.json must contain at least one model")
        prior_f32 = tuple(np.asarray(value, dtype=np.float32) for value in prior)
        self.native_geometry_model = None
        self.native_feature_stepper = None
        self.feature_path = "python"
        self.model_path = "model_stepper"
        if hasattr(wnn_native, "GeometryFeatureStepper"):
            self.native_feature_stepper = wnn_native.GeometryFeatureStepper(*prior_f32)
            self.feature_path = "native"
        self.members: list[tuple[_Stepper, np.float32, int, np.ndarray]] = []
        if len(spec) == 1 and hasattr(wnn_native, "GeometryPredictor"):
            item = spec[0]
            weights = _load_weights(HERE / str(item["weights"]))
            if weights["mean"].shape == (_INPUT_WIDTH,):
                self.native_geometry_model = wnn_native.GeometryPredictor(
                    weights,
                    *prior_f32,
                    precision=precision,
                )
                self.native_geometry_coefficient = np.float32(item["coef"])
                if not np.isfinite(self.native_geometry_coefficient):
                    raise ValueError("member coefficients must be finite")
                self.model_path = "native_geometry_predictor"
                return
        for member in spec:
            name = str(member["weights"])
            coefficient = np.float32(member["coef"])
            if not np.isfinite(coefficient):
                raise ValueError("member coefficients must be finite")
            weights = _load_weights(HERE / name)
            stepper = wnn_native.ModelStepper(weights, fast_activations=True, precision=precision)
            input_width = len(weights["mean"])
            row_buffer = np.empty((1, input_width), dtype=np.float32)
            self.members.append((cast(_Stepper, stepper), coefficient, input_width, row_buffer))
        if self.native_feature_stepper is None:
            self.geometry = _GeometryStream(prior)
            self.events = _EventStream()

    def _reset(self) -> None:
        if self.native_geometry_model is not None:
            self.native_geometry_model.reset()
        elif self.native_feature_stepper is not None:
            self.native_feature_stepper.reset()
        else:
            self.geometry.reset()
            self.events.reset()
        for model, _, _, _ in self.members:
            model.reset()

    def predict(self, data_point: _DataPoint) -> np.ndarray | None:
        """Advance state for every row and return the fixed blend when requested."""
        seq_ix = int(data_point.seq_ix)
        if seq_ix != self.seq_ix:
            self.seq_ix = seq_ix
            self._reset()
        row = np.asarray(data_point.state, dtype=np.float32)
        if row.shape != (_RAW_WIDTH,) or not np.isfinite(row).all():
            raise ValueError("expected one finite raw row with shape (112,)")
        row = np.ascontiguousarray(row)
        if self.native_geometry_model is not None:
            value = self.native_geometry_model.step(row, bool(data_point.need_prediction))
            if value is None:
                return None
            prediction = np.asarray(value, dtype=np.float32) * self.native_geometry_coefficient
            if not np.isfinite(prediction).all():
                prediction = np.nan_to_num(prediction, nan=0.0, posinf=2.0, neginf=-2.0)
            return prediction
        if self.native_feature_stepper is not None:
            features = self.native_feature_stepper.step(row)
            events = features[:_EVENT_WIDTH]
            geometry = features[_EVENT_WIDTH:]
        else:
            events = self.events.update(row)
            geometry = self.geometry.update(row)
        prediction = np.zeros(2, dtype=np.float32)
        for model, coefficient, width, model_row in self.members:
            model_row[0, :_RAW_WIDTH] = row
            model_row[0, _RAW_WIDTH : _RAW_WIDTH + _EVENT_WIDTH] = events
            if width == _INPUT_WIDTH:
                model_row[0, _RAW_WIDTH + _EVENT_WIDTH :] = geometry
            prediction += coefficient * model.run(model_row)[0, :2]
        if not data_point.need_prediction:
            return None
        if not np.isfinite(prediction).all():
            prediction = np.nan_to_num(prediction, nan=0.0, posinf=2.0, neginf=-2.0)
        return prediction
