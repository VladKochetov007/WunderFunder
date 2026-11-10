from __future__ import annotations

import importlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

HERE = Path(__file__).resolve().parent
FEATURES = {
    "v2": {"ssse3", "sse4_1", "sse4_2", "popcnt", "cx16", "lahf_lm"},
    "avx2": {"avx", "avx2", "fma"},
    "v3": {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"},
}
BASE_MEMBERS = (
    ("tl3c_w18final_distilled_s5", "event211", "w18.npz", 0.34),
    ("tl3c_fttall_tlead_s5", "tiedlead263", "tiedlead5.npz", 0.33),
    ("tl3c_fttall_tlead_s6", "tiedlead263", "tiedlead6.npz", 0.33),
)
BASE_WEIGHT = 0.9
LSTM_WEIGHT = 0.1


class DataPoint(Protocol):
    """The competition provides one chronological row per callback."""

    seq_ix: int
    state: object
    need_prediction: bool


def _native_level() -> str:
    flags: set[str] = set()
    with Path("/proc/cpuinfo").open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
                break
    if "lzcnt" in flags:
        flags.add("abm")
    requested = os.environ.get("WNN_NATIVE_LEVEL")
    if requested is not None:
        if requested not in FEATURES or not flags >= FEATURES[requested] | FEATURES["v2"]:
            raise RuntimeError("requested WNN_NATIVE_LEVEL is unknown or unsupported by this CPU")
        return requested
    for level in ("v3", "avx2", "v2"):
        if flags >= FEATURES[level] | FEATURES["v2"]:
            return level
    raise RuntimeError("no packaged x86-64 ISA is supported on this CPU")


def _read_members() -> tuple[str, list[dict[str, object]]]:
    """Validate canonical TL3 members carried by the PR51 native runtime."""
    contract = json.loads((HERE / "members.json").read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("schema") != "tl3c-native-members-v1":
        raise ValueError("unsupported native member contract")
    mode = contract.get("mode")
    if mode != "dynamic_int8":
        raise ValueError("the pinned CPU candidate requires the PR51 dynamic_int8 baseline")
    members = contract.get("members")
    if not isinstance(members, list) or len(members) != len(BASE_MEMBERS):
        raise ValueError("the pinned native baseline requires exactly three members")
    for member, expected in zip(members, BASE_MEMBERS, strict=True):
        if not isinstance(member, dict):
            raise ValueError("native member entries must be objects")
        expected_run, expected_family, expected_weights, expected_coef = expected
        if (
            member.get("run") != expected_run
            or member.get("family") != expected_family
            or member.get("weights") != expected_weights
            or isinstance(member.get("coef"), bool)
            or not isinstance(member.get("coef"), (int, float))
            or not math.isclose(float(member["coef"]), expected_coef, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("native member order, identity, weights, or coefficient changed")
    return mode, members


def _load_lstm_weights() -> dict[str, np.ndarray]:
    """Read the pinned LSTM tensor archive without deserializing executable objects."""
    import numpy as np
    from lstm_cpu import TENSOR_SHAPES

    with np.load(HERE / "lstm.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(TENSOR_SHAPES):
            raise ValueError("LSTM archive tensor names differ from the pinned schema")
        weights: dict[str, np.ndarray] = {}
        for name, shape in TENSOR_SHAPES.items():
            value = arrays[name]
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError(f"unexpected LSTM tensor: {name}")
            if not np.isfinite(value).all():
                raise ValueError(f"LSTM tensor is not finite: {name}")
            weights[name] = np.ascontiguousarray(value, dtype=np.float32)
    return weights


class PredictionModel:
    """Stream canonical TL3 through PR51 alongside LSTM with separate sequence states."""

    def __init__(self) -> None:
        if sys.version_info[:2] != (3, 11) or platform.system() != "Linux":
            raise RuntimeError("this native archive requires Linux CPython 3.11")
        if platform.machine().lower() not in ("x86_64", "amd64"):
            raise RuntimeError("this native archive requires x86-64")
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[key] = "1"
        import numpy as np
        from lstm_cpu import LSTM192CPU
        from threadpoolctl import threadpool_limits

        self._threadpool_limiter = threadpool_limits(limits=1)

        self.native_level = _native_level()
        native_dir = HERE / self.native_level
        sys.path.insert(0, str(native_dir))
        native = importlib.import_module("wnn_cpp")
        if Path(native.__file__).resolve().parent != native_dir:
            raise RuntimeError("wnn_cpp was imported from outside this submission")

        self.mode, members = _read_members()
        weights: list[dict[str, np.ndarray]] = []
        coefficients: list[float] = []
        for member in members:
            filename = str(member["weights"])
            if Path(filename).name != filename or not filename.endswith(".npz"):
                raise ValueError("invalid native weight filename")
            with np.load(HERE / filename, allow_pickle=False) as arrays:
                weights.append(
                    {
                        name: np.ascontiguousarray(arrays[name], dtype=np.float32)
                        for name in arrays.files
                    }
                )
            coefficients.append(float(member["coef"]))
        if not math.isclose(sum(coefficients), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("native baseline coefficients must sum to one")
        self.model = native.Ensemble(
            weights,
            coefficients,
            fast_activations=True,
            half_weights=False,
            dynamic_int8=True,
            quant_backend="auto",
        )
        self.quant_backend = native.quant_backend()
        self.member_count = len(weights)
        self.lstm = LSTM192CPU(_load_lstm_weights())
        self.seq_ix: int | None = None
        self.last_components: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self) -> None:
        """Clear native and LSTM feature state before a new sequence."""
        self.model.reset()
        self.lstm.reset()
        self.seq_ix = None
        self.last_components = None

    def predict_components(self, data_point: DataPoint) -> tuple[np.ndarray, np.ndarray]:
        """Advance both predictors once and expose the current raw component outputs.

        Args:
            data_point: One competition callback with raw112 state.

        Returns:
            Float32 `(baseline, LSTM)` vectors, each shaped `(2,)`.

        Raises:
            ValueError: If the callback row does not contain 112 finite features.
        """
        import numpy as np

        if self.seq_ix != data_point.seq_ix:
            self.reset()
            self.seq_ix = data_point.seq_ix
        row = np.ascontiguousarray(data_point.state, dtype=np.float32)
        if row.shape != (112,) or not np.isfinite(row).all():
            raise ValueError("expected one finite raw112 row")
        baseline = np.asarray(self.model.step(row), dtype=np.float32)
        lstm = self.lstm.step(row)
        if baseline.shape != (2,):
            raise ValueError("native baseline must return a two-output vector")
        self.last_components = (
            np.ascontiguousarray(baseline, dtype=np.float32),
            np.ascontiguousarray(lstm, dtype=np.float32),
        )
        return self.last_components

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """Advance every row and return the unclipped fixed 90/10 mixture when scored."""
        import numpy as np

        baseline, lstm = self.predict_components(data_point)
        prediction = np.asarray(
            BASE_WEIGHT * baseline.astype(np.float64) + LSTM_WEIGHT * lstm.astype(np.float64),
            dtype=np.float32,
        )
        return prediction if data_point.need_prediction else None
