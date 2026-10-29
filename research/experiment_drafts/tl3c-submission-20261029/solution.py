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


class PredictionModel:
    """Advance every raw row, with recurrent and feature state reset at each sequence."""

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

        self.native_level = _native_level()
        native_dir = HERE / self.native_level
        sys.path.insert(0, str(native_dir))
        native = importlib.import_module("wnn_cpp")
        if Path(native.__file__).resolve().parent != native_dir:
            raise RuntimeError("wnn_cpp was imported from outside this submission")
        contract = json.loads((HERE / "members.json").read_text(encoding="utf-8"))
        if contract.get("schema") != "tl3c-native-members-v1":
            raise ValueError("unsupported members schema")
        self.mode = contract["mode"]
        if self.mode not in ("f32", "dynamic_int8"):
            raise ValueError("unsupported arithmetic mode")
        weights, coefficients = [], []
        for member in contract["members"]:
            filename = member["weights"]
            if Path(filename).name != filename or not filename.endswith(".npz"):
                raise ValueError("invalid weight filename")
            coefficient = float(member["coef"])
            if not math.isfinite(coefficient) or coefficient <= 0:
                raise ValueError("invalid ensemble coefficient")
            with np.load(HERE / filename, allow_pickle=False) as arrays:
                weights.append(
                    {
                        name: np.ascontiguousarray(arrays[name], dtype=np.float32)
                        for name in arrays.files
                    }
                )
            coefficients.append(coefficient)
        if not coefficients or not math.isclose(sum(coefficients), 1.0, abs_tol=1e-12):
            raise ValueError("ensemble coefficients must sum to one")
        self.model = native.Ensemble(
            weights,
            coefficients,
            fast_activations=self.mode == "dynamic_int8",
            half_weights=False,
            dynamic_int8=self.mode == "dynamic_int8",
            quant_backend="auto",
        )
        self.quant_backend = native.quant_backend()
        self.member_count = len(weights)
        self.seq_ix: int | None = None

    def reset(self) -> None:
        """Clear all native feature and recurrent state before an independent sequence."""
        self.model.reset()
        self.seq_ix = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """Advance state even on warmup/unscored rows; return only the two requested outputs."""
        import numpy as np

        if self.seq_ix != data_point.seq_ix:
            self.reset()
            self.seq_ix = data_point.seq_ix
        row = np.ascontiguousarray(data_point.state, dtype=np.float32)
        if row.shape != (112,):
            raise ValueError("expected 112 raw features per row")
        prediction = self.model.step(row)
        return prediction if data_point.need_prediction else None
