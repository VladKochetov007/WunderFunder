"""Submission: native streaming GRU trained with scoring-propensity weights.

Files next to this module: weights.npz (model state dict) and two builds of the
Rust extension, v3/ (x86-64-v3: AVX2, FMA, BMI) and v2/ (x86-64-v2 baseline).
The v3 build is loaded only when the CPU reports every flag it needs.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3_FLAGS = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}


def _cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _native_dir():
    forced = os.environ.get("WNN_NATIVE_LEVEL")
    if forced in ("v2", "v3"):
        return HERE / forced
    return HERE / ("v3" if _V3_FLAGS <= _cpu_flags() else "v2")


sys.path.insert(0, str(_native_dir()))
import wnn_native  # noqa: E402


class PredictionModel:
    def __init__(self):
        with np.load(HERE / "weights.npz") as w:
            weights = {k: np.ascontiguousarray(w[k], dtype=np.float32) for k in w.files}
        self.predictor = wnn_native.Predictor(weights, fast_activations=True)
        self.seq_ix = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.predictor.reset()
        out = self.predictor.step(np.ascontiguousarray(data_point.state, dtype=np.float32))
        if not data_point.need_prediction:
            return None
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=2.0, neginf=-2.0)
        return out
