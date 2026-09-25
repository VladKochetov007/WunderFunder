"""Submission: blend of two native streaming GRUs.

weights_gru.npz is our GRU (wMSE + auxiliary heads, trained on train + validation);
weights_base.npz is the organizer baseline rewritten exactly as the same model family
(scripts/convert_baseline.py). Both run in the Rust extension shipped as v3/ (x86-64-v3)
and v2/ (baseline x86-64); v3 is loaded only when the CPU reports every flag it needs.
Output = W_GRU * gru + W_BASE * BASE_SCALE * baseline.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_V3_FLAGS = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}
W_GRU, W_BASE, BASE_SCALE = 0.5, 0.5, 0.25


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


def _load(name):
    with np.load(HERE / name) as w:
        return {k: np.ascontiguousarray(w[k], dtype=np.float32) for k in w.files}


class PredictionModel:
    def __init__(self):
        self.gru = wnn_native.Predictor(_load("weights_gru.npz"), fast_activations=True)
        self.base = wnn_native.Predictor(_load("weights_base.npz"), fast_activations=True)
        self.coef = np.array([W_GRU, W_BASE * BASE_SCALE], dtype=np.float32)
        self.seq_ix = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.gru.reset()
            self.base.reset()
        row = np.ascontiguousarray(data_point.state, dtype=np.float32)
        g = self.gru.step(row)
        b = self.base.step(row)
        if not data_point.need_prediction:
            return None
        out = self.coef[0] * g + self.coef[1] * b
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=2.0, neginf=-2.0)
        return out.astype(np.float32)
