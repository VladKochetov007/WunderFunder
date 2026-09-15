"""Submission: native streaming GRU trained with scoring-propensity weights.

Files next to this module: weights.npz (model state dict), wnn_native*.so
(Rust event features + GRU, abi3 for CPython >= 3.11).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

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
