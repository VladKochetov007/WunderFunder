"""Submission: blend of native streaming GRUs.

members.json lists weight files (model state dicts of wnn.nn.RNNModel, head sliced to the two
targets) with blend coefficients and encoder ("linear" or "conv"); the organizer baseline enters as an exact RNNModel rewrite
(scripts/convert_baseline.py) with its output scale folded into its coefficient. All members run
in the Rust extension shipped as v3/ (x86-64-v3) and v2/ (x86-64-v2); v3 is loaded only when the
CPU reports every flag it needs.
"""

import json
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


def _load(name):
    with np.load(HERE / name) as w:
        return {k: np.ascontiguousarray(w[k], dtype=np.float32) for k in w.files}


def _predictor(m):
    # enc="conv" members need a build that runs the conv encoder (older builds would ignore its weights).
    enc = m.get("enc", "linear")
    if enc not in getattr(wnn_native, "ENCODERS", ("linear",)):
        raise RuntimeError(f"{m['weights']}: native build does not support enc={enc}")
    p = wnn_native.Predictor(_load(m["weights"]), fast_activations=True)
    if getattr(p, "encoder", "linear") != enc:
        raise RuntimeError(f"{m['weights']}: weights select enc={p.encoder}, members.json says {enc}")
    return p


class PredictionModel:
    def __init__(self):
        spec = json.loads((HERE / "members.json").read_text())
        self.members = [_predictor(m) for m in spec]
        self.coef = [np.float32(m["coef"]) for m in spec]
        self.seq_ix = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            for m in self.members:
                m.reset()
        row = np.ascontiguousarray(data_point.state, dtype=np.float32)
        out = None
        for m, c in zip(self.members, self.coef):
            v = c * m.step(row)
            out = v if out is None else out + v
        if not data_point.need_prediction:
            return None
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=2.0, neginf=-2.0)
        return out.astype(np.float32)
