"""Native recurrent ensemble: 16-bit weights, the scorer callback in Rust, fixed members and weights.

members.json lists weight files with blend coefficients (GRU or LSTM is inferred from the weight
shapes; head.0 may be k*H wide for merged heads). WNN_PRECISION selects the weight storage:
"i16" (default; int16 with an fp32 scale per row, |delta| < 1e-4 vs fp32 on trained members),
"f16" (IEEE half, ~15% faster here, |delta| ~4e-4) or "f32". CPUs without AVX2/FMA/F16C keep fp32
automatically (logged). CPU level: v3 when the full x86-64-v3 flag set is present, otherwise the
avx2+fma build when present, otherwise v2; the 16-bit path gives bitwise-identical outputs on all
three. Membership and coefficients never change during a run: the native runtime governor is
disabled (budget_s=None), so prediction identity does not depend on wall-clock speed.
"""

import json
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
_V3_FLAGS = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}
_AVX2_FLAGS = {"avx", "avx2", "fma"}
_PRECISION_FILE = HERE / "precision.txt"  # written at packaging time; the scorer cannot set env vars
PRECISION = os.environ.get("WNN_PRECISION") or (
    _PRECISION_FILE.read_text().strip() if _PRECISION_FILE.is_file() else "i16")


def _cpu_flags():
    try:
        with open("/proc/cpuinfo") as source:
            for line in source:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _level():
    forced = os.environ.get("WNN_NATIVE_LEVEL")
    if forced in ("v2", "avx2", "v3"):
        return forced
    flags = _cpu_flags()
    if flags >= _V3_FLAGS:
        return "v3"
    if flags >= _AVX2_FLAGS and (HERE / "avx2").is_dir():
        return "avx2"
    return "v2"


_LEVEL = _level()
sys.path.insert(0, str(HERE / _LEVEL))
import wnn_native  # noqa: E402


def _load(name):
    with np.load(HERE / name) as weights:
        return {key: np.ascontiguousarray(weights[key], dtype=np.float32) for key in weights.files}


class PredictionModel:
    def __init__(self):
        members = json.loads((HERE / "members.json").read_text())
        self.model = wnn_native.EnsemblePredictor(
            [_load(member["weights"]) for member in members],
            [np.float32(member["coef"]) for member in members],
            fast_activations=True,
            precision=PRECISION,
            budget_s=None,
        )
        # The scorer calls model.predict(data_point); bind it straight to the native method.
        self.predict = self.model.predict
        print(
            f"[wnn] level={_LEVEL} members={len(members)} precision={self.model.precision}",
            file=sys.stderr,
            flush=True,
        )

    def predict(self, data_point):  # shadowed by the instance attribute set in __init__
        return self.model.predict(data_point)
