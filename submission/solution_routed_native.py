"""Market-class-routed native ensembles: fixed groups, fixed weights, no runtime governor.

routing.json:
  {"groups": {"a": [{"weights": "w0.npz", "coef": 0.5}, ...], "b": [...]},
   "class_group": {"0": "a", "1": "a", "2": "a", "3": "b"},
   "detector": {"weights": "detector.npz", "lo": 17, "hi": 21, "from_row": 50}}

Every group ensemble is stepped on the warm-up rows (0-98), and so is the detector: a member with its full head
whose auxiliary market-class logits sit at [lo, hi). Softmax class probabilities are averaged over warm-up rows
from `from_row` on; at the first row that needs a prediction the most probable class picks the group, and only
that group is stepped for the rest of the sequence. Groups reset themselves on a new seq_ix; the detector and the
route reset here. Only rows up to the current one are used. Weights are stored as int16 with per-row fp32 scales
("i16"); CPU level selection is as in solution_native.py and outputs are identical across levels.
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
WARMUP = 99


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
        cfg = json.loads((HERE / "routing.json").read_text())
        self.groups = {
            name: wnn_native.EnsemblePredictor(
                [_load(m["weights"]) for m in members],
                [np.float32(m["coef"]) for m in members],
                fast_activations=True,
                precision=PRECISION,
                budget_s=None,
            )
            for name, members in cfg["groups"].items()
        }
        self.class_group = {int(k): v for k, v in cfg["class_group"].items()}
        det = cfg["detector"]
        self.detector = wnn_native.Predictor(_load(det["weights"]), fast_activations=True, precision=PRECISION)
        self.lo, self.hi, self.from_row = int(det["lo"]), int(det["hi"]), int(det["from_row"])
        self.seq = None
        self.route = None
        self.votes = np.zeros(self.hi - self.lo, dtype=np.float64)
        print(f"[wnn] level={_LEVEL} routed groups={ {k: len(v) for k, v in cfg['groups'].items()} } "
              f"precision={PRECISION}", file=sys.stderr, flush=True)

    def predict(self, data_point):
        if data_point.seq_ix != self.seq:
            self.seq = data_point.seq_ix
            self.route = None
            self.votes[:] = 0.0
            self.detector.reset()
        if self.route is None:
            step = data_point.step_in_seq
            if step < WARMUP:
                logits = self.detector.step_all(np.asarray(data_point.state, dtype=np.float32))
                if step >= self.from_row:
                    z = logits[self.lo:self.hi].astype(np.float64)
                    e = np.exp(z - z.max())
                    self.votes += e / e.sum()
                for group in self.groups.values():
                    group.predict(data_point)
                return None
            self.route = self.groups[self.class_group[int(np.argmax(self.votes))]]
        return self.route.predict(data_point)
