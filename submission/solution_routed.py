"""Market-class-routed native GRU ensembles with a runtime governor.

routing.json:
  {"groups": {"a": [{"weights": "w0.npz", "coef": 0.25}, ...], "b": [...]},
   "class_group": {"0": "a", "1": "a", "2": "a", "3": "b"},
   "detector": {"weights": "det.npz", "lo": 17, "hi": 21, "from_row": 50}}

During the warm-up rows every group ensemble and the class detector (a member with its full
head, whose auxiliary market-class logits sit at [lo, hi)) are stepped; class probabilities are
averaged from `from_row` on. At the first row that needs a prediction the class decides the group,
and only that group is stepped for the rest of the sequence. Everything uses rows up to the
current one only; state resets at every new seq_ix.

CPU level and governor follow solution_ensemble.py: every level runs every member; members are
dropped from the end of each group only if the measured rate would overrun the budget.
"""

import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
_V3_FLAGS = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}
_AVX2_FLAGS = {"avx", "avx2", "fma"}
EXPECTED_ROWS = float(os.environ.get("WNN_EXPECTED_ROWS", 39_400_000))
BUDGET_S = float(os.environ.get("WNN_BUDGET_S", 52 * 60))


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

_CACHE = {}


def _load(name):
    if name not in _CACHE:
        with np.load(HERE / name) as weights:
            _CACHE[name] = {k: np.ascontiguousarray(weights[k], dtype=np.float32) for k in weights.files}
    return _CACHE[name]


def _log(message):
    print(f"[wnn] {message}", file=sys.stderr, flush=True)


class _Group:
    def __init__(self, members):
        self.weights = [_load(m["weights"]) for m in members]
        self.coefs = [float(m["coef"]) for m in members]
        self.total = sum(self.coefs)
        self.k = len(members)
        self.build()

    def build(self):
        coefs = self.coefs[: self.k]
        scale = self.total / sum(coefs)
        self.model = wnn_native.EnsemblePredictor(
            self.weights[: self.k], [np.float32(c * scale) for c in coefs], fast_activations=True
        )


class PredictionModel:
    def __init__(self):
        cfg = json.loads((HERE / "routing.json").read_text())
        self.groups = {name: _Group(members) for name, members in cfg["groups"].items()}
        self.class_group = {int(c): g for c, g in cfg["class_group"].items()}
        det = cfg["detector"]
        self.detector = wnn_native.Predictor(_load(det["weights"]), fast_activations=True)
        self.lo, self.hi, self.from_row = int(det["lo"]), int(det["hi"]), int(det["from_row"])
        self.k_max = max(g.k for g in self.groups.values())
        self.seq_ix = None
        self.active = None
        self.probs = np.zeros(self.hi - self.lo)
        self.step_in_seq = 0
        self.seqs = self.rows = 0
        self.t_start = self.win_t0 = None
        self.win_rows = 0
        self.win_step = 0.0
        _log(f"level={_LEVEL} groups={ {n: g.k for n, g in self.groups.items()} }")

    def _govern(self, now):
        self.seqs += 1
        if self.seqs == 1:
            self.win_t0, self.win_rows, self.win_step = now, self.rows, 0.0
            return
        rows = self.rows - self.win_rows
        k = self.k_max
        if k == 1 or rows < 20_000:
            return
        remaining = EXPECTED_ROWS - self.rows
        if remaining <= 0:
            return
        allowed = (BUDGET_S - (now - self.t_start)) / remaining
        per_row = (now - self.win_t0) / rows
        if per_row <= allowed:
            return
        per_member = self.win_step / rows / k
        overhead = per_row - per_member * k
        k_new = int((allowed - overhead) // per_member) if per_member > 0 else 1
        k_new = max(1, min(k - 1, k_new))
        _log(f"governor: rows={self.rows} per_row={per_row * 1e6:.1f}us allowed={allowed * 1e6:.1f}us "
             f"members {k}->{k_new}")
        self.k_max = k_new
        for group in self.groups.values():
            if group.k > k_new:
                group.k = k_new
                group.build()
        self.win_t0, self.win_rows, self.win_step = now, self.rows, 0.0

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            now = time.perf_counter()
            if self.t_start is None:
                self.t_start = now
            else:
                self._govern(now)
            self.seq_ix = data_point.seq_ix
            for group in self.groups.values():
                group.model.reset()
            self.detector.reset()
            self.active = None
            self.probs[:] = 0.0
            self.step_in_seq = 0
        row = np.ascontiguousarray(data_point.state, dtype=np.float32)
        t0 = time.perf_counter()
        if self.active is None:
            logits = np.asarray(self.detector.step_all(row)[self.lo : self.hi], dtype=np.float64)
            if self.step_in_seq >= self.from_row or not self.probs.any():
                e = np.exp(logits - logits.max())
                if self.step_in_seq >= self.from_row:
                    self.probs += e / e.sum()
                else:
                    self.probs[:] = e / e.sum()
            outs = {name: g.model.step(row) for name, g in self.groups.items()}
            if data_point.need_prediction:
                name = self.class_group.get(int(np.argmax(self.probs)), next(iter(self.groups)))
                self.active = self.groups[name]
                output = outs[name]
            else:
                output = None
        else:
            output = self.active.model.step(row)
        self.win_step += time.perf_counter() - t0
        self.rows += 1
        self.step_in_seq += 1
        if not data_point.need_prediction:
            return None
        return output
