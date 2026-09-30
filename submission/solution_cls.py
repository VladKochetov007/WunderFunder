"""C++ GRU ensemble with an online market-class detector and per-class output scales (LB shift probes).

members.json as in solution_cpp.py; classes.json:
  {"detector": ["det_a.npz", "det_b.npz"], "from_row": 50, "decide_row": 99, "scale": {"3": 1.4}}

det_a / det_b are one aux-struct GRU exported twice with the head rows of the market-class logits
(0, 1) and (2, 3) in the target slots, so the engine's two outputs are two class logits each. They are
stepped only until decide_row (or the first row that needs a prediction); class probabilities are
averaged from from_row on, the argmax class is kept for the rest of the sequence, and every output of
that sequence is multiplied by scale[class] (default 1). Everything uses rows up to the current one.
Governor as in solution_cpp.py (the detector cost is a fixed ~100 rows per sequence).
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
import wnn_cpp  # noqa: E402


def _load(name):
    with np.load(HERE / name) as weights:
        return {key: np.ascontiguousarray(weights[key], dtype=np.float32) for key in weights.files}


def _log(message):
    print(f"[wnn] {message}", file=sys.stderr, flush=True)


class PredictionModel:
    def __init__(self):
        members = json.loads((HERE / "members.json").read_text())
        cfg = json.loads((HERE / "classes.json").read_text())
        self.weights = [_load(member["weights"]) for member in members]
        self.coefs = [float(member["coef"]) for member in members]
        self.total_coef = sum(self.coefs)
        self.k = len(members)
        self._build()
        self.detectors = [wnn_cpp.Predictor(_load(name), fast_activations=True) for name in cfg["detector"]]
        self.from_row, self.decide_row = int(cfg["from_row"]), int(cfg["decide_row"])
        self.scales = {int(c): float(s) for c, s in cfg["scale"].items()}
        self.probs = np.zeros(4)
        self.scale = None
        self.step_in_seq = 0
        self.class_counts = [0, 0, 0, 0]
        self.seq_ix = None
        self.seqs = 0
        self.rows = 0
        self.t_start = None
        self.win_t0 = None  # measurement window starts after the first (cold) sequence
        self.win_rows = 0
        self.win_step = 0.0
        _log(f"level={_LEVEL} members={self.k} half={wnn_cpp.HALF_SUPPORTED} scales={self.scales}")

    def _build(self):
        coefs = self.coefs[: self.k]
        scale = self.total_coef / sum(coefs)
        self.model = wnn_cpp.Ensemble(
            self.weights[: self.k],
            [float(c * scale) for c in coefs],
            fast_activations=True,
            half_weights=True,  # fp16 dense weights on v3 builds (ignored on v2/avx2)
        )

    def _govern(self, now):
        """Drop members if the measured cost would overrun the budget for the remaining rows."""
        self.seqs += 1
        if self.seqs == 1:
            self.win_t0, self.win_rows, self.win_step = now, self.rows, 0.0
            return
        rows = self.rows - self.win_rows
        if self.k == 1 or rows < 20_000:
            return
        remaining_rows = EXPECTED_ROWS - self.rows
        if remaining_rows <= 0:
            return
        allowed = (BUDGET_S - (now - self.t_start)) / remaining_rows
        per_row = (now - self.win_t0) / rows
        if per_row <= allowed:
            return
        per_member = self.win_step / rows / self.k
        overhead = per_row - per_member * self.k
        k_new = int((allowed - overhead) // per_member) if per_member > 0 else 1
        k_new = max(1, min(self.k - 1, k_new))
        _log(
            f"governor: rows={self.rows} elapsed={now - self.t_start:.0f}s per_row={per_row * 1e6:.1f}us "
            f"allowed={allowed * 1e6:.1f}us members {self.k}->{k_new}"
        )
        self.k = k_new
        self._build()
        self.win_t0, self.win_rows, self.win_step = now, self.rows, 0.0

    def _decide(self):
        c = int(np.argmax(self.probs))
        self.class_counts[c] += 1
        self.scale = self.scales.get(c, 1.0)

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            now = time.perf_counter()
            if self.t_start is None:
                self.t_start = now
            else:
                self._govern(now)
                if self.seqs % 200 == 0:
                    _log(f"seqs={self.seqs} classes={self.class_counts}")
            self.seq_ix = data_point.seq_ix
            self.model.reset()
            for det in self.detectors:
                det.reset()
            self.probs[:] = 0.0
            self.scale = None
            self.step_in_seq = 0
        row = np.ascontiguousarray(data_point.state, dtype=np.float32)
        t0 = time.perf_counter()
        output = self.model.step(row)
        self.win_step += time.perf_counter() - t0
        if self.scale is None:
            logits = np.concatenate([det.step(row) for det in self.detectors]).astype(np.float64)
            if self.step_in_seq >= self.from_row or not self.probs.any():
                e = np.exp(logits - logits.max())
                if self.step_in_seq >= self.from_row:
                    self.probs += e / e.sum()
                else:
                    self.probs[:] = e / e.sum()
            if data_point.need_prediction or self.step_in_seq + 1 >= self.decide_row:
                self._decide()
        self.rows += 1
        self.step_in_seq += 1
        if not data_point.need_prediction:
            return None
        return output * self.scale
