"""Black-box checks of a packaged solution: parity vs cached batch predictions, reset, edge cases.

usage: check_pkg.py PKG_DIR {keep|routed_all} [--governor]
"""
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS  # noqa: E402

pkg, kind = Path(sys.argv[1]).resolve(), sys.argv[2]
GOV = "--governor" in sys.argv


@dataclass(frozen=True)
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def load_solution():
    spec = importlib.util.spec_from_file_location("sol", pkg / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sol = load_solution()
d = np.load(ROOT / "cache/valid_targets.npz")
rg_all, mc = d["row_group"], d["market_class"]
pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")  # memory_map defaults to False

C = {
    "keep": [["ab67_final_since_s4_gcorr", "ab67_final_since_s3_gcorr", "ab84_final_ch500_s5_gclp", "ab84_final_onorm_s5_gclp"]] * 4,
    "routed_all": [["ab85_final_ema6_s3_gclpall", "ab85_final_ema6_s4_gclpall", "ab85_final_ch500_s5_gclpall", "ab85_final_onorm_s5_gclpall"]] * 3
    + [["ab85_final_since_s4_gclpall", "ab85_final_since_s3_gclpall", "ab85_final_ch500_s5_gclpall", "ab85_final_onorm_s5_gclpall"]],
}[kind]


def features(i):
    t = pf.read_row_group(int(rg_all[i]), columns=["seq_ix", *FEATURE_COLUMNS], use_threads=False)
    x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    return int(t["seq_ix"][0].as_py()), x


def expected(i):
    return np.mean([np.asarray(np.load(ROOT / f"cache/pred_{n}_valid.npy", mmap_mode="r")[i], np.float64) for n in C[mc[i]]], 0)


def run(model, seq, x, need_from=99):
    out = np.full((len(x), 2), np.nan, np.float32)
    for t in range(len(x)):
        v = model.predict(DataPoint(seq, t, t >= need_from, x[t]))
        if t < need_from:
            assert v is None, f"non-None during warm-up at {t}"
        else:
            v = np.asarray(v, dtype=np.float32)
            assert v.shape == (2,) and np.isfinite(v).all(), (t, v)
            out[t] = v
    return out


# pick sequences: two holdout-ish ones per branch (class 3 and not 3)
from wnn.splits import valid_row_groups  # noqa: E402

_, held = valid_row_groups(ROOT / "datasets/valid.parquet")
hidx = np.flatnonzero(np.isin(rg_all, held))
picks = [hidx[mc[hidx] == 3][0], hidx[mc[hidx] == 0][0], hidx[mc[hidx] == 2][0]]
report = []

# 1. parity + causality: identical results when the sequence runs after another one (reset works)
m = sol.PredictionModel()
seqs = {i: features(i) for i in picks}
t0 = time.perf_counter()
outs = {}
for i in picks:
    s, x = seqs[i]
    outs[i] = run(m, s, x)
el = time.perf_counter() - t0
rows = sum(len(seqs[i][1]) for i in picks)
report.append(f"per_row_us={el / rows * 1e6:.1f} (this machine, {len(picks)} seqs, level={sol._LEVEL})")
for i in picks:
    e = expected(i)[99:]
    o = outs[i][99:].astype(np.float64)
    corr = [np.corrcoef(o[:, k], e[:, k])[0, 1] for k in range(2)]
    report.append(f"parity idx={i} class={mc[i]}: corr={corr[0]:.6f},{corr[1]:.6f} maxabs={np.abs(o - e).max():.2e}")

# fresh model on the last pick only: must equal the in-stream result bit for bit
m2 = sol.PredictionModel()
s, x = seqs[picks[-1]]
fresh = run(m2, s, x)
report.append(f"reset: fresh-vs-after-other-seqs max diff={np.nanmax(np.abs(fresh - outs[picks[-1]])):.3e}")

# 2. causality: perturb future rows, the prefix must not change
m3 = sol.PredictionModel()
x2 = x.copy()
x2[5000:] = x2[5000:] * 1.37 + 0.5
pert = run(m3, s, x2)
report.append(f"causality: max diff on rows<5000 after perturbing rows>=5000: {np.nanmax(np.abs(pert[:5000] - fresh[:5000])):.3e}; rows>=5000 differ: {np.nanmax(np.abs(pert[5000:] - fresh[5000:])) > 0}")

# 3. edge cases
m4 = sol.PredictionModel()
xn = x[:3000].copy()
xn[0, :] = np.nan
xn[500, 3] = np.inf
xn[501, :50] = np.nan
xn[502, 7] = -np.inf
try:
    o = run(m4, s + 10_000_000, xn)
    report.append(f"nan/inf rows: all required outputs finite: {np.isfinite(o[99:]).all()}")
except Exception as ex:  # noqa: BLE001
    report.append(f"nan/inf rows: FAILED {ex!r}")
try:
    m5 = sol.PredictionModel()
    o = run(m5, 7, x[:40], need_from=0)  # first row needs a prediction, sequence of 40 rows
    o2 = run(m5, 8, x[:10], need_from=0)
    report.append(f"need_prediction from row 0 + short sequences: finite={np.isfinite(o).all() and np.isfinite(o2).all()}")
except Exception as ex:  # noqa: BLE001
    report.append(f"first-row need_prediction: FAILED {ex!r}")
try:
    m6 = sol.PredictionModel()
    big = x[:300].copy()
    big[200:] = 1e20
    o = run(m6, 9, big)
    report.append(f"extreme finite inputs (1e20): finite={np.isfinite(o[99:]).all()} const_after={np.ptp(o[210:], 0)}")
except Exception as ex:  # noqa: BLE001
    report.append(f"extreme inputs: FAILED {ex!r}")
try:
    m7 = sol.PredictionModel()
    o = run(m7, 11, x[:200].astype(np.float64))  # float64 state
    report.append(f"float64 state accepted: {np.isfinite(o[99:]).all()}")
except Exception as ex:  # noqa: BLE001
    report.append(f"float64 state: FAILED {ex!r}")

# returned arrays must not alias an internal buffer
m8 = sol.PredictionModel()
a = m8.predict(DataPoint(1, 0, True, x[0]))
a_copy = np.array(a, copy=True)
for t in range(1, 50):
    m8.predict(DataPoint(1, t, True, x[t]))
report.append(f"no aliasing of returned array: {np.array_equal(a, a_copy)}")

if GOV:
    # governor under a tiny budget: members dropped only at sequence boundaries, outputs finite
    sol.BUDGET_S = 1.0
    sol.EXPECTED_ROWS = 10 * 20_000
    mg = sol.PredictionModel()
    for r in range(4):
        o = run(mg, 100 + r, x)
        report.append(f"governor seq {r}: k={getattr(mg, 'k', None) or getattr(mg, 'k_max', None)} finite={np.isfinite(o[99:]).all()}")

print("\n".join(report))
