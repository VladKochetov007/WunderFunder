"""Which matrices' half rounding drives the fp16 output error? Rounds subsets of a member's
weights to binary16 in numpy and runs the fp32 engine (no F16 kernel involved)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import engine_fp16_check as c

native = c.load_native(Path(sys.argv[1]))
w = c.load_npz(c.H128[0])
seqs = list(c.read_sequences(c.VALID, [0]))
r16 = lambda a: a.astype(np.float16).astype(np.float32)

def run(weights):
    return c.run_direct(c.ensemble(native, [weights], "f32"), seqs)

base = run(w)
def variant(keys, fold=False):
    v = dict(w)
    for k in keys:
        if k == "inp.weight" and fold:
            v[k] = (r16(w[k] / w["std"]) * w["std"]).astype(np.float32)
        else:
            v[k] = r16(w[k])
    return v

gru = [k for k in w if k.startswith("gru.weight")]
for name, keys in [("inp(folded)", ["inp.weight"]), ("gru_l0", [k for k in gru if k.endswith("l0")]),
                   ("gru_l1", [k for k in gru if k.endswith("l1")]), ("head.0", ["head.0.weight"]),
                   ("gru_all", gru), ("gru+head0", gru + ["head.0.weight"]),
                   ("all", ["inp.weight"] + gru + ["head.0.weight"])]:
    d = np.abs(run(variant(keys, fold=True)) - base)
    print(f"{name:12s} max {d.max():.2e} mean {d.mean():.2e} p99.9 {np.quantile(d, 0.999):.2e}")
full = c.run_direct(c.ensemble(native, [w], "f16"), seqs)
d = np.abs(full - base); print(f"{'engine f16':12s} max {d.max():.2e} mean {d.mean():.2e}")

# Per-row int16 fixed point (scale = max|row| / 32767), input projection per column of W/std.
def q16_rows(a):
    s = np.abs(a).max(axis=1, keepdims=True) / 32767.0
    s[s == 0] = 1.0
    return (np.round(a / s) * s).astype(np.float32)
v = dict(w)
for k in gru + ["head.0.weight"]:
    v[k] = q16_rows(w[k])
folded = w["inp.weight"] / w["std"]
v["inp.weight"] = (q16_rows(folded.T).T * w["std"]).astype(np.float32)
d = np.abs(run(v) - base); print(f"{'int16 rows':12s} max {d.max():.2e} mean {d.mean():.2e} p99.9 {np.quantile(d, 0.999):.2e}")
