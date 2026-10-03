"""Compare pair_dump outputs: OLD.npz vs NEW.npz bitwise per precision, and NEW q16x vs NEW f32
(max abs, global WP delta on the scored rows).   pair_compare.py OLD.npz NEW.npz OUT.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts"))
import engine_fp16_check as e  # noqa: E402

old, new, out = np.load(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3])
groups = [int(g) for g in new["groups"]]
seqs = list(e.read_sequences(e.VALID, groups))
need = np.stack([s[3] for s in seqs])
bits = lambda a: a.view(np.uint32)  # noqa: E731
report = {"groups": groups, "old_vs_new_bitwise": {}, "parity_vs_f32": {}}
for p in new.files:
    if p in old.files and p != "groups":
        report["old_vs_new_bitwise"][p] = bool(np.array_equal(bits(old[p]), bits(new[p])))
f32 = new["f32"]
wp32 = e.global_wp(f32, seqs)
for p in ("q16x", "q16x_portable", "i16", "f16"):
    if p not in new.files:
        continue
    d = np.abs(new[p] - f32)
    wp = e.global_wp(new[p], seqs)
    report["parity_vs_f32"][p] = {
        "max_abs_all_rows": float(d.max()),
        "max_abs_need_rows": float(d[need].max()),
        "mean_abs_need_rows": float(d[need].mean()),
        "wp": wp, "wp_f32": wp32, "delta_wp": wp - wp32,
    }
if "q16x" in new.files and "q16x_portable" in new.files:
    report["q16x_vs_portable_bitwise"] = bool(np.array_equal(bits(new["q16x"]), bits(new["q16x_portable"])))
for p in ("q16x", "q16x_portable"):
    if p + "_repeat" in new.files:
        report[p + "_repeat_bitwise"] = bool(np.array_equal(bits(new[p]), bits(new[p + "_repeat"])))
out.write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
