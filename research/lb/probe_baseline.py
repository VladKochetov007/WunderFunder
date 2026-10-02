"""Holdout effect of blending the heuristic baseline into the distilled GRU256 student (H-066 probe)."""
import sys
from pathlib import Path
import numpy as np
import importlib.util
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("lbp", ROOT / "research/lb/lb_probe.py")
lbp = importlib.util.module_from_spec(spec); spec.loader.exec_module(lbp)
vt, idx = lbp.holdout_index()
y = np.asarray(vt["y"][idx]); m = np.asarray(vt["mask"][idx]); cls = np.asarray(vt["market_class"][idx])
d = lbp.member_pred("c20261001_final_d256_s3|c20261001_final_d256_s3_gcorrh", idx)
b = lbp.member_pred("baseline", idx)
f = lbp.mix_factor(lbp.PUB143, cls)
base = lbp.class_stats(y, d, m, cls)
for w in (0.0, 0.05, 0.0875 / 0.65, 0.2, 0.3):
    st = lbp.class_stats(y, d + w * b, m, cls)
    print(f"w_baseline/w_student {w:.3f}: holdout {lbp.wp(st):.5f} ({(lbp.wp(st)-lbp.wp(base))*1e4:+.1f}e-4) "
          f"pub143 {(lbp.wp(st, f)-lbp.wp(base, f))*1e4:+.1f}e-4 cls {[round((lbp.wp(st[c:c+1])-lbp.wp(base[c:c+1]))*1e4,1) for c in range(4)]}")
print("corr(student, baseline) holdout rows:", np.corrcoef(d[..., 0][m], b[..., 0][m])[0, 1].round(3))
