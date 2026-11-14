"""Zero-fit ceiling for per-since-band affine readout calibration (Freebuff R3 B5, PR #93 C04 family).

Two-fold cross-fit over 400 valid sequences: fit |y|-weighted affine maps per since band (and per band x market
class) on one half, apply to the other, score pooled official WP. Reports WP by band for the raw predictions.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/home/vlad/development/wunder/research_tmp/chrono_root")
SRC = Path("/home/vlad/development/wunder/research_tmp/pr49_wt")
sys.path[:0] = [str(SRC / "src"), str(SRC)]
from wnn.features import FEATURE_NAMES, event_features  # noqa: E402
from wnn.gmetric import global_wp  # noqa: E402

run, n = sys.argv[1], 400
with np.load(ROOT / "cache/valid_targets.npz") as z:
    pick = np.sort(np.random.default_rng(0).choice(len(z["row_group"]), n, replace=False))
    y, mask, groups, cls = z["y"][pick].astype(np.float32), z["mask"][pick], z["row_group"][pick], z["market_class"][pick]
side = json.loads((ROOT / f"cache/pred_{run}_valid.json").read_text())
rg = side["row_groups"]
pos = {int(g): i for i, g in enumerate(json.loads(rg) if isinstance(rg, str) else rg)}
arr = np.load(ROOT / f"cache/pred_{run}_valid.npy", mmap_mode="r")
p = np.stack([np.asarray(arr[pos[int(g)]], np.float32) for g in groups])
pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
feat_cols = [c for c in pf.schema_arrow.names if c not in ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")]
k = FEATURE_NAMES.index("since_i0px")
since = np.zeros(y.shape[:2], np.float32)
for i, g in enumerate(groups):
    t = pf.read_row_group(int(g), columns=feat_cols, use_threads=False)
    x = np.column_stack([t[c].to_numpy() for c in feat_cols]).astype(np.float32)
    since[i] = np.expm1(event_features(x)[:, k])
band = np.digitize(since, [8, 32, 128])
ones = lambda m: np.ones(int(m.sum()), bool)  # noqa: E731
wp = lambda pp, m: float(global_wp(y[m], pp[m], ones(m))["wp"])  # noqa: E731
print("sequences", n, "scored rows", int(mask.sum()), "raw WP", round(wp(p, mask), 6))
for b, name in enumerate(["since<8", "8-32", "32-128", ">=128"]):
    m = mask & (band == b)
    wsh = np.abs(np.clip(y[m], -2, 2)).sum() / np.abs(np.clip(y[mask], -2, 2)).sum()
    print(f"  band {name}: rows {m.mean() / mask.mean():.3f} |y|-mass {wsh:.3f} WP {wp(p, m):.4f}")


def crossfit(key):
    out = p.copy()
    fold = np.arange(n) % 2
    for f in (0, 1):
        tr, te = (fold != f)[:, None] & mask, (fold == f)[:, None] & mask
        for v in np.unique(key[mask]):
            a, b = tr & (key == v), te & (key == v)
            if a.sum() < 1000 or b.sum() == 0:
                continue
            for t in range(2):
                w = np.abs(np.clip(y[a][:, t], -2, 2)).astype(np.float64)
                X = np.column_stack([p[a][:, t], np.ones(int(a.sum()))]).astype(np.float64)
                coef = np.linalg.lstsq(X * np.sqrt(w)[:, None], np.clip(y[a][:, t], -2, 2) * np.sqrt(w), rcond=None)[0]
                out[..., t][b] = coef[0] * p[..., t][b] + coef[1]
    return wp(out, mask)


print("cross-fit affine by since band      :", round(crossfit(band), 6))
print("cross-fit affine by band x class    :", round(crossfit(band * 4 + cls[:, None]), 6))
print("cross-fit affine by class only      :", round(crossfit(np.broadcast_to(cls[:, None], band.shape)), 6))
