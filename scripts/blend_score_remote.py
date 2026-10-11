"""Score equal-weight (or w=weight) blends of banked fold predictions where they live.

usage: blend_score_remote.py FOLD "nameA+nameB+nameC" ["nameA:2+nameB"] ...   (names without pred_ prefix / _valid suffix)
Official per-target-averaged WP (GlobalAccumulator). Streams in chunks of 40 sequences; no full-array load.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.gmetric import GlobalAccumulator  # noqa: E402

SCORED_FROM_ROW = 99


def main() -> int:
    fold = int(sys.argv[1])
    split = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json").read_text())
    groups = [int(g) for g in split["evaluation"]["valid"]]
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as c:
        position = {int(g): i for i, g in enumerate(c["row_group"])}
        index = np.array(sorted(position[g] for g in groups))
        y = np.asarray(c["y"][index], dtype=np.float32)
        mask = np.asarray(c["mask"][index], dtype=bool)
    mask &= (np.arange(y.shape[1]) >= SCORED_FROM_ROW)[None, :]
    for spec in sys.argv[2:]:
        parts = []
        for item in spec.split("+"):
            name, _, w = item.partition(":")
            parts.append((float(w or 1), np.load(ROOT / f"cache/pred_{name}_valid.npy", mmap_mode="r")))
        total = sum(w for w, _ in parts)
        acc = GlobalAccumulator()
        for s in range(0, len(index), 40):
            part = index[s:s + 40]
            p = sum(w * np.asarray(a[part], dtype=np.float32) for w, a in parts) / total
            acc.add(y[s:s + 40], p, mask[s:s + 40])
        r = acc.result()
        print(f"{r['wp']:.5f} t0 {r['t0']:.5f} t1 {r['t1']:.5f}  {spec}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
