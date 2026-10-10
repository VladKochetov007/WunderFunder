"""Score banked fold predictions where they live (no 300 MB transfers): official per-target-averaged WP on a fold's evaluation groups.

usage: score_prediction_remote.py FOLD name [name ...]   (names without the pred_ prefix and _valid suffix)
Predictions are (1873, 20000, 2) float32 aligned to the order of cache/valid_targets.npz row groups.
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
    for name in sys.argv[2:]:
        a = np.load(ROOT / f"cache/pred_{name}_valid.npy", mmap_mode="r")
        acc = GlobalAccumulator()
        for s in range(0, len(index), 40):
            part = index[s:s + 40]
            acc.add(y[s:s + 40], np.asarray(a[part], dtype=np.float32), mask[s:s + 40])
        r = acc.result()
        print(f"{name} {r['wp']:.5f} t0 {r['t0']:.5f} t1 {r['t1']:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
