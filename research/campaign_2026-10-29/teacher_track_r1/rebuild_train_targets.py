"""Rebuild cache/train_targets.npz in the chronological scratch root with all 10,607 groups.

The shared Sep-26 cache omits row group 528 (undecodable in the pre-Oct-3 download, E-035).
The current train.parquet decodes it, the PR49 fit used it (10,325 = 10,607 - 282 excluded),
and the PR49 full-train export validator requires it. Same arrays as PR49 cache_targets.py.
Usage: rebuild_train_targets.py SCRATCH_ROOT PR49_SRC
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root, src = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, src)
from wnn.data import market_class  # noqa: E402

out = root / "cache" / "train_targets.npz"
if out.exists() and out.stat().st_nlink > 1:
    out.unlink()  # break the hardlink; never truncate the shared inode
pf = pq.ParquetFile(root / "datasets" / "train.parquet")
groups = list(range(pf.num_row_groups))
y = np.empty((len(groups), 20_000, 2), np.float32)
seq = np.empty(len(groups), np.int64)
for i, g in enumerate(groups):
    t = pf.read_row_group(g, columns=["seq_ix", "t0", "t1"], use_threads=False)
    y[i, :, 0] = t["t0"].to_numpy()
    y[i, :, 1] = t["t1"].to_numpy()
    seq[i] = t["seq_ix"][0].as_py()
cls = np.array([market_class(yy) for yy in y])
np.savez(out, y=y, mask=np.zeros((len(groups), 20_000), bool), seq_ix=seq, row_group=np.array(groups), market_class=cls)
print(out, y.shape, np.bincount(cls + 1))
