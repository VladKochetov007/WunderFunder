"""Compare a verified package's scorer-harness predictions with batch GPU predictions.

    .venv/bin/python scripts/check_package.py VERIFY.npz RUN:COEF [RUN:COEF ...]
VERIFY.npz comes from scripts/verify_submission.py on cache/valid_subset20.parquet; RUN is a run
with cache/pred_<RUN>_valid.npy ('baseline' = the organizer model). Rows < 99 are warm-up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def main():
    v = np.load(sys.argv[1])
    parts = [(m.partition(":")[0], float(m.partition(":")[2])) for m in sys.argv[2:]]
    sub = pq.read_table(ROOT / "cache" / "valid_subset20.parquet", columns=["seq_ix"]).column(0).to_numpy()[::20000]
    allseq = pq.ParquetFile(ROOT / "datasets" / "valid.parquet").read(columns=["seq_ix"]).column(0).to_numpy()[::20000]
    rgs = [int(np.flatnonzero(allseq == s)[0]) for s in sub]
    pos = {int(g): i for i, g in enumerate(np.load(ROOT / "cache" / "valid_targets.npz")["row_group"])}
    arrs = {r: np.load(ROOT / "cache" / f"pred_{r}_valid.npy", mmap_mode="r") for r, _ in parts}
    batch = np.concatenate([sum(c * np.asarray(arrs[r][pos[g]], np.float64) for r, c in parts) for g in rgs])
    p = v["pred"].reshape(-1, 2).astype(np.float64)
    s = np.tile(np.arange(20000), len(rgs)) >= 99
    d = np.abs(p[s] - batch[s]).max(1)
    print("corr", [round(float(np.corrcoef(p[s][:, k], batch[s][:, k])[0, 1]), 6) for k in range(2)],
          "maxdiff", round(float(d.max()), 4), "p99", round(float(np.quantile(d, 0.99)), 4))


if __name__ == "__main__":
    main()
