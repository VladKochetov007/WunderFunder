"""Assemble out-of-fold validation predictions from K-fold runs (train_rnn --valid-fold k).

    scripts/limited .venv/bin/python scripts/assemble_oof.py ab26_tv 4
Reads cache/pred_<prefix>_f<k>_valid.npy (each covers all validation rows) and keeps, for every
sequence, the prediction of the model whose fold held it out. Writes cache/pred_<prefix>_oof_valid.npy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    prefix, k = sys.argv[1], int(sys.argv[2])
    rg = np.load(ROOT / "cache" / "valid_targets.npz")["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz")
    vmask = cf["split"] == "valid"
    fold_map = dict(zip(cf["row_group"][vmask].tolist(), cf["fold"][vmask].tolist()))
    fold = np.array([fold_map[int(g)] for g in rg]) % k
    out = np.lib.format.open_memmap(ROOT / "cache" / f"pred_{prefix}_oof_valid.npy", mode="w+", dtype=np.float32, shape=(len(rg), 20_000, 2))
    for f in range(k):
        p = np.load(ROOT / "cache" / f"pred_{prefix}_f{f}_valid.npy", mmap_mode="r")
        idx = np.flatnonzero(fold == f)
        out[idx] = p[idx]
    out.flush()
    print("folds", np.bincount(fold))


if __name__ == "__main__":
    main()
