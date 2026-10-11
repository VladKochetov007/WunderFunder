"""Fit per-column quantile tables of the 112 raw columns for train_rnn.py --input-act.

    python scripts/build_quantile_table.py --out cache/quantile_table_fold1.npz --groups fit_groups.npy --split train

Uses every ``--stride``-th row of the listed fitting row groups only (inputs, never labels or evaluation components).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_rnn  # noqa: E402
from wnn.data import SequenceFile  # noqa: E402
from wnn.input_act import fit_quantile_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--split", choices=["train", "valid"], default="train")
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--max-groups", type=int, default=2000)
    args = ap.parse_args()
    groups = np.load(args.groups).astype(int).tolist()[: args.max_groups]
    reader = SequenceFile(train_rnn.ROOT / "datasets" / f"{args.split}.parquet")
    rows = np.concatenate([reader.read(g, targets=False, mask=False).features[::args.stride] for g in groups])
    np.savez(args.out, table=fit_quantile_table(rows), groups=len(groups), stride=args.stride)
    print(f"wrote {args.out} from {len(rows)} rows of {len(groups)} sequences")


if __name__ == "__main__":
    main()
