"""Cache targets, scoring mask and market class of a labelled parquet file.

    scripts/limited .venv/bin/python scripts/cache_targets.py valid
Writes cache/<split>_targets.npz: y (B, T, 2) float32, mask (B, T) bool (all
False for train), seq_ix, market_class. Reads only the needed columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.chronological_source_index import OFFICIAL_SOURCE_PINS, build_source_index
from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, market_class

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "valid"))
    parser.add_argument("limit", nargs="?", type=int)
    parser.add_argument(
        "--source-qualified",
        action="store_true",
        help="build the official thin index after checking the explicit source pin",
    )
    parser.add_argument("--expected-bytes", type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-row-groups", type=int)
    args = parser.parse_args()
    if args.source_qualified:
        if args.limit is not None:
            parser.error("limit cannot be used with --source-qualified")
        pin = OFFICIAL_SOURCE_PINS[args.split]
        supplied = (args.expected_bytes, args.expected_sha256, args.expected_row_groups)
        required = (pin.byte_count, pin.sha256, pin.row_group_count)
        if supplied != required:
            parser.error(
                "--source-qualified requires the exact official bytes, SHA-256, and groups"
            )
        output = build_source_index(
            ROOT,
            args.split,
            expected_bytes=pin.byte_count,
            expected_sha256=pin.sha256,
            expected_row_groups=pin.row_group_count,
        )
        print(output)
        return
    if any(
        value is not None
        for value in (args.expected_bytes, args.expected_sha256, args.expected_row_groups)
    ):
        parser.error("source identity options require --source-qualified")

    split = args.split
    limit = args.limit
    pf = pq.ParquetFile(ROOT / "datasets" / f"{split}.parquet")
    has_mask = "is_scored" in pf.schema_arrow.names
    groups = [
        g for g in range(pf.num_row_groups) if split != "train" or g not in CORRUPT_TRAIN_ROW_GROUPS
    ]
    if limit:
        groups = groups[:limit]
    cols = ["seq_ix", "t0", "t1"] + (["is_scored"] if has_mask else [])
    y = np.empty((len(groups), 20_000, 2), np.float32)
    m = np.zeros((len(groups), 20_000), bool)
    seq = np.empty(len(groups), np.int64)
    for i, g in enumerate(groups):
        t = pf.read_row_group(g, columns=cols, use_threads=False)
        y[i, :, 0] = t["t0"].to_numpy()
        y[i, :, 1] = t["t1"].to_numpy()
        if has_mask:
            m[i] = t["is_scored"].to_numpy()
        seq[i] = t["seq_ix"][0].as_py()
    cls = np.array([market_class(yy) for yy in y])
    np.savez(
        ROOT / "cache" / f"{split}_targets.npz",
        y=y,
        mask=m,
        seq_ix=seq,
        row_group=np.array(groups),
        market_class=cls,
    )
    print(split, y.shape, np.bincount(cls + 1))


if __name__ == "__main__":
    main()
