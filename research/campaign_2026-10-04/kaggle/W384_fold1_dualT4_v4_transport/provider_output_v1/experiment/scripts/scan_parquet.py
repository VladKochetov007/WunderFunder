"""Decode every row group; report ones that fail. Read-only integrity scan."""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import pyarrow.parquet as pq


def _one(args):
    path, g = args
    try:
        pq.ParquetFile(path).read_row_group(g, use_threads=False)
        return None
    except Exception as e:  # noqa: BLE001
        return (g, type(e).__name__, str(e)[:120])


def main():
    path = sys.argv[1]
    n = pq.ParquetFile(path).num_row_groups
    with Pool(3) as pool:
        bad = [r for r in pool.imap_unordered(_one, [(path, g) for g in range(n)], chunksize=16) if r]
    print(json.dumps({"file": path, "row_groups": n, "failed": sorted(bad)}))


if __name__ == "__main__":
    main()
