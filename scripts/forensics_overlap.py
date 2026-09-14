"""Data forensics: sequence boundary adjacency/overlap across train+valid, sentinel prevalence.

Reads only the first/last K rows of every sequence. Parallel over row groups.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

from wnn.data import FEATURE_COLUMNS

K = 50
SENTINEL = np.float32(-5.19934)


def _hashes(rows):
    return [hashlib.blake2b(r.tobytes(), digest_size=12).hexdigest() for r in rows]


def _one(args):
    path, g = args
    t = pq.ParquetFile(path).read_row_group(g, columns=["seq_ix", *FEATURE_COLUMNS], use_threads=False)
    x = np.column_stack([t.column(c).to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    sent = np.isclose(x, SENTINEL, atol=1e-4).mean(axis=0)
    return (int(t.column("seq_ix")[0].as_py()), _hashes(x[:K]), _hashes(x[-K:]), sent)


def main():
    jobs = [(f"datasets/{s}.parquet", g) for s, n in (("train", 10607), ("valid", 1873)) for g in range(n)]
    with Pool(6) as pool:
        res = pool.map(_one, jobs, chunksize=32)
    split = ["train"] * 10607 + ["valid"] * 1873
    head_index = defaultdict(list)  # hash of a head row -> (seq idx, row offset)
    for i, (_, head, _, _) in enumerate(res):
        for off, h in enumerate(head):
            head_index[h].append((i, off))
    links = []
    for i, (_, _, tail, _) in enumerate(res):
        for off, h in enumerate(tail):
            for j, hoff in head_index.get(h, []):
                if j != i:
                    links.append((i, j, off, hoff))
    pairs = defaultdict(int)
    for i, j, _, _ in links:
        pairs[(i, j)] += 1
    kinds = defaultdict(int)
    for (i, j), cnt in pairs.items():
        kinds[f"{split[i]}->{split[j]}"] += 1
    sent = np.stack([r[3] for r in res])
    cols = np.flatnonzero(sent.mean(0) > 0)
    out = {
        "sequences": len(res),
        "linked_pairs": len(pairs),
        "linked_pairs_by_split": dict(kinds),
        "matching_rows_per_pair_median": float(np.median(list(pairs.values()))) if pairs else 0,
        "within_head_duplicate_hashes": int(sum(len(v) > 1 for v in head_index.values())),
        "sentinel_columns": {FEATURE_COLUMNS[c]: round(float(sent[:, c].mean()), 4) for c in cols},
        "seqs_with_any_sentinel_frac": round(float((sent.sum(1) > 0).mean()), 4),
    }
    print(json.dumps(out, indent=1))
    if pairs:
        ex = sorted(pairs.items(), key=lambda kv: -kv[1])[:5]
        for (i, j), cnt in ex:
            print("example", split[i], res[i][0], "->", split[j], res[j][0], "matching rows", cnt)


if __name__ == "__main__":
    main()
