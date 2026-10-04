"""Recover time adjacency of sequences (V1): tail of one sequence -> head of another.

    scripts/limited .venv/bin/python scripts/sequence_chains.py
Reads the first/last 3 rows of every train and validation sequence, links each tail to its
nearest head (L1 over the 112 features, excluding the a-columns that restart per sequence is
not assumed: all columns used), keeps mutual best matches whose distance is within the range of
consecutive-row distances inside sequences, and builds chains. Writes cache/chains.npz with
split, row_group, next (index of the successor or -1), chain id and position in chain.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


_FILES: dict = {}


def _edge(args):
    split, g = args
    pf = _FILES.get(split)
    if pf is None:  # open once per worker: train metadata alone is ~1 GB
        pf = _FILES[split] = pq.ParquetFile(ROOT / "datasets" / f"{split}.parquet")
    t = pf.read_row_group(g, columns=FEATURE_COLUMNS + ["t0", "t1"], use_threads=False)
    a = np.column_stack([t.column(c).to_numpy() for c in FEATURE_COLUMNS + ["t0", "t1"]]).astype(np.float32)
    return a[:2].copy(), a[-2:].copy(), a[10000:10002].copy()


def main():
    jobs = [("train", g) for g in range(10607) if g not in CORRUPT_TRAIN_ROW_GROUPS] + [("valid", g) for g in range(1873)]
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "6"))) as ex:
        res = list(ex.map(_edge, jobs, chunksize=16))
    head = np.stack([r[0][0, :112] for r in res])
    tail = np.stack([r[1][-1, :112] for r in res])
    mid = np.stack([r[2] for r in res])
    ref = np.abs(mid[:, 1, :112] - mid[:, 0, :112]).sum(1)  # consecutive-row distance inside sequences
    from sklearn.neighbors import NearestNeighbors

    nn_h = NearestNeighbors(n_neighbors=2, metric="manhattan").fit(head)
    d, idx = nn_h.kneighbors(tail)
    nn_t = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(tail)
    _, back = nn_t.kneighbors(head)
    n = len(res)
    nxt = np.full(n, -1)
    thr = np.quantile(ref, 0.99)
    for i in range(n):
        j = idx[i, 0] if idx[i, 0] != i else idx[i, 1]
        dij = d[i, 0] if idx[i, 0] != i else d[i, 1]
        if back[j, 0] == i and dij <= thr:
            nxt[i] = j
    prev = np.full(n, -1)
    prev[nxt[nxt >= 0]] = np.flatnonzero(nxt >= 0)
    chain = np.full(n, -1)
    pos = np.zeros(n, int)
    c = 0
    for i in np.flatnonzero(prev < 0):
        k, p = i, 0
        seen = set()
        while k >= 0 and k not in seen:
            seen.add(k)
            chain[k], pos[k] = c, p
            k, p = nxt[k], p + 1
        c += 1
    split = np.array([s for s, _ in jobs])
    rg = np.array([g for _, g in jobs])
    t_tail = np.stack([r[1][-1, 112:] for r in res])
    t_head = np.stack([r[0][0, 112:] for r in res])
    linked = np.flatnonzero(nxt >= 0)
    same_target = np.mean(np.all(np.isclose(t_tail[linked], t_head[nxt[linked]]), 1))
    rnd = np.random.default_rng(0).permutation(n)
    rand_target = np.mean(np.all(np.isclose(t_tail, t_head[rnd]), 1))
    sizes = np.bincount(chain[chain >= 0])
    print({"sequences": n, "links": int(len(linked)), "chains": int(c), "chain_size_median": float(np.median(sizes)), "chain_size_max": int(sizes.max()),
           "link_dist_median": float(np.median(d[linked, 0])), "within_seq_consecutive_dist_median": float(np.median(ref)), "threshold": float(thr),
           "target_identical_across_link": round(float(same_target), 4), "target_identical_random": round(float(rand_target), 4),
           "mixed_split_links": int(np.sum(split[linked] != split[nxt[linked]]))})
    np.savez(ROOT / "cache" / "chains.npz", split=split, row_group=rg, next=nxt, chain=chain, pos=pos)


if __name__ == "__main__":
    main()
