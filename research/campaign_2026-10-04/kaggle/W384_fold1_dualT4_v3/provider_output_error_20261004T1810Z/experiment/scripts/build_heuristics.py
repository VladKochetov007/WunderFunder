"""Explanation dataset: heuristic features on scored rows plus a sample of unscored rows.

    scripts/limited .venv/bin/python scripts/build_heuristics.py valid
Writes cache/heur_<split>.npz: X (N, K) float32 heuristic features, raw (N, 112) float32,
seq (cache row index), step, scored (bool), names. Unscored rows are sampled at 3% (seeded)
so the scoring-mask process can be explained too.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wnn.data import SequenceFile
from wnn.heuristics import NAMES, heuristic_features

ROOT = Path(__file__).resolve().parents[1]
UNSCORED_RATE = 0.03


def _run(args):
    i, g, split = args
    s = SequenceFile(ROOT / "datasets" / f"{split}.parquet").read(g)
    f = heuristic_features(s.features)
    rng = np.random.default_rng(1000 + i)
    scored = s.scored & (np.arange(len(f)) >= 99)
    pick = scored | ((rng.random(len(f)) < UNSCORED_RATE) & ~scored & (np.arange(len(f)) >= 99))
    r = np.flatnonzero(pick)
    return i, r, f[r], s.features[r], scored[r]


def main():
    split = sys.argv[1]
    groups = np.load(ROOT / "cache" / f"{split}_targets.npz")["row_group"]
    parts = []
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "6"))) as ex:
        for res in ex.map(_run, [(i, int(g), split) for i, g in enumerate(groups)], chunksize=8):
            parts.append(res)
    parts.sort(key=lambda p: p[0])
    seq = np.concatenate([np.full(len(p[1]), p[0], np.int32) for p in parts])
    np.savez(ROOT / "cache" / f"heur_{split}.npz", X=np.concatenate([p[2] for p in parts]), raw=np.concatenate([p[3] for p in parts]),
             seq=seq, step=np.concatenate([p[1] for p in parts]).astype(np.int32), scored=np.concatenate([p[4] for p in parts]),
             names=np.array(NAMES))
    print(len(seq), "rows")


if __name__ == "__main__":
    main()
