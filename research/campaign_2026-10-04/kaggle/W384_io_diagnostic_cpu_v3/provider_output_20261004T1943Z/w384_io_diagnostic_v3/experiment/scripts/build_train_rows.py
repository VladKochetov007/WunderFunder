"""Row sample from train sequences for non-recurrent members (GBM on heuristic features).

    scripts/limited .venv/bin/python scripts/build_train_rows.py 3000 0.03
Samples a fraction of required rows from N random train sequences and stores heuristic
features, raw features, targets, the market class and the causal scoring propensity p(x)
(prop_v1 on CPU) to cache/heur_train.npz.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, SequenceFile, market_class
from wnn.heuristics import NAMES, heuristic_features

ROOT = Path(__file__).resolve().parents[1]
_PROP = None


def _prop():
    global _PROP
    if _PROP is None:
        import torch

        from wnn.propensity import load_propensity

        torch.set_num_threads(1)
        _PROP = load_propensity(ROOT / "artifacts" / "runs" / "prop_v1" / "model.pt", "cpu")
    return _PROP


def _run(args):
    g, rate = args
    import torch

    from wnn.features import event_features
    from wnn.propensity import propensity

    s = SequenceFile(ROOT / "datasets" / "train.parquet").read(g)
    f = heuristic_features(s.features)
    inp = torch.from_numpy(np.concatenate([s.features, event_features(s.features)], 1))[None]
    p = propensity(_prop(), inp)[0].numpy()
    rng = np.random.default_rng(g)
    r = np.flatnonzero((np.arange(len(f)) >= 99) & (rng.random(len(f)) < rate))
    return f[r], s.features[r], s.targets[r], p[r].astype(np.float32), np.full(len(r), market_class(s.targets), np.int8), np.full(len(r), g, np.int32)


def main():
    n, rate = int(sys.argv[1]), float(sys.argv[2])
    usable = [g for g in range(10607) if g not in CORRUPT_TRAIN_ROW_GROUPS]
    groups = np.random.default_rng(7).choice(usable, n, replace=False)
    parts = []
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "8"))) as ex:
        for res in ex.map(_run, [(int(g), rate) for g in groups], chunksize=8):
            parts.append(res)
    X, raw, y, p, c, g = (np.concatenate([q[i] for q in parts]) for i in range(6))
    np.savez(ROOT / "cache" / "heur_train.npz", X=X, raw=raw, y=y, prop=p, cls=c, group=g, names=np.array(NAMES))
    print(len(y), "rows from", n, "sequences")


if __name__ == "__main__":
    main()
