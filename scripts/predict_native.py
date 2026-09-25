"""Batch validation predictions from a native submission package, one sequence per task.

    scripts/limited .venv/bin/python scripts/predict_native.py submissions/ab9_pow2_h128/pkg ab9
Writes cache/pred_<name>_valid.npy (B, T, 2) float32 in cache row order.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_PRED = None


def _init(pkg):
    global _PRED
    sys.path.insert(0, str(Path(pkg) / "v3"))
    import wnn_native

    with np.load(Path(pkg) / "weights.npz") as w:
        weights = {k: np.ascontiguousarray(w[k], dtype=np.float32) for k in w.files}
    _PRED = wnn_native.Predictor(weights, fast_activations=True)


def _run(g):
    from wnn.data import SequenceFile

    x = SequenceFile(ROOT / "datasets" / "valid.parquet").read(g, targets=False, mask=False).features
    _PRED.reset()
    out = np.empty((len(x), 2), np.float32)
    for i in range(len(x)):
        out[i] = _PRED.step(np.ascontiguousarray(x[i]))
    return out


def main():
    pkg, name = sys.argv[1], sys.argv[2]
    groups = np.load(ROOT / "cache" / "valid_targets.npz")["row_group"]
    pred = np.empty((len(groups), 20_000, 2), np.float32)
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "8")), initializer=_init, initargs=(pkg,)) as ex:
        for i, p in enumerate(ex.map(_run, [int(g) for g in groups], chunksize=4)):
            pred[i] = p
    np.save(ROOT / "cache" / f"pred_{name}_valid.npy", pred)


if __name__ == "__main__":
    main()
