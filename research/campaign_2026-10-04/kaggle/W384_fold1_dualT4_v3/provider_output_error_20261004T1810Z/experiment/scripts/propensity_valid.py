"""Causal scoring propensity p_hat(x) (prop_v1) on every validation row, CPU.

    scripts/limited .venv/bin/python scripts/propensity_valid.py
Writes cache/prop_v1_valid.npy (B, T) float16 in cache row order.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_M = None


def _run(g):
    global _M
    import torch

    from wnn.data import SequenceFile
    from wnn.features import event_features
    from wnn.propensity import load_propensity, propensity

    if _M is None:
        torch.set_num_threads(1)
        _M = load_propensity(ROOT / "artifacts" / "runs" / "prop_v1" / "model.pt", "cpu")
    x = SequenceFile(ROOT / "datasets" / "valid.parquet").read(g, targets=False, mask=False).features
    inp = torch.from_numpy(np.concatenate([x, event_features(x)], 1))[None]
    return propensity(_M, inp)[0].numpy().astype(np.float16)


def main():
    rg = np.load(ROOT / "cache" / "valid_targets.npz")["row_group"]
    out = np.lib.format.open_memmap(ROOT / "cache" / "prop_v1_valid.npy", mode="w+", dtype=np.float16, shape=(len(rg), 20_000))
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "2"))) as ex:
        for i, p in enumerate(ex.map(_run, [int(g) for g in rg], chunksize=8)):
            out[i] = p
    out.flush()


if __name__ == "__main__":
    main()
