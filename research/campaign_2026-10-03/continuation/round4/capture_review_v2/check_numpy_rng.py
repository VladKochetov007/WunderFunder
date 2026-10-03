"""Read small post-state artifacts one at a time; reject NumPy RNG dtype drift.

Run ONLY on a controller-provided frozen measurement output directory. Does not
load pre-states, raw datasets or CUDA tensors, write actor files, or approve jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    paths = sorted(options.directory.rglob("post.pt")) + sorted(options.directory.rglob("original_default_*.pt"))
    if not paths:
        raise RuntimeError("No complete post/baseline artifacts supplied")
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        rng = state["rng"]["numpy"]
        if type(rng) is not tuple or len(rng) != 5 or rng[0] != "MT19937":
            raise RuntimeError(f"NumPy RNG state grammar differs: {path}")
        values = rng[1]
        if type(values) is not np.ndarray or values.dtype != np.dtype("uint32") or values.shape != (624,):
            raise RuntimeError(f"NumPy RNG dtype/shape differs: {path}")
        del state


if __name__ == "__main__":
    main()
