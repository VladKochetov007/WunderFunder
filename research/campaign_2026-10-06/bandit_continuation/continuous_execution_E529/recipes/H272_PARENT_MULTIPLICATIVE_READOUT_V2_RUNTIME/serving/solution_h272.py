"""Source-ready scorer adapter for a later reviewed H272 native ZIP.

The parent and readout NPZ members are intentionally absent until a fixed
epoch-1 fit has been admitted and the complete native serving checks pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "native"))
import wnn_native  # noqa: E402


def _load_weights(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in archive.files
        }


class PredictionModel:
    """Native row-causal parent plus the fixed multiplicative contextual head."""

    def __init__(self) -> None:
        parent_path = HERE / "parent_model.npz"
        readout_path = HERE / "multiplicative_head.npz"
        if not parent_path.is_file() or not readout_path.is_file():
            raise FileNotFoundError(
                "H272 ZIP is not assembled: reviewed fixed-epoch parent/head NPZs are required"
            )
        self.model = wnn_native.ConditionalPredictor(
            _load_weights(parent_path),
            _load_weights(readout_path),
            fast_activations=False,
            precision="f32",
        )
        self.predict = self.model.predict
