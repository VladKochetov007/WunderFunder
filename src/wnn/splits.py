"""Frozen evaluation splits (research/contract.yaml).

The holdout is defined by a hash of ``seq_ix`` so it does not depend on file
order or on any experiment's permutation seed. Nothing may be fitted, tuned or
probed on holdout sequences; they are read only by the grader.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pyarrow.parquet as pq

HOLDOUT_MODULUS = 5


def is_holdout(seq_ix: int) -> bool:
    return int(hashlib.sha256(str(int(seq_ix)).encode()).hexdigest(), 16) % HOLDOUT_MODULUS == 0


def valid_row_groups(path) -> tuple[np.ndarray, np.ndarray]:
    """(dev_row_groups, holdout_row_groups) for a validation parquet file."""
    pf = pq.ParquetFile(path)
    seq = pf.read(columns=["seq_ix"]).column(0).to_numpy()[::20_000]
    hold = np.array([is_holdout(s) for s in seq])
    groups = np.arange(pf.num_row_groups)
    return groups[~hold], groups[hold]


def dev_halves(path) -> tuple[np.ndarray, np.ndarray]:
    """Split dev row groups in two by a salted seq_ix hash (for fine-tune-on-A, score-on-B)."""
    pf = pq.ParquetFile(path)
    seq = pf.read(columns=["seq_ix"]).column(0).to_numpy()[::20_000]
    dev, _ = valid_row_groups(path)
    half_a = np.array([int(hashlib.sha256(f"devhalf:{int(seq[g])}".encode()).hexdigest(), 16) % 2 == 0 for g in dev])
    return dev[half_a], dev[~half_a]
