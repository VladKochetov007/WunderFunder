"""Row-group loaders (one row group = one sequence)."""
import numpy as np
import pyarrow.parquet as pq

ROOT = "/home/vlad/development/wunder/wnn_connectome_starterpack/datasets"
_pf = {}


def pf(split="train"):
    if split not in _pf:
        _pf[split] = pq.ParquetFile(f"{ROOT}/{split}.parquet")
    return _pf[split]


def columns(split="train"):
    return pf(split).schema_arrow.names


def load(rg, cols=None, split="train"):
    """Return dict col -> np.ndarray (float32 for floats)."""
    t = pf(split).read_row_group(rg, columns=cols)
    out = {}
    for name in t.column_names:
        a = t.column(name).to_numpy()
        out[name] = a.astype(np.float32) if a.dtype == np.float64 else a
    return out


def load_mat(rg, cols, split="train"):
    t = pf(split).read_row_group(rg, columns=cols)
    return np.stack([t.column(c).to_numpy() for c in cols], 1).astype(np.float32)


def n_seq(split="train"):
    return pf(split).metadata.num_row_groups
