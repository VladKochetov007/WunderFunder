"""Shared loaders for the nonlinear EDA (row group = one sequence)."""
import numpy as np
import pyarrow.parquet as pq

ROOT = "/home/vlad/development/wunder/wnn_connectome_starterpack"
OUT = ROOT + "/research/campaign_2026-10-03/eda/nonlinear"
FEATS = ([f"i0_p{i}" for i in range(22)] + [f"i0_v{i}" for i in range(22)] + [f"i0_dp{i}" for i in range(4)]
         + [f"i0_dv{i}" for i in range(4)] + [f"i1_p{i}" for i in range(22)] + [f"i1_v{i}" for i in range(22)]
         + [f"i1_dp{i}" for i in range(4)] + [f"i1_dv{i}" for i in range(4)] + [f"a{i}" for i in range(8)])


def pfile(split):
    return pq.ParquetFile(f"{ROOT}/datasets/{split}.parquet")


def seq_ids(split, n, seed=0):
    pf = pfile(split)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pf.metadata.num_row_groups, n, replace=False))


def load(split, rg, cols=None):
    """Return dict of float32 arrays for one sequence."""
    pf = pfile(split)
    cols = cols or (FEATS + ["t0", "t1"] + (["is_scored"] if split == "valid" else []))
    tb = pf.read_row_group(int(rg), columns=cols)
    return {c: tb.column(c).to_numpy().astype(np.float32 if c != "is_scored" else bool) for c in cols}


def unit_of(t):
    v = np.unique(np.abs(t)); v = v[v > 0]
    return float(v.min()) if len(v) else np.nan
