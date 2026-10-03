# Shared helpers: column names, row-group reader, lattice scale of targets.
import numpy as np, pyarrow.parquet as pq

ROOT = "/home/vlad/development/wunder/wnn_connectome_starterpack"
TRAIN = f"{ROOT}/datasets/train.parquet"
VALID = f"{ROOT}/datasets/valid.parquet"


def cols(inst, grp, rng):
    return [f"{inst}_{grp}{k}" for k in rng]


I0P = cols("i0", "p", range(22)); I0V = cols("i0", "v", range(22))
I0DP = cols("i0", "dp", range(4)); I0DV = cols("i0", "dv", range(4))
I1P = cols("i1", "p", range(22)); I1V = cols("i1", "v", range(22))
I1DP = cols("i1", "dp", range(4)); I1DV = cols("i1", "dv", range(4))
A = [f"a{k}" for k in range(8)]
FEATS = I0P + I0V + I0DP + I0DV + I1P + I1V + I1DP + I1DV + A


def read(path, g, columns=None):
    return pq.ParquetFile(path).read_row_group(g, columns=columns).to_pandas()


def lattice(t0, t1):
    """Smallest positive spacing of the common target grid (t = c * integer)."""
    u = np.unique(np.abs(np.r_[t0, t1]).astype(np.float64))
    u = u[u > 1e-6]
    if len(u) == 0:
        return np.nan
    c = u.min()
    # refine: c may be a multiple of the true unit if the smallest |t| is 2 ticks
    for div in (1, 2, 3, 4, 5):
        cc = c / div
        if np.all(np.abs(u / cc - np.round(u / cc)) < 1e-3):
            best = cc
            break
    else:
        best = np.nan
    return best
