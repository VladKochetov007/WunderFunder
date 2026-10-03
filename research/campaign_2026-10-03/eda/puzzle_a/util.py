import numpy as np, pyarrow.parquet as pq
ROOT = "/home/vlad/development/wunder/wnn_connectome_starterpack/datasets/"
_pf = {}
def pf(name):
    if name not in _pf:
        _pf[name] = pq.ParquetFile(ROOT + name + ".parquet")
    return _pf[name]
def load(name, rg, cols=None):
    t = pf(name).read_row_group(rg, columns=cols)
    return t.to_pandas()
def cols_of(prefix):
    return [f"{prefix}_p{i}" for i in range(22)] + [f"{prefix}_v{i}" for i in range(22)] + [f"{prefix}_dp{i}" for i in range(4)] + [f"{prefix}_dv{i}" for i in range(4)]
