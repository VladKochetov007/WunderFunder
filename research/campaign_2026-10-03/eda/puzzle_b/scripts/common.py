import numpy as np, pyarrow.parquet as pq, pandas as pd
ROOT = '/home/vlad/development/wunder/wnn_connectome_starterpack'
OUT = ROOT + '/research/campaign_2026-10-03/eda/puzzle_b'
TRAIN = ROOT + '/datasets/train.parquet'
VALID = ROOT + '/datasets/valid.parquet'
_pf = {}
def pf(path=TRAIN):
    if path not in _pf:
        _pf[path] = pq.ParquetFile(path)
    return _pf[path]
def cols(path=TRAIN):
    return pf(path).schema_arrow.names
def seq(i, columns=None, path=TRAIN):
    t = pf(path).read_row_group(i, columns=columns)
    return t.to_pandas()
def nrg(path=TRAIN):
    return pf(path).metadata.num_row_groups
