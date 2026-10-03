import pyarrow.parquet as pq, numpy as np
TRAIN = 'datasets/train.parquet'; VALID = 'datasets/valid.parquet'
OUT = 'research/campaign_2026-10-03/eda/statphys/'
def feat_names(pf):
    return pf.schema_arrow.names[pf.schema_arrow.names.index('i0_p0'):pf.schema_arrow.names.index('a7') + 1]
def load(pf, rg, extra=()):
    fn = feat_names(pf)
    t = pf.read_row_group(int(rg), columns=fn + ['t0', 't1'] + list(extra))
    X = np.stack([t.column(c).to_numpy() for c in fn], 1).astype(np.float32)
    Y = np.stack([t.column('t0').to_numpy(), t.column('t1').to_numpy()], 1).astype(np.float32)
    ex = {c: t.column(c).to_numpy() for c in extra}
    a = np.abs(Y); a = a[a > 1e-6]; u = float(a.min()) if len(a) else np.nan
    return X, Y, u, ex
def wp(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    if w.sum() == 0: return np.nan
    my = (w * y).sum() / w.sum(); mp = (w * p).sum() / w.sum()
    c = (w * (y - my) * (p - mp)).sum(); v = np.sqrt((w * (y - my) ** 2).sum() * (w * (p - mp) ** 2).sum())
    return c / v if v > 0 else np.nan
def ucls(u):
    return {0.1254: 'S', 0.4685: 'M', 0.6569: 'L'}.get(round(u, 4), f'{u:.4f}')
