import pyarrow.parquet as pq, numpy as np, sys
np.set_printoptions(precision=4, suppress=True, linewidth=200)
f = pq.ParquetFile('datasets/valid.parquet')
tb = f.read_row_group(0).to_pandas()
print(tb.shape, tb.seq_ix.iloc[0])
cols = tb.columns
X = tb.iloc[:, 4:116].to_numpy(np.float32)
names = list(cols[4:116])
for i, n in enumerate(names):
    x = X[:, i]
    d = np.diff(x)
    print(f"{n:8s} mean={x.mean():10.4f} std={x.std():9.4f} min={x.min():9.4f} max={x.max():9.4f} nuniq={len(np.unique(x)):6d} frac_dchg={np.mean(d!=0):.3f} first={x[:5]}")
for t in ['t0', 't1']:
    y = tb[t].to_numpy()
    print(t, 'mean', y.mean(), 'std', y.std(), 'q', np.quantile(y, [0, .01, .1, .5, .9, .99, 1]), 'frac0', np.mean(y == 0), 'nuniq', len(np.unique(y)))
    print(' acf', [np.corrcoef(y[:-k], y[k:])[0, 1] for k in [1, 2, 5, 10, 20, 50, 100, 200, 500]])
print('corr t0 t1', np.corrcoef(tb.t0, tb.t1)[0, 1])
print('is_scored frac', tb.is_scored.mean(), 'first scored', np.where(tb.is_scored)[0][:20])
sc = tb.is_scored.to_numpy()
print('scored runs:', np.diff(np.where(np.diff(sc.astype(int)) != 0)[0])[:30])
