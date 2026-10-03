# Per-sequence census: tick unit (class), fraction of zero targets, mean |T| in ticks, change rates, corr(t0,t1)
import pyarrow.parquet as pq, numpy as np, sys
fn = sys.argv[1]; out = sys.argv[2]
pf = pq.ParquetFile(fn)
cols = ['t0', 't1'] + (['is_scored'] if 'valid' in fn else [])
rows = []
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=['seq_ix'] + cols)
    y0 = t.column('t0').to_numpy().astype(np.float64); y1 = t.column('t1').to_numpy().astype(np.float64)
    a = np.abs(np.concatenate([y0, y1])); a = a[a > 1e-6]
    u = a.min() if len(a) else np.nan
    T0 = y0[99:] / u; T1 = y1[99:] / u
    sc = t.column('is_scored').to_numpy()[99:].mean() if 'is_scored' in cols else np.nan
    rows.append((t.column('seq_ix')[0].as_py(), rg, u, np.mean(T0 == 0), np.mean(T1 == 0), np.mean(np.abs(T0)), np.mean(np.abs(T1)),
                 np.mean(np.diff(T0) != 0), np.mean(np.diff(T1) != 0), np.corrcoef(T0, T1)[0, 1] if T0.std() > 0 and T1.std() > 0 else np.nan,
                 np.mean(np.abs(y0[99:]) >= 2), sc))
np.save(out, np.array(rows, dtype=np.float64))
print('done', len(rows))
