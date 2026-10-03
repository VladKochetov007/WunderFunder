import pyarrow.parquet as pq, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
f = pq.ParquetFile('datasets/valid.parquet')
rng = np.random.default_rng(1)
rgs = rng.choice(f.metadata.num_row_groups, 40, replace=False)
for P in [2, 4, 8, 16]:
    c0 = np.zeros(P); c1 = np.zeros(P); cs = np.zeros(P); n = np.zeros(P); cb = np.zeros(P)
    per_seq_best = []
    for rg in rgs:
        tb = f.read_row_group(int(rg), columns=['step_in_seq', 'is_scored', 't0', 't1', 'i0_p0'])
        st = tb.column('step_in_seq').to_numpy()[1:]
        d0 = np.diff(tb.column('t0').to_numpy()) != 0; d1 = np.diff(tb.column('t1').to_numpy()) != 0
        sc = tb.column('is_scored').to_numpy()[1:]
        db = np.diff(tb.column('i0_p0').to_numpy()) != 0
        ph = st % P
        for p in range(P):
            m = ph == p
            c0[p] += d0[m].sum(); c1[p] += d1[m].sum(); cs[p] += sc[m].sum(); n[p] += m.sum(); cb[p] += db[m].sum()
        if P == 8:
            h = np.array([d0[ph == p].mean() for p in range(8)])
            per_seq_best.append(np.argmax(h))
    print('P', P, '\n  dt0', c0 / n, '\n  dt1', c1 / n, '\n  scored', cs / n, '\n  dbid', cb / n)
    if P == 8: print('  per-seq argmax phase of dt0:', np.bincount(per_seq_best, minlength=8))
