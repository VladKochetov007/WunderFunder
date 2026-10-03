import numpy as np, pyarrow.parquet as pq
np.set_printoptions(linewidth=200, precision=5, suppress=True)
pf = pq.ParquetFile('datasets/valid.parquet')
for g in [0,1,2]:
    t = pf.read_row_group(g, columns=['step_in_seq','is_scored','t0','t1']).to_pandas()
    for c in ['t0','t1']:
        v = t[c].values.astype(np.float64)
        u, cnt = np.unique(np.round(v,4), return_counts=True)
        print(g, c, 'nuniq', len(u), 'top', list(zip(u[np.argsort(-cnt)][:8], np.sort(-cnt)[:8]*-1)))
        print('   frac change step->step', np.mean(np.diff(v)!=0), 'frac zero', np.mean(v==0))
        # gcd-ish quantum
        nz = np.abs(v[v!=0]); q = nz.min(); print('   min |nz|', q, 'ratio check', np.unique(np.round(nz/q,3))[:15])
    s = t.is_scored.values
    print('   corr t0,t1', np.corrcoef(t.t0, t.t1)[0,1], 'scored frac', s.mean())
    # scored run lengths
    d = np.diff(np.r_[0, s.astype(int), 0]); st = np.where(d==1)[0]; en = np.where(d==-1)[0]
    rl = en-st; print('   scored runs n', len(rl), 'len stats', np.percentile(rl,[0,25,50,75,100]))
    gaps = st[1:]-en[:-1]; print('   gap stats', np.percentile(gaps,[0,25,50,75,100]))
