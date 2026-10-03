import pyarrow.parquet as pq, numpy as np, pandas as pd
pf=pq.ParquetFile('datasets/valid.parquet')
q=0.125433
for g in range(3):
    t=pf.read_row_group(g, columns=['t0','t1','is_scored','step_in_seq']).to_pandas()
    for c in ['t0','t1']:
        k=t[c].values/q
        print(g,c,'max frac dev from int', np.abs(k-np.round(k)).max(), 'nuniq', len(np.unique(t[c])), 'changes', (np.diff(t[c].values)!=0).sum())
    d0=np.diff(t.t0.values); d1=np.diff(t.t1.values)
    both=(d0!=0)&(d1!=0); same=both&(np.abs(d0-d1)<1e-4)
    print(' rows t0 changes',(d0!=0).sum(),'t1 changes',(d1!=0).sum(),'both',both.sum(),'both same amount',same.sum())
    s=t.is_scored.values
    print(' scored frac', s.mean(), 'first scored', np.argmax(s), 'runs', (np.diff(s.astype(int))!=0).sum())
    print(' d0 values', pd.Series(np.round(d0[d0!=0]/q)).value_counts().head(8).to_dict())
    print(' d1 values', pd.Series(np.round(d1[d1!=0]/q)).value_counts().head(8).to_dict())
