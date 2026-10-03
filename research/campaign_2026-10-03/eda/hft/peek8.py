import pyarrow.parquet as pq, numpy as np, pandas as pd
from collections import Counter
pf=pq.ParquetFile('datasets/valid.parquet')
qs=Counter()
for g in range(0,1873,7):
    t=pf.read_row_group(g,columns=['t0','t1']).to_pandas()
    v=np.unique(np.abs(np.r_[np.diff(t.t0.values),np.diff(t.t1.values),t.t0.values,t.t1.values]))
    v=v[v>1e-6]; q=v.min()
    # check all multiples
    k=v/q; ok=np.abs(k-np.round(k)).max()
    qs[(round(float(q),5), ok<1e-3)]+=1
for k,v in sorted(qs.items()): print(k,v)
for g in [0,74,111,555]:
    t=pf.read_row_group(g,columns=['t0','t1']).to_pandas()
    s=t.t0.values+t.t1.values
    print(g,'t0+t1 value counts', pd.Series(np.round(s,4)).value_counts().head(10).to_dict())
    print('   t0 vc', pd.Series(np.round(t.t0.values,4)).value_counts().head(8).to_dict())
