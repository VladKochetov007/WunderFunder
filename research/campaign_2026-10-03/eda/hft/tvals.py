import pyarrow.parquet as pq, numpy as np, pandas as pd
pf=pq.ParquetFile('datasets/valid.parquet')
for g in [1,74,111,0,555]:
    t=pf.read_row_group(g,columns=['t0','t1']).to_pandas()
    for c in ['t0']:
        vc=pd.Series(t[c].values).value_counts().sort_index()
        print(g,c,' '.join(f'{k:.4f}:{v}' for k,v in vc.items()))
