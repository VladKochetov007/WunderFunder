import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
pf=pq.ParquetFile('datasets/valid.parquet')
t=pf.read_row_group(0).to_pandas()
print(t.shape)
d=t.describe().T[['mean','std','min','50%','max']]
d['nuniq']=[t[c].nunique() for c in d.index]
d['frac_change']=[(t[c].diff()!=0).mean() for c in d.index]
print(d.to_string())
print(t[['step_in_seq','need_prediction','is_scored','t0','t1']].iloc[95:130].to_string())
