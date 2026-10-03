import numpy as np, pyarrow.parquet as pq
np.set_printoptions(linewidth=200, precision=4, suppress=True)
pf = pq.ParquetFile('datasets/valid.parquet')
t = pf.read_row_group(0).to_pandas()
print(t.shape)
feat = [c for c in t.columns if c not in ('seq_ix','step_in_seq','need_prediction','is_scored','t0','t1')]
d = t[feat]
desc = d.describe().T[['mean','std','min','max']]
desc['nuniq'] = [d[c].nunique() for c in feat]
desc['frac0'] = [(d[c]==0).mean() for c in feat]
desc['frac_chg'] = [(np.diff(d[c].values)!=0).mean() for c in feat]
import pandas as pd
pd.set_option('display.max_rows', 200); pd.set_option('display.width', 200)
print(desc)
print(t[['t0','t1']].describe())
print('is_scored frac', t.is_scored.mean())
s = t.is_scored.values.astype(int)
print('scored runs: first idx', np.argmax(s), 'transitions', np.sum(np.abs(np.diff(s))))
print(t[['i0_p0','i0_p1','i0_p10','i0_p11','i0_p12','i0_p21']].head(10))
print(t[['i0_dp0','i0_dp1','i0_dp2','i0_dp3','i0_dv0','i0_dv1','i0_dv2','i0_dv3']].head(10))
print(t[['a0','a1','a2','a3','a4','a5','a6','a7']].head(10))
