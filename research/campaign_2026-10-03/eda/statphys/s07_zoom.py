import pyarrow.parquet as pq, numpy as np, pandas as pd, sys
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 200)
pf = pq.ParquetFile('datasets/train.parquet')
rg = int(sys.argv[1]); a, b = int(sys.argv[2]), int(sys.argv[3])
t = pf.read_row_group(rg).to_pandas()
u = np.unique(np.abs(t.t0.values)); u = u[u > 1e-6][0]
out = pd.DataFrame({'s': t.step_in_seq})
for j in [0, 3, 10]: out[f'b{j}'] = pd.factorize(t[f'i0_p{j}'].values, sort=True)[0]
for j in [11, 14, 19]: out[f'a{j}'] = pd.factorize(t[f'i0_p{j}'].values, sort=True)[0]
for j in [0, 1]: out[f'bv{j}'] = t[f'i0_v{j}'].round(2)
for j in [11, 12]: out[f'av{j}'] = t[f'i0_v{j}'].round(2)
out['dp0'] = t.i0_dp0.round(2); out['dv0'] = t.i0_dv0.round(2); out['dv3'] = t.i0_dv3.round(2)
for j in range(8): out[f'A{j}'] = t[f'a{j}'].round(2)
out['T0'] = (t.t0 / u).round(2); out['T1'] = (t.t1 / u).round(2)
print('unit', u)
print(out.iloc[a:b].to_string(index=False))
