import pyarrow.parquet as pq, numpy as np, pandas as pd
pf=pq.ParquetFile('datasets/valid.parquet')
P0=[f'i0_p{i}' for i in range(22)]
t=pf.read_row_group(74,columns=P0+['t0','t1','i0_dv0','i0_dp0']).to_pandas()
q=0.65687
U,sid=np.unique(t[P0].values,axis=0,return_inverse=True); sid=sid.ravel()
k0=np.round(t.t0.values/q).astype(int); k1=np.round(t.t1.values/q).astype(int)
tr=np.r_[0,(np.diff(t.i0_dv0.values)!=0)].astype(int)
key=list(zip(sid,k0,k1))
out=[]; start=0
for i in range(1,len(key)+1):
    if i==len(key) or key[i]!=key[start]:
        out.append(f'{start}:{key[start][0]}({key[start][1]},{key[start][2]})x{i-start}t{tr[start:i].sum()}')
        start=i
print(len(out)); print('  '.join(out[1000:1250]))
