import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
pf=pq.ParquetFile('datasets/valid.parquet')
t=pf.read_row_group(74).to_pandas()
q=0.65687
df=pd.DataFrame(dict(step=t.step_in_seq, k0=np.round(t.t0/q).astype(int), k1=np.round(t.t1/q).astype(int),
  p0=t.i0_p0.round(3), p3=t.i0_p3.round(3), p11=t.i0_p11.round(3), p19=t.i0_p19.round(3),
  dp0=t.i0_dp0.round(2), dp3=t.i0_dp3.round(2), dv0=t.i0_dv0.round(2), v0=t.i0_v0.round(2), v11=t.i0_v11.round(2), a7=t.a7.round(2), a6=t.a6.round(2)))
ch=np.r_[True,(np.diff(t[[f'i0_p{i}' for i in range(22)]].values,axis=0)!=0).any(1)]
tr=np.r_[True,(np.diff(t[[f'i0_dv{i}' for i in range(4)]].values,axis=0)!=0).any(1)]
df['bk']=ch.astype(int); df['tr']=tr.astype(int)
sub=df.iloc[12000:12160]
print(sub[(sub.bk==1)|(sub.tr==1)|(sub.index%10==0)].to_string())
