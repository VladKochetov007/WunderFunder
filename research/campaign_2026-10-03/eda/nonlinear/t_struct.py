import pyarrow.parquet as pq, numpy as np
pf=pq.ParquetFile('datasets/valid.parquet')
rng=np.random.default_rng(0)
idx=rng.choice(pf.metadata.num_row_groups,40,replace=False)
units=[];fr=[];blocks=[]
for k,i in enumerate(idx):
    t=pf.read_row_group(int(i),columns=['step_in_seq','is_scored','t0','t1']).to_pandas()
    s=t.is_scored.values
    for c in ['t0','t1']:
        v=np.unique(np.abs(t[c].values)); v=v[v>0]
        units.append(v.min())
    fr.append(s.mean())
    # block structure of is_scored
    d=np.diff(s.astype(int)); starts=np.where(d==1)[0]+1; ends=np.where(d==-1)[0]+1
    if k<6:
        print('seq',i,'scored frac %.3f'%s.mean(),'first scored',np.argmax(s),'n_blocks',len(starts),'starts',starts[:8],'ends',ends[:8])
        t0=t.t0.values; t1=t.t1.values
        u=np.unique(np.abs(t0)); print('  t0 unique abs head', u[:6], 'ratio to min', (u[1:6]/u[1]).round(3))
print('units', np.round(units,5)[:20], 'min/max', min(units), max(units))
print('scored frac mean', np.mean(fr), np.min(fr), np.max(fr))
