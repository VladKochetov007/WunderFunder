import numpy as np, pyarrow.parquet as pq
np.set_printoptions(linewidth=220, precision=3, suppress=True)
pf = pq.ParquetFile('datasets/valid.parquet')
names = pf.schema_arrow.names
feat = names[4:116]
C0=[];C1=[];L0=[];L1=[]
for g in range(0,20):
    t = pf.read_row_group(g).to_pandas()
    X = t[feat].values.astype(np.float64); y0=t.t0.values.astype(np.float64); y1=t.t1.values.astype(np.float64)
    dX = np.diff(X,axis=0); dy0=np.diff(y0); dy1=np.diff(y1)
    def cc(A,b):
        A=A-A.mean(0); b=b-b.mean(); den=np.sqrt((A**2).sum(0)*(b**2).sum())+1e-12; return (A*b[:,None]).sum(0)/den
    C0.append(cc(dX,dy0)); C1.append(cc(dX,dy1)); L0.append(cc(X,y0)); L1.append(cc(X,y1))
C0=np.nanmean(C0,0);C1=np.nanmean(C1,0);L0=np.nanmean(L0,0);L1=np.nanmean(L1,0)
o=np.argsort(-np.abs(C0))
print('corr(dfeat, dt0) top:'); 
for i in o[:25]: print('  %-8s dT0 %+.3f dT1 %+.3f | lvl T0 %+.3f T1 %+.3f'%(feat[i],C0[i],C1[i],L0[i],L1[i]))
o=np.argsort(-np.abs(L0))
print('corr(level, t0) top:')
for i in o[:15]: print('  %-8s dT0 %+.3f dT1 %+.3f | lvl T0 %+.3f T1 %+.3f'%(feat[i],C0[i],C1[i],L0[i],L1[i]))
