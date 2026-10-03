import numpy as np, pyarrow.parquet as pq
np.set_printoptions(linewidth=220, precision=3, suppress=True)
pf = pq.ParquetFile('datasets/valid.parquet')
names = pf.schema_arrow.names
P0 = [f'i0_p{k}' for k in range(22)]
cols = P0+['t0','t1']
dX=[];dY=[];X=[];Y=[];G=[]
for g in range(0,30):
    t = pf.read_row_group(g, columns=cols).to_pandas()
    A = t[P0].values.astype(np.float64); y=t[['t0','t1']].values.astype(np.float64)
    dX.append(np.diff(A,axis=0)); dY.append(np.diff(y,axis=0)); X.append(A); Y.append(y); G.append(np.full(len(A),g))
dX=np.vstack(dX); dY=np.vstack(dY)
tr = slice(0, 15*19999); te = slice(15*19999, None)
for j in range(2):
    b,*_ = np.linalg.lstsq(dX[tr], dY[tr,j], rcond=None)
    pr = dX[te]@b; r = np.corrcoef(pr, dY[te,j])[0,1]
    print('t%d dP->dt  OOS corr %.3f  R2 %.3f'%(j,r,r*r)); print('   beta', b)
# does nonzero dP coincide with target change?
nzP = np.any(dX!=0,axis=1); nzY = dY[:,0]!=0
print('P(dt0!=0 | dP!=0)=%.3f  P(dt0!=0 | dP==0)=%.3f  frac dP!=0 %.3f'%(nzY[nzP].mean(), nzY[~nzP].mean(), nzP.mean()))
nzY1 = dY[:,1]!=0
print('P(dt1!=0 | dP!=0)=%.3f  P(dt1!=0 | dP==0)=%.3f'%(nzY1[nzP].mean(), nzY1[~nzP].mean()))
# PCA of dP
U,S,Vt = np.linalg.svd(dX[nzP]-dX[nzP].mean(0), full_matrices=False)
print('dP PCA explained', (S**2/np.sum(S**2))[:8])
