import numpy as np, pyarrow.parquet as pq
np.set_printoptions(linewidth=220, precision=3, suppress=True)
pf = pq.ParquetFile('datasets/valid.parquet')
P0 = [f'i0_p{k}' for k in range(22)]
def acf(x, maxlag):
    x = x - x.mean(); n=len(x); f=np.fft.rfft(x, 2*n); a=np.fft.irfft(f*np.conj(f))[:maxlag+1]; return a/a[0]
lags=[1,2,3,5,8,10,15,20,30,40,50,60,80,100,150,200,300]
R0=[];R1=[];NE=[]; dacf0=[]; dacf1=[]
for g in range(0,40):
    t = pf.read_row_group(g, columns=P0+['t0','t1']).to_pandas()
    A=t[P0].values; ev = np.r_[True, np.any(np.diff(A,axis=0)!=0,axis=1)]
    y0=t.t0.values[ev].astype(np.float64); y1=t.t1.values[ev].astype(np.float64)
    NE.append(ev.sum())
    R0.append(acf(y0,300)); R1.append(acf(y1,300))
    dacf0.append(acf(np.diff(y0),60)); dacf1.append(acf(np.diff(y1),60))
print('events per seq', np.percentile(NE,[0,10,50,90,100]))
R0=np.mean(R0,0);R1=np.mean(R1,0)
print('lag', lags); print('ev-acf t0', R0[lags]); print('ev-acf t1', R1[lags])
d0=np.mean(dacf0,0);d1=np.mean(dacf1,0)
print('acf of event-diff t0 lags1..60:', d0[1:61]); print('acf of event-diff t1:', d1[1:61])
