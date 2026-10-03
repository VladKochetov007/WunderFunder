import numpy as np, pyarrow.parquet as pq
np.set_printoptions(linewidth=220, precision=4, suppress=True)
pf = pq.ParquetFile('datasets/valid.parquet')
def acf(x, maxlag):
    x = x - x.mean(); n=len(x); f=np.fft.rfft(x, 2*n); a=np.fft.irfft(f*np.conj(f))[:maxlag+1]; return a/a[0]
lags = [1,2,5,10,20,50,100,200,300,500,700,1000,1500,2000]
A0=[];A1=[];Q=[]
for g in range(0,40):
    t = pf.read_row_group(g, columns=['t0','t1']).to_pandas()
    a,b = t.t0.values.astype(np.float64), t.t1.values.astype(np.float64)
    nz = np.abs(np.r_[a,b]); nz=nz[nz>0]; q=nz.min(); Q.append(q)
    A0.append(acf(a,2000)); A1.append(acf(b,2000))
    if g<6:
        s=a+b; d=a-b
        print(g,'q',round(q,4),'t0+t1 /q uniq',np.unique(np.round(s/q)).astype(int)[:20],' std t0 %.3f t1 %.3f sum %.3f diff %.3f'%(a.std(),b.std(),s.std(),d.std()), 'chg frac', np.mean(np.diff(a)!=0))
A0=np.mean(A0,0);A1=np.mean(A1,0)
print('lag ', lags); print('acf0', A0[lags]); print('acf1', A1[lags])
print('Q quantiles', np.percentile(Q,[0,10,50,90,100]))
