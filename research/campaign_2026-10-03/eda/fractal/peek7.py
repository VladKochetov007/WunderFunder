import numpy as np, pyarrow.parquet as pq
np.set_printoptions(linewidth=220, precision=3, suppress=True)
pf = pq.ParquetFile('datasets/valid.parquet')
P0 = [f'i0_p{k}' for k in range(22)]; V0=[f'i0_v{k}' for k in range(22)]; P1=[f'i1_p{k}' for k in range(22)]
D0=['i0_dp0','i0_dv0']; D1=['i1_dp0','i1_dv0']
def acf(x, maxlag):
    x = x - x.mean(); n=len(x); f=np.fft.rfft(x, 2*n); a=np.fft.irfft(f*np.conj(f))[:maxlag+1]; return a/a[0]
acc={};gaps=[]
for g in range(0,20):
    t = pf.read_row_group(g, columns=P0+V0+P1+D0+D1+['a2','t0']).to_pandas()
    def ev(cols): A=t[cols].values; return np.r_[True, np.any(np.diff(A,axis=0)!=0,axis=1)].astype(float)
    e = {'P0':ev(P0[:11]),'Pask0':ev(P0[11:]),'V0':ev(V0),'P1':ev(P1),'D0':ev(D0),'D1':ev(D1),'dt0':np.r_[1.,(np.diff(t.t0.values)!=0)]}
    for k,v in e.items(): acc.setdefault(k,[]).append(acf(v,40))
    idx=np.where(e['P0']+e['Pask0']>0)[0]; gaps.append(np.diff(idx))
    if g==0:
        a2=t.a2.values; print('a2 first diffs', np.diff(a2[:30]))
        print('bid P change rows', np.where(e['P0'][:200])[0][:40])
        print('ask P change rows', np.where(e['Pask0'][:200])[0][:40])
        print('trade rows', np.where(e['D0'][:200])[0][:40])
for k,v in acc.items(): print(k, 'mean rate', '', np.mean(v,0)[1:31])
G=np.concatenate(gaps); u,c=np.unique(G,return_counts=True); print('gap dist', list(zip(u[:20],c[:20])))
