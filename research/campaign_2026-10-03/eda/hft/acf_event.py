import pyarrow.parquet as pq, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
pf=pq.ParquetFile('datasets/valid.parquet')
P0=[f'i0_p{i}' for i in range(22)]
sc=pd.read_csv('research/campaign_2026-10-03/eda/hft/scan_seq.csv')
L=600
res={}
for qv,grp in sc.groupby('q'):
    acc=np.zeros(L); accd=np.zeros(L); n=0
    for g in grp.g.values[:12]:
        t=pf.read_row_group(int(g),columns=P0+['t0','t1']).to_pandas()
        bk=np.r_[True,(np.diff(t[P0].values,axis=0)!=0).any(1)]
        for c in ['t0','t1']:
            y=t[c].values[bk].astype(np.float64); y=y-y.mean()
            f=np.fft.rfft(y,2**16); a=np.fft.irfft(f*np.conj(f))[:L]; acc+=a/a[0]
            # only rows where target changed (distinct values)
            y2=t[c].values.astype(np.float64); ch=np.r_[True,np.diff(y2)!=0]; y2=y2[ch]-y2[ch].mean()
            f=np.fft.rfft(y2,2**16); a=np.fft.irfft(f*np.conj(f))[:L]; accd+=a/a[0]
            n+=1
    res[qv]=(acc/n,accd/n)
fig,ax=plt.subplots(1,2,figsize=(13,4))
for qv,(a,b) in res.items():
    ax[0].plot(a,label=f'q={qv}'); ax[1].plot(b,label=f'q={qv}')
    print(qv,'event-acf', np.round(a[[1,2,5,10,20,50,100,200,400]],3))
ax[0].set_title('target ACF in i0 book-event time'); ax[1].set_title('target ACF over target-change rows')
for x in ax: x.legend(); x.grid(alpha=.3); x.axhline(0,c='k',lw=.5)
plt.tight_layout(); plt.savefig('research/campaign_2026-10-03/eda/hft/fig04_acf_eventtime.png',dpi=85)
