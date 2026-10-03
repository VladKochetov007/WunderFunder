import pyarrow.parquet as pq, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
pf=pq.ParquetFile('datasets/valid.parquet')
L=4000
acc={'t0':np.zeros(L),'t1':np.zeros(L)}; cc=np.zeros(2*L-1); n=0
for g in range(0,1873,47):
    t=pf.read_row_group(g,columns=['t0','t1']).to_pandas()
    for c in ['t0','t1']:
        y=t[c].values.astype(np.float64); y=y-y.mean()
        f=np.fft.rfft(y,2**16); a=np.fft.irfft(f*np.conj(f))[:L]; acc[c]+=a/a[0]
    y0=t.t0.values-t.t0.values.mean(); y1=t.t1.values-t.t1.values.mean()
    f0=np.fft.rfft(y0,2**16); f1=np.fft.rfft(y1,2**16); x=np.fft.irfft(f0*np.conj(f1))
    xc=np.r_[x[-(L-1):],x[:L]]/np.sqrt((y0**2).sum()*(y1**2).sum()); cc+=xc
    n+=1
fig,ax=plt.subplots(1,2,figsize=(13,4))
for c in acc: ax[0].plot(acc[c]/n,label=c)
ax[0].axhline(0,c='k',lw=.5); ax[0].set_xlabel('lag (rows)'); ax[0].set_title('ACF of targets (mean over 40 valid seqs)'); ax[0].legend(); ax[0].grid(alpha=.3)
lags=np.arange(-(L-1),L); ax[1].plot(lags,cc/n); ax[1].set_title('xcorr t0(t) vs t1(t-k)'); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig('research/campaign_2026-10-03/eda/hft/fig01_target_acf.png',dpi=90)
a0=acc['t0']/n; a1=acc['t1']/n
for k in [1,5,10,20,50,100,200,300,500,1000,1500,2000,3000]: print(k, round(a0[k],3), round(a1[k],3))
print('xcorr lag0',cc[L-1]/n,'argmax lag',lags[np.argmax(cc)], (cc/n).max())
