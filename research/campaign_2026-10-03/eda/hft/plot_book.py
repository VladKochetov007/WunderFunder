import pyarrow.parquet as pq, numpy as np, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
pf=pq.ParquetFile('datasets/valid.parquet')
g=int(sys.argv[1]); a,b=int(sys.argv[2]),int(sys.argv[3])
t=pf.read_row_group(g).to_pandas().iloc[a:b]
fig,ax=plt.subplots(6,1,figsize=(15,18),sharex=True)
x=t.step_in_seq.values
for i in range(11): ax[0].plot(x,t[f'i0_p{i}'],lw=.8,label=f'p{i}')
ax[0].legend(ncol=11,fontsize=7); ax[0].set_title(f'seq g={g} i0 bid price cols')
for i in range(11,22): ax[1].plot(x,t[f'i0_p{i}'],lw=.8,label=f'p{i}')
ax[1].legend(ncol=11,fontsize=7); ax[1].set_title('i0 ask price cols')
ax[2].plot(x,t.t0,label='t0'); ax[2].plot(x,t.t1,label='t1'); ax[2].plot(x,t.t0+t.t1,label='t0+t1',alpha=.6); ax[2].plot(x,(t.t0-t.t1)/2,label='(t0-t1)/2',alpha=.6); ax[2].legend(); ax[2].grid(alpha=.3)
for i in range(4): ax[3].plot(x,t[f'i0_dp{i}'],lw=.8,label=f'dp{i}')
for i in range(4): ax[3].plot(x,t[f'i0_dv{i}'],lw=.8,ls='--',label=f'dv{i}')
ax[3].legend(ncol=8,fontsize=7); ax[3].set_title('i0 trades')
for i in range(8): ax[4].plot(x,t[f'a{i}'],lw=.8,label=f'a{i}')
ax[4].legend(ncol=8,fontsize=7); ax[4].set_title('a cols')
for i in range(0,11,2): ax[5].plot(x,t[f'i1_p{i}'],lw=.8,label=f'i1_p{i}')
ax[5].legend(ncol=8,fontsize=7); ax[5].set_title('i1 bid cols (subset)')
plt.tight_layout(); plt.savefig(sys.argv[4],dpi=80)
