import pyarrow.parquet as pq, numpy as np, pandas as pd, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'research/campaign_2026-10-03/eda/hft')
from decode_lib import decode, P0
pf=pq.ParquetFile('datasets/valid.parquet')
sc=pd.read_csv('research/campaign_2026-10-03/eda/hft/scan_seq.csv')
H=np.r_[1,2,5,10,20,50,100,150,200,300,400,600,800,1000,1500,2000,3000,5000]
HE=np.r_[1,2,5,10,20,30,50,75,100,150,200,300,500,1000]
res={}; rese={}
for qv,grp in sc.groupby(sc.q.round(3)):
    C0=np.zeros(len(H)); C1=np.zeros(len(H)); CE0=np.zeros(len(HE)); CE1=np.zeros(len(HE)); n=0
    for g in grp.g.values[:12]:
        t=pf.read_row_group(int(g),columns=P0+['t0','t1']).to_pandas()
        y0=t.t0.values.astype(np.float64); y1=t.t1.values.astype(np.float64)
        v=np.unique(np.abs(np.r_[np.diff(y0),np.diff(y1),y0,y1])); q=v[v>1e-6].min()
        A,B,sp,bk=decode(t[P0].values); M=(A+B)/2
        R0=y0/q+(A-M); R1=-(y1/q)+(B-M)   # F0-M, F1-M (ticks)
        T=len(M)
        for j,h in enumerate(H):
            D=M[h:]-M[:T-h]
            C0[j]+=np.corrcoef(R0[:T-h],D)[0,1]; C1[j]+=np.corrcoef(R1[:T-h],D)[0,1]
        ie=np.where(bk)[0]; Me=M[ie]; R0e=R0[ie]; R1e=R1[ie]; Te=len(ie)
        for j,h in enumerate(HE):
            D=Me[h:]-Me[:Te-h]
            CE0[j]+=np.corrcoef(R0e[:Te-h],D)[0,1]; CE1[j]+=np.corrcoef(R1e[:Te-h],D)[0,1]
        n+=1
    res[qv]=(C0/n,C1/n); rese[qv]=(CE0/n,CE1/n)
    print(qv,'rows: ',' '.join(f'{h}:{c:.2f}/{d:.2f}' for h,c,d in zip(H,C0/n,C1/n)))
    print(qv,'events:',' '.join(f'{h}:{c:.2f}/{d:.2f}' for h,c,d in zip(HE,CE0/n,CE1/n)))
fig,ax=plt.subplots(1,2,figsize=(13,4))
for qv in res:
    ax[0].semilogx(H,res[qv][0],'-o',label=f'q={qv} F0'); ax[0].semilogx(H,res[qv][1],'--x',label=f'q={qv} F1')
    ax[1].semilogx(HE,rese[qv][0],'-o',label=f'q={qv} F0'); ax[1].semilogx(HE,rese[qv][1],'--x',label=f'q={qv} F1')
ax[0].set_title('corr(target-implied F-M, decoded M(t+h)-M(t)), h in rows'); ax[1].set_title('same, h in i0 book events')
for x in ax: x.grid(alpha=.3); x.legend(fontsize=7)
plt.tight_layout(); plt.savefig('research/campaign_2026-10-03/eda/hft/fig05_horizon.png',dpi=85)
