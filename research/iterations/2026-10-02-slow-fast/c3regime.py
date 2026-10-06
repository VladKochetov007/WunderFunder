"""Class 3: which label-free sequence stats predict per-sequence WP (d256 OOF), and how train/valid differ on them."""
import numpy as np,pyarrow.parquet as pq
from scipy.stats import spearmanr
exec(open("mix.py").read().split("def wp(")[0])
P,sel=load("d256")
sv=pq.read_table(R+"cache/seqstats_valid.parquet").to_pydict();st=pq.read_table(R+"cache/seqstats_train.parquet").to_pydict()
rgi={g:i for i,g in enumerate(sv["row_group"])}
labelish=("t0","t1","S_","D_","ev_target","scored")
keys=[k for k in sv if k not in("row_group","seq_ix","cls") and not k.startswith(labelish)]
idx=np.where(sel&(cls==3))[0];wps=[]
for i in idx:
    r=[]
    for j in range(2):
        y=np.clip(Y[i,:,j],-2,2)[M[i]];p=np.clip(P[i,:,j],-2,2)[M[i]];w=np.abs(y)
        if w.sum()==0 or p.std()==0:r.append(np.nan);continue
        my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum();r.append((w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum()))
    wps.append(np.nanmean(r))
wps=np.array(wps);print("class3 seqs",len(idx),"median per-seq WP",np.nanmedian(wps).round(3))
tc=np.array(st["cls"])==3;vc=np.array(sv["cls"])==3
rows=[]
for k in keys:
    x=np.array([sv[k][rgi[rg[i]]] for i in idx],float)
    if np.nanstd(x)==0:continue
    rho=spearmanr(x,wps,nan_policy="omit").correlation
    a=np.array(st[k],float)[tc];b=np.array(sv[k],float)[vc]
    smd=(np.nanmean(b)-np.nanmean(a))/np.sqrt((np.nanvar(a)+np.nanvar(b))/2+1e-12)
    rows.append((k,rho,smd))
rows.sort(key=lambda r:-abs(r[1]))
print("stat  spearman(WP)  valid-train SMD (class 3)")
for k,rho,smd in rows[:25]:print(f"{k:16s} {rho:+.3f} {smd:+.3f}")
