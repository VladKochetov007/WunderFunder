"""Diagnostic 6: within-segment output smoothing (segment = rows between i0 price events, where labels are constant).
Uses OOF valid preds (folds 0+1) of given runs. Variants causal: freeze at event row, segment-EMA (reset at event)."""
import sys,numpy as np,pyarrow.parquet as pq
sys.path.insert(0,"/home/grinchenkoeg/WunderFunder/src")
from wnn.data import I0,FEATURE_COLUMNS
exec(open("tailrank.py").read().split("def wp")[0])  # Y,M,rg,cls,fk,groups
R="/home/grinchenkoeg/WunderFunder-creative/"
pf=pq.ParquetFile(R+"datasets/valid.parquet");names=pf.schema_arrow.names
sel=np.isin(fk,[0,1]);idx=np.where(sel)[0]
# event flags per valid sequence
import os
evf=R+"cache/valid_i0px_events.npy"
if os.path.exists(evf):EV=np.load(evf)
else:
    cb,ca=FEATURE_COLUMNS[I0.bid_px.start],FEATURE_COLUMNS[I0.ask_px.start];print("cols",cb,ca,pf.num_row_groups,flush=True)
    EV=np.zeros(Y.shape[:2],bool)
    for i in range(pf.num_row_groups):
        t=pf.read_row_group(i,columns=[cb,ca]);b=t.column(0).to_numpy();a=t.column(1).to_numpy()
        EV[i,1:]=(b[1:]!=b[:-1])&(a[1:]!=a[:-1]);EV[i,0]=True
    np.save(evf,EV)
def wp(P,g):
    out=[]
    for j in range(2):
        y=np.clip(Y[g][...,j],-2,2)[M[g]].astype(np.float64);p=np.clip(P[g][...,j],-2,2)[M[g]].astype(np.float64);w=np.abs(y)
        my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum();out.append((w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum()))
    return np.mean(out)
def seg_ema(P,ev,a):
    Q=P.copy()
    for t in range(1,P.shape[1]):
        keep=~ev[:,t]
        Q[keep,t]=Q[keep,t-1]+a*(P[keep,t]-Q[keep,t-1])
    return Q
# check label constancy within segments
seg_chg=(np.abs(np.diff(Y[idx],axis=1)).sum(-1)>0)&~EV[idx,1:]
print("label changes off-event rows:",int(seg_chg.sum()),"of",int((np.abs(np.diff(Y[idx],axis=1)).sum(-1)>0).sum()))
for run in sys.argv[1:]:
    P=np.zeros_like(Y,dtype=np.float32)
    for k in (0,1):
        p=np.load(R+f"cache/pred_{run}_f{k}_valid.npy",mmap_mode="r");i=np.where(fk==k)[0];P[i]=p[i]
    Ps=P[idx];ev=EV[idx]
    print(f"\n{run}: variant".ljust(28)+" ".join(f"{k:>9s}" for k in groups))
    base={k:wp(P,g&sel) for k,g in groups.items()}
    print("raw".ljust(28)+" ".join(f"{v:9.4f}" for v in base.values()))
    for a in (0.0,0.05,0.1,0.25,0.5):
        Q=P.copy();Q[idx]=seg_ema(Ps,ev,a) if a>0 else seg_ema(Ps,ev,0.0)
        for name,g3 in (("all",None),("c3only",cls==3)):
            Z=P.copy()
            if g3 is None:Z[idx]=Q[idx]
            else:
                ii=idx[g3[idx]];Z[ii]=Q[ii]
            print(f"ema a={a} {name}".ljust(28)+" ".join(f"{wp(Z,g&sel)-base[k]:+9.4f}" for k,g in groups.items()),flush=True)
