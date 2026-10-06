import sys,pickle,numpy as np
from collections import Counter
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"src")
from wnn.data import SequenceFile
from wnn.ticks import reconstruct,CLASS3_QUANTUM as q
moves,spreads=pickle.load(open(S+"cache/class3_moves_full.pkl","rb"))
t=np.load(S+"cache/valid_targets.npz");rg=t["row_group"];cls=t["market_class"];sf=SequenceFile(S+"datasets/valid.parquet")
diff=Counter();persist=[];trade_rate=[[],[]];vol_rate=[[],[]]
for i in np.where(cls==3)[0][:60]:
    s=sf.read(int(rg[i]));X=s.features;p=X[:,0:22];y=np.rint(s.targets/q).astype(int)
    a,b,unk=reconstruct(p,moves,spreads)
    if unk:continue
    n=len(y);mid2=a+b;sp=a-b
    F0=y[:,0]+a;F1=b-y[:,1];capS=y[:,1]==-(sp+2);capL=y[:,0]==-(sp+2);F=np.where(capL,F1,F0)
    ev=np.r_[0,np.flatnonzero((np.diff(a)!=0)|(np.diff(b)!=0))+1];nxt=np.searchsorted(ev,np.arange(n),side="right")
    cm=np.r_[0,np.cumsum(mid2[ev])];k=nxt
    Fev=np.where(k+4<=len(ev),np.rint((cm[np.minimum(k+4,len(ev))]-cm[np.minimum(k,len(ev))])/8.0),np.nan)
    d=F-Fev;rows=np.arange(99,n-10);good=~np.isnan(d[rows])
    for v in d[rows][good]:diff[int(v)]+=1
    # within an inter-event interval, is the mismatch constant? (hidden shift at an earlier row)
    bad=(d!=0)&~np.isnan(d)
    # change in mismatch status inside intervals: rows where bad flips without a visible event
    flip=np.flatnonzero(np.diff(bad.astype(int))!=0)+1
    inside=~np.isin(flip,ev)
    persist.append(inside.mean() if len(flip) else np.nan)
    # at flip rows inside intervals: what else changes? trades (dv), volumes
    fl=flip[inside]
    if len(fl):
        dv=(np.abs(np.diff(X[:,48:52],axis=0)).sum(1)>0);vv=(np.abs(np.diff(X[:,22:44],axis=0)).sum(1)>0)
        trade_rate[0].append(dv[fl-1].mean());trade_rate[1].append(dv.mean())
        vol_rate[0].append(vv[fl-1].mean());vol_rate[1].append(vv.mean())
tot=sum(diff.values());print("F_true - F_law (ticks) distribution:",{k:round(v/tot,4) for k,v in sorted(diff.items())})
print("share of mismatch-status flips that happen WITHOUT a visible price event: %.3f"%np.nanmean(persist))
print("at those flips: trade-change rate %.3f (baseline %.3f), volume-change rate %.3f (baseline %.3f)"%(np.mean(trade_rate[0]),np.mean(trade_rate[1]),np.mean(vol_rate[0]),np.mean(vol_rate[1])))
