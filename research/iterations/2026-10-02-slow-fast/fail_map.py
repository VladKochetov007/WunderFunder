import sys,pickle,numpy as np
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"src");sys.path.insert(0,S+"scripts")
from i1swap_aug import labels_with_mask
from wnn.data import SequenceFile
from wnn.ticks import reconstruct
moves,spreads=pickle.load(open(S+"cache/class3_moves_full.pkl","rb"))
t=np.load(S+"cache/valid_targets.npz");rg=t["row_group"];cls=t["market_class"];sf=SequenceFile(S+"datasets/valid.parquet")
rows=[]
for i in np.where(cls==3)[0][:80]:
    s=sf.read(int(rg[i]));p=s.features[:,0:22];y=s.targets
    lab,ok=labels_with_mask(p,moves,spreads);a,b,_=reconstruct(p,moves,spreads)
    n=len(y);ev=np.r_[True,(np.diff(a)!=0)|(np.diff(b)!=0)];last=np.maximum.accumulate(np.where(ev,np.arange(n),0));age=np.arange(n)-last
    sp=a-b;code_known=np.array([tuple(np.round(r,4)) in spreads for r in p]) if False else None
    # rows until label known (events ahead) proxy: distance to 4th future event
    evpos=np.flatnonzero(ev);nxt=np.searchsorted(evpos,np.arange(n),side="right")
    d4=np.where(nxt+3<len(evpos),evpos[np.minimum(nxt+3,len(evpos)-1)]-np.arange(n),n)
    bad=(np.abs(lab-y)>1e-3).any(1)
    m=ok&(np.arange(n)>=99)
    rows.append(np.c_[bad[m],age[m],sp[m],d4[m],np.abs(np.clip(y[m],-2,2)).sum(1)])
R=np.concatenate(rows);bad,age,sp,d4,w=R.T
print("overall fail %.3f, weighted fail %.3f"%(bad.mean(),(bad*w).sum()/w.sum()))
for name,v,edges in(("age(rows since last event)",age,[0,1,4,16,64,256,1e9]),("spread ticks",sp,[-1e9,0.5,1.5,2.5,3.5,1e9]),("rows to 4th next event",d4,[0,8,32,128,512,1e9])):
    print(name)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(v>=lo)&(v<hi)
        if m.sum():print(f"  [{lo:g},{hi:g}) share {m.mean():.3f} fail {bad[m].mean():.3f}")
