"""Diagnostic 7: is there information in the decoded class-3 book state beyond the GRU?
Cross-fit (fold 0 <-> fold 1, chain folds) a GBM on decoded-state features (+GRU pred) predicting the |y|-weighted residual;
report WP of GRU, lookup-only, GRU+residual on c3 / c3quiet / c3active, scored rows only."""
import sys,pickle,numpy as np
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"src")
from wnn.data import SequenceFile
from wnn.ticks import reconstruct,CLASS3_QUANTUM as q
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
exec(open("tailrank.py").read().split("def wp")[0])
run=sys.argv[1] if len(sys.argv)>1 else "d256boots4"
moves,spreads=pickle.load(open(S+"cache/class3_moves_full.pkl","rb"));sf=SequenceFile(S+"datasets/valid.parquet")
P=np.zeros(Y.shape,np.float32)
for k in (0,1):
    p=np.load(S+f"cache/pred_{run}_f{k}_valid.npy",mmap_mode="r");i=np.where(fk==k)[0];P[i]=p[i]
rows=[];skipped=0
for i in np.where((cls==3)&np.isin(fk,[0,1]))[0]:
    s=sf.read(int(rg[i]));a,b,unk=reconstruct(s.features[:,0:22],moves,spreads)
    if unk:skipped+=1;continue
    n=len(a);sp=a-b;ev=np.r_[0,np.flatnonzero((np.diff(a)!=0)|(np.diff(b)!=0))+1]
    k=np.searchsorted(ev,np.arange(n),side="right")-1  # index of last event at/before t
    da=np.r_[0,np.diff(a[ev])];db=np.r_[0,np.diff(b[ev])]
    def lag(x,l):j=k-l;return np.where(j>=0,x[np.maximum(j,0)],0)
    since=np.arange(n)-ev[k];since2=np.arange(n)-np.where(k-1>=0,ev[np.maximum(k-1,0)],0)
    cnt=np.r_[0,np.cumsum(np.isin(np.arange(1,n),ev))];rate100=cnt-np.r_[np.zeros(100),cnt[:-100]];rate1000=cnt-np.r_[np.zeros(1000),cnt[:-1000]]
    m=M[i];t=np.where(m)[0]
    X=np.column_stack([sp,lag(da,0),lag(db,0),lag(da,1),lag(db,1),lag(da,2),lag(db,2),np.log1p(since),np.log1p(since2),rate100,rate1000,(a-a[ev[k]]),])[t]
    rows.append((i,X,P[i][t],Y[i][t]))
print("c3 seqs",len(rows),"skipped (decode unknown)",skipped,flush=True)
def wpj(y,p):
    y=np.clip(y,-2,2).astype(np.float64);p=np.clip(p,-2,2).astype(np.float64);w=np.abs(y);my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum()
    return (w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum())
seq=np.array([r[0] for r in rows]);F=fk[seq]
cat=lambda ids,j:np.concatenate([rows[x][j] for x in ids])
out={}
for mode in ("gru","lookup","gru+resid"):
    preds={}
    for te in (0,1):
        tr_ids=np.where(F!=te)[0];te_ids=np.where(F==te)[0]
        Xtr,Ptr,Ytr=cat(tr_ids,1),cat(tr_ids,2),np.clip(cat(tr_ids,3),-2,2);Xte,Pte=cat(te_ids,1),cat(te_ids,2)
        pr=np.zeros_like(Pte)
        for j in range(2):
            if mode=="gru":pr[:,j]=Pte[:,j]
            elif mode=="lookup":
                m=HGB(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=200).fit(Xtr,Ytr[:,j],sample_weight=np.abs(Ytr[:,j])+1e-3);pr[:,j]=m.predict(Xte)
            else:
                m=HGB(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=200).fit(np.c_[Xtr,Ptr],Ytr[:,j]-Ptr[:,j],sample_weight=np.abs(Ytr[:,j])+1e-3);pr[:,j]=Pte[:,j]+m.predict(np.c_[Xte,Pte])
        for x,ii in enumerate(te_ids):pass
        preds[te]=(te_ids,pr)
    for gname in ("c3","c3quiet","c3active"):
        g=groups[gname];ys=[];ps=[]
        for te,(ids,pr) in preds.items():
            off=0
            for x in ids:
                L=len(rows[x][3])
                if g[rows[x][0]]:ys.append(rows[x][3]);ps.append(pr[off:off+L])
                off+=L
        yy=np.concatenate(ys);pp=np.concatenate(ps)
        out[(mode,gname)]=np.mean([wpj(yy[:,j],pp[:,j]) for j in range(2)])
    print(mode," ".join(f"{g}={out[(mode,g)]:.4f}" for g in ("c3","c3quiet","c3active")),flush=True)
