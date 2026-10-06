import numpy as np,pyarrow.parquet as pq,glob,os,re
R="/home/grinchenkoeg/WunderFunder-creative/"
t=np.load(R+"cache/valid_targets.npz");Y=t["y"];M=t["mask"].copy();M[:,:99]=False;rg=t["row_group"];cls=t["market_class"]
cf=np.load(R+"cache/chain_folds.npz");v=cf["split"]=="valid";fold=dict(zip(cf["row_group"][v],cf["fold"][v]));fk=np.array([fold[g] for g in rg])
sv=pq.read_table(R+"cache/seqstats_valid.parquet").to_pydict();st=pq.read_table(R+"cache/seqstats_train.parquet").to_pydict();rgi={g:i for i,g in enumerate(sv["row_group"])}
feats=["i0_v12_nuniq","i0_dp0_std","a5_mean","a3_std"];sign=np.array([1,1,-1,-1]);tc=np.array(st["cls"])==3
T=np.column_stack([np.array(st[k],float)[tc] for k in feats]);mu,sd=T.mean(0),T.std(0)
Z=((np.column_stack([np.array([sv[k][rgi[g]] for g in rg],float) for k in feats])-mu)/sd*sign).mean(1)
groups={"all":np.ones(len(rg),bool),"c012":cls!=3,"c3":cls==3,"c3quiet":(cls==3)&(Z<-1),"c3active":(cls==3)&(Z>=-0.5)}
def wp(P,g):
    out=[]
    for j in range(2):
        y=np.clip(Y[g][...,j],-2,2)[M[g]].astype(np.float64);p=np.clip(P[g][...,j],-2,2)[M[g]].astype(np.float64);w=np.abs(y)
        my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum();out.append((w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum()))
    return np.mean(out)
runs=sorted({re.match(r"pred_(.+)_f0_valid.npy",os.path.basename(f)).group(1) for f in glob.glob(R+"cache/pred_*_f0_valid.npy")})
runs=[r for r in runs if os.path.exists(R+f"cache/pred_{r}_f1_valid.npy")]
sel=np.isin(fk,[0,1]);print("n:",{k:int((g&sel).sum()) for k,g in groups.items()})
res=[]
for r in runs:
    P=np.zeros_like(Y)
    for k in (0,1):
        p=np.load(R+f"cache/pred_{r}_f{k}_valid.npy",mmap_mode="r");i=np.where(fk==k)[0];P[i]=p[i]
    res.append((r,*[wp(P,g&sel) for g in groups.values()]))
res.sort(key=lambda x:-x[4])
print("run".ljust(17)+" ".join(f"{k:>9s}" for k in groups)+"  quiet-c3 minus c3")
for r in res:print(f"{r[0]:16s} "+" ".join(f"{x:9.4f}" for x in r[1:])+f"  {r[4]-r[3]:+.4f}")
