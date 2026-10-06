import numpy as np,pyarrow.parquet as pq
R="/home/grinchenkoeg/WunderFunder-creative/"
t=np.load(R+"cache/valid_targets.npz");Y=t["y"];M=t["mask"].copy();M[:,:99]=False;rg=t["row_group"];cls=t["market_class"]
cf=np.load(R+"cache/chain_folds.npz");v=cf["split"]=="valid";fold=dict(zip(cf["row_group"][v],cf["fold"][v]));pure=dict(zip(cf["row_group"][v],cf["pure"][v]))
fk=np.array([fold[g] for g in rg]);pu=np.array([pure[g] for g in rg]);Z=np.load(R+"research/iterations/2026-10-02-slow-fast/quiet_z.npz")["valid"]
def wp(P,g):
    out=[]
    for j in range(2):
        y=np.clip(Y[g][...,j],-2,2)[M[g]].astype(np.float64);p=np.clip(P[g][...,j],-2,2)[M[g]].astype(np.float64);w=np.abs(y)
        my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum();out.append((w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum()))
    return np.mean(out)
def L(n):return np.load(R+f"cache/pred_{n}_valid.npy",mmap_mode="r")
for T,Ks in ((3,(0,1,2)),(2,(0,1,3))):
    g=fk==T;i=np.where(g)[0]
    groups={"all":g,"pure":g&pu,"c012":g&(cls!=3),"c3":g&(cls==3),"c3quiet":g&(cls==3)&(Z<-1)}
    def show(name,P):print(f"T{T} {name:28s}"+" ".join(f"{k}={wp(P,gg):.4f}" for k,gg in groups.items()))
    full=np.array(L(f"ft128H_f{T}"));full2=(full+np.array(L(f"ft128s4H_f{T}")))/2
    kf=[np.array(L(f"nft{K}_f{T}")) for K in Ks]
    show("full valid (1 model)",full);show("full valid (2 seeds)",full2)
    for K,P in zip(Ks,kf):show(f"nested member excl {K}",P)
    show("k-fold avg (3 models)",sum(kf)/3)
    show("k-fold avg + full",(sum(kf)/3+full)/2)
