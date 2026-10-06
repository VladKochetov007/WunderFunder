"""Composite validation: for each arm (2 seeds) compute WP on valid fold 0 (true mask), train half 1 (propensity-expected mask)
and the pooled composite; compare deltas vs base with known public deltas."""
import hashlib,os,numpy as np
R="/home/grinchenkoeg/WunderFunder-creative/"
tv=np.load(R+"cache/valid_targets.npz");Yv=tv["y"];Mv=tv["mask"].copy();Mv[:,:99]=False;rgv=tv["row_group"]
cf=np.load(R+"cache/chain_folds.npz");v=cf["split"]=="valid";fo=dict(zip(cf["row_group"][v],cf["fold"][v]));iv=np.array([i for i,g in enumerate(rgv) if fo[g]==0])
tt=np.load(R+"cache/train_targets.npz");Yt=tt["y"];rgt=tt["row_group"];prop=np.load(R+"cache/propcache_prop_f0_train.npy",mmap_mode="r")
half=np.array([int(hashlib.sha256(f"half:{g}".encode()).hexdigest(),16)%2 for g in rgt]);it=np.random.default_rng(1).choice(np.where(half==1)[0],2000,replace=False)
def sums(Y,P,W):
    y=np.clip(Y,-2,2).astype(np.float64);p=np.clip(P,-2,2).astype(np.float64);w=np.abs(y)*W
    return np.array([w.sum(),(w*y).sum(),(w*p).sum(),(w*y*y).sum(),(w*p*p).sum(),(w*y*p).sum()])
def wp(S):
    out=[]
    for W,sy,sp,syy,spp,syp in S:
        my,mp=sy/W,sp/W;out.append((syp/W-my*mp)/np.sqrt((syy/W-my*my)*(spp/W-mp*mp)))
    return float(np.mean(out))
def score(name):
    Pv=np.load(R+f"cache/pred_{name}_valid.npy",mmap_mode="r");Pt=np.load(R+f"cache/pred_{name}_train.npy",mmap_mode="r")
    Sv=np.zeros((2,6));St=np.zeros((2,6))
    for i in iv:
        for j in range(2):Sv[j]+=sums(Yv[i,:,j],np.asarray(Pv[i,:,j]),Mv[i].astype(float))
    for i in it:
        q=np.asarray(prop[rgt[i]]).astype(float);q[:99]=0
        for j in range(2):St[j]+=sums(Yt[i,:,j],np.asarray(Pt[i,:,j]),q)
    return wp(Sv),wp(St),wp(Sv+St)
public={"fullft":+0.0005,"v2":-0.0018,"vtm0":-0.0007,"trainonly":-0.0024,"adam":-0.0002}
res={}
for arm in ("base","fullft","v2","vtm0","trainonly","adam"):
    ss=[s for s in (3,4) if os.path.exists(R+f"cache/pred_cmp_{arm}_s{s}_train.npy")];r=np.array([score(f"cmp_{arm}_s{s}") for s in ss]);res[arm]=r.mean(0)
    print(f"{arm:10s} valid {r[:,0].round(4)} train {r[:,1].round(4)} comp {r[:,2].round(4)}",flush=True)
b=res["base"];print("\narm        public   d_valid  d_trainHO  d_composite")
for arm,pub in public.items():
    d=res[arm]-b;print(f"{arm:10s} {pub:+.4f}  {d[0]:+.4f}  {d[1]:+.4f}   {d[2]:+.4f}")
