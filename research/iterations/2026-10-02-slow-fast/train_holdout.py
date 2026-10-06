"""Train-holdout evaluation: model trained on train half 0 (+valid folds != 0) scored on train half 1 with propensity-expected masks."""
import sys,hashlib,numpy as np
R="/home/grinchenkoeg/WunderFunder-creative/"
run=sys.argv[1]
t=np.load(R+"cache/train_targets.npz");rg=t["row_group"];Y=t["y"];cls=t["market_class"]
P=np.load(R+f"cache/pred_{run}_train.npy",mmap_mode="r");prop=np.load(R+"cache/propcache_prop_f0_train.npy",mmap_mode="r")
half=np.array([int(hashlib.sha256(f"half:{g}".encode()).hexdigest(),16)%2 for g in rg])
rng=np.random.default_rng(0)
def pooled(idx,mode):
    out=[]
    for j in range(2):
        S=np.zeros(6)
        for i in idx:
            y=np.clip(Y[i,99:,j],-2,2).astype(np.float64);p=np.clip(np.asarray(P[i,99:,j]),-2,2).astype(np.float64);q=np.asarray(prop[rg[i],99:]).astype(np.float64)
            m=q if mode=="expected" else (rng.random(len(q))<q).astype(float)
            w=np.abs(y)*m;S+=[w.sum(),(w*y).sum(),(w*p).sum(),(w*y*y).sum(),(w*p*p).sum(),(w*y*p).sum()]
        W,sy,sp,syy,spp,syp=S;my,mp=sy/W,sp/W;out.append((syp/W-my*mp)/np.sqrt((syy/W-my*my)*(spp/W-mp*mp)))
    return np.mean(out)
for h in (1,0):
    idx=np.where(half==h)[0];sub=np.random.default_rng(1).choice(idx,1500,replace=False)
    print(f"{run} train half {h} ({'held out' if h==1 else 'IN TRAINING'}): expected-mask WP {pooled(sub,'expected'):.4f} sampled-mask WP {pooled(sub,'sampled'):.4f}",flush=True)
