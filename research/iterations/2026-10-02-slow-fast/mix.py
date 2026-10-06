import numpy as np,sys
R="/home/grinchenkoeg/WunderFunder-creative/"
t=np.load(R+"cache/valid_targets.npz");Y=t["y"];M=t["mask"].copy();M[:,:99]=False;rg=t["row_group"];cls=t["market_class"]
cf=np.load(R+"cache/chain_folds.npz");v=cf["split"]=="valid";fold=dict(zip(cf["row_group"][v],cf["fold"][v]));fk=np.array([fold[g] for g in rg])
def load(run):
    P=np.zeros_like(Y);sel=np.zeros(len(rg),bool)
    for k in (0,1):
        p=np.load(R+f"cache/pred_{run}_f{k}_valid.npy",mmap_mode="r");i=np.where(fk==k)[0];P[i]=p[i];sel[i]=True
    return P,sel
def wp(P,sel,sw):
    out=[]
    for j in range(2):
        y=np.clip(Y[sel][...,j],-2,2);p=np.clip(P[sel][...,j],-2,2);w=np.abs(y)*M[sel]*sw[sel][:,None]
        my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum()
        out.append((w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum()))
    return np.mean(out)
for run in sys.argv[1:]:
    P,sel=load(run);vm=np.bincount(cls[sel],minlength=4)/sel.sum()
    res=[]
    for name,mix in (("valid",vm),("train",np.array([.123,.119,.329,.430])),("c3x1.5",vm*np.array([1,1,1,1.5])),("c2c3heavy",np.array([.1,.1,.35,.45]))):
        sw=(mix/vm)[cls];res.append(f"{name} {wp(P,sel,sw):.4f}")
    print(run," | ".join(res))
