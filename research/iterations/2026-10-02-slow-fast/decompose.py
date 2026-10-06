"""Where does pooled WP come from? Slow (sequence/class level) vs fast (within-sequence) prediction components.
Uses held-out fold predictions (fold0+fold1, all valid fold sequences). Diagnostic only."""
import numpy as np,sys
R="/home/grinchenkoeg/WunderFunder-creative/"
run=sys.argv[1] if len(sys.argv)>1 else "d256"
t=np.load(R+"cache/valid_targets.npz");Y=t["y"];M=t["mask"].copy();M[:,:99]=False;rg=t["row_group"];cls=t["market_class"]
cf=np.load(R+"cache/chain_folds.npz");v=cf["split"]=="valid";fold=dict(zip(cf["row_group"][v],cf["fold"][v]))
fk=np.array([fold[g] for g in rg])
P=np.zeros_like(Y);sel=np.zeros(len(rg),bool)
for k in (0,1):
    p=np.load(R+f"cache/pred_{run}_f{k}_valid.npy",mmap_mode="r");i=np.where(fk==k)[0];P[i]=p[i];sel[i]=True
Y,M,P,cls=Y[sel],M[sel],P[sel],cls[sel]
def wp(P,Y=Y,M=M,idx=None):
    if idx is not None:P,Y,M=P[idx],Y[idx],M[idx]
    out=[]
    for j in range(2):
        y=np.clip(Y[...,j],-2,2)[M].astype(np.float64);p=np.clip(P[...,j],-2,2)[M].astype(np.float64);w=np.abs(y)
        my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum()
        out.append((w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum()))
    return np.mean(out)
print("seqs",sel.sum(),"class counts",np.bincount(cls))
print(f"base {wp(P):.4f}")
for c in range(4): print(f" class{c} within-class WP {wp(P,idx=cls==c):.4f}")
# weighted means on scored rows per sequence
W=np.abs(np.clip(Y,-2,2))*M[...,None]
smY=(W*np.clip(Y,-2,2)).sum(1)/W.sum(1);smP=(W*P).sum(1)/W.sum(1)
print("seq-mean corr (target vs pred) per target:",[np.corrcoef(smY[:,j],smP[:,j])[0,1].round(3) for j in range(2)])
print(f"oracle within-seq (p - seq wmean p) {wp(P-smP[:,None,:]):.4f}")
print(f"only seq wmean p                     {wp(np.broadcast_to(smP[:,None,:],P.shape).copy()):.4f}")
cm=np.array([[smP[cls==c,j].mean() for j in range(2)] for c in range(4)])
print(f"p - class mean                       {wp(P-cm[cls][:,None,:]):.4f}")
# causal EMA de-meaning of predictions (deployable)
for span in (500,2000,8000):
    a=2/(span+1);E=np.zeros_like(P);e=P[:,0].copy()
    for s in range(P.shape[1]):
        e=e+a*(P[:,s]-e);E[:,s]=e
    for lam in (0.5,1.0):
        print(f"causal p - {lam}*EMA{span}              {wp(P-lam*E):.4f}")
# target side: share of weighted target variance that is between-sequence
for j in range(2):
    y=np.clip(Y[...,j],-2,2);w=W[...,j];mu=(w*y).sum()/w.sum()
    tot=(w*(y-mu)**2).sum();btw=(w.sum(1)*(smY[:,j]-mu)**2).sum();print(f"t{j}: between-seq share of weighted target var {btw/tot:.3f}")
