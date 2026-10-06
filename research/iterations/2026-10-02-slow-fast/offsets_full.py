"""Oracle headroom of per-class prediction offsets/scales on held-out valid folds (diagnostic upper bound)."""
import numpy as np,sys
from scipy.optimize import minimize
R="/home/grinchenkoeg/WunderFunder-creative/"
run=sys.argv[1]
t=np.load(R+"cache/valid_targets.npz");Y=t["y"];M=t["mask"].copy();M[:,:99]=False;rg=t["row_group"];cls=t["market_class"]
cf=np.load(R+"cache/chain_folds.npz");v=cf["split"]=="valid";fold=dict(zip(cf["row_group"][v],cf["fold"][v]));fk=np.array([fold[g] for g in rg])
P=np.zeros_like(Y);sel=np.zeros(len(rg),bool)
for k in (0,1):
    p=np.load(R+f"cache/pred_{run}_valid.npy",mmap_mode="r");i=np.where(fk==k)[0];P[i]=p[i];sel[i]=True
# flatten scored rows
rows=[];C=[];F=[]
for j in range(2):pass
idx=np.where(sel)[0]
ys=[];ps=[];cs=[];fs=[]
for i in idx:
    m=M[i];ys.append(Y[i][m]);ps.append(P[i][m]);cs.append(np.full(m.sum(),cls[i]));fs.append(np.full(m.sum(),fk[i]))
y=np.clip(np.concatenate(ys),-2,2).astype(np.float64);p=np.concatenate(ps).astype(np.float64);c=np.concatenate(cs);f=np.concatenate(fs)
def wp1(yy,pp):
    pp=np.clip(pp,-2,2);w=np.abs(yy);my=(w*yy).sum()/w.sum();mp=(w*pp).sum()/w.sum()
    return (w*(yy-my)*(pp-mp)).sum()/np.sqrt((w*(yy-my)**2).sum()*(w*(pp-mp)**2).sum())
def score(theta,sub,mode):
    out=[]
    for j in range(2):
        th=theta.reshape(2,4,-1)[j]
        pj=p[sub,j]*(th[c[sub],1] if mode=="affine" else 1)+th[c[sub],0]
        out.append(wp1(y[sub,j],pj))
    return np.mean(out)
for mode in ("offset","affine"):
    k=1 if mode=="offset" else 2
    x0=np.zeros(2*4*k);
    if k==2:x0.reshape(2,4,2)[:,:,1]=1
    allr=np.ones(len(y),bool);base=score(x0,allr,mode)
    r=minimize(lambda th:-score(th,allr,mode),x0,method="Powell",options=dict(maxiter=4000,xtol=1e-3))
    # honest: fit on one fold, eval on other
    cross=[]
    for a,b in ((0,1),(1,0)):
        ra=minimize(lambda th:-score(th,f==a,mode),x0,method="Powell",options=dict(maxiter=4000,xtol=1e-3))
        cross.append(score(ra.x,f==b,mode)-score(x0,f==b,mode))
    print(f"{run} {mode}: base {base:.4f} oracle {-r.fun:.4f} (+{-r.fun-base:.4f}) crossfold gains {np.round(cross,4)} params {np.round(r.x.reshape(2,4,k),3).tolist()}")
