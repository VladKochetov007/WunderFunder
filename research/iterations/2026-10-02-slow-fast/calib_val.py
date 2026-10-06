"""Calibrate a test-like weighting of local validation (fold 0+1 OOF) to reproduce public probe levels.
Weights per sequence: class share multiplier m_c (classes 0..3) times exp(-beta*z) for class-3 quiet index z.
Targets (public, midpoint ensemble): full .6639, c3zero .5473, c3only .4017, c3x0.7 .6556. Proxy ensemble: d256+ref+refs4+c500."""
import numpy as np
from scipy.optimize import least_squares
exec(open("tailrank.py").read().split("runs=sorted")[0])
sel=np.isin(fk,[0,1])
def L(r):
    P=np.zeros_like(Y)
    for k in (0,1):
        p=np.load(R+f"cache/pred_{r}_f{k}_valid.npy",mmap_mode="r");i=np.where(fk==k)[0];P[i]=p[i]
    return P
E=(L("d256")+L("ref")+L("refs4")+L("c500"))/4
idx=np.where(sel)[0];cl=cls[idx];zz=np.where(cl==3,Z[idx],0.0)
# per-sequence sufficient stats for fast weighted pooled WP
def stats(P,scale3=1.0,zero=None):
    S=np.zeros((len(idx),2,6))
    for n,i in enumerate(idx):
        m=M[i];y=np.clip(Y[i][m],-2,2).astype(np.float64);p=P[i][m].astype(np.float64)
        if cls[i]==3:p=p*scale3
        if zero is not None and zero(cls[i]):p=p*0
        p=np.clip(p,-2,2);w=np.abs(y)
        S[n]=np.stack([w.sum(0),(w*y).sum(0),(w*p).sum(0),(w*y*y).sum(0),(w*p*p).sum(0),(w*y*p).sum(0)],-1)
    return S
def wpS(S,sw):
    T=(S*sw[:,None,None]).sum(0);out=[]
    for W,sy,sp,syy,spp,syp in T:
        my,mp=sy/W,sp/W;out.append((syp/W-my*mp)/np.sqrt((syy/W-my*my)*(spp/W-mp*mp)))
    return np.mean(out)
S_full=stats(E);S_z=stats(E,zero=lambda c:c==3);S_o=stats(E,zero=lambda c:c!=3);S_7=stats(E,scale3=0.7)
obs=np.array([0.6639,0.5473,0.4017,0.6556])
def sw(x):
    m=np.exp(np.r_[0.0,x[:3]]);return m[cl]*np.exp(-x[3]*zz)
def res(x):return [wpS(S_full,sw(x))-obs[0],wpS(S_z,sw(x))-obs[1],wpS(S_o,sw(x))-obs[2],wpS(S_7,sw(x))-obs[3]]
x0=np.zeros(4);print("unweighted local:",np.round(np.array(res(x0))+obs,4))
r=least_squares(res,x0,bounds=([-3,-3,-3,-3],[3,3,3,3]))
print("fit residuals",np.round(r.fun,4),"params m1..m3 (log, vs class0)",np.round(r.x[:3],3),"beta",round(r.x[3],3))
w=sw(r.x);share=np.array([w[cl==c].sum() for c in range(4)])/w.sum();print("implied class shares (by sequences)",share.round(3))
np.save("calib_weights.npy",np.c_[idx,w]);print("saved")
