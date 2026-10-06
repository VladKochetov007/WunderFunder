"""Invert public probe scores (full, c3zero, c3only) into test class-3 share and per-group covariance degradation."""
import numpy as np
from scipy.optimize import least_squares
exec(open("mix.py").read().split("def wp(")[0])
P,sel=load("d256");cl=cls[sel];Yc=np.clip(Y[sel],-2,2);Pc=np.clip(P[sel],-2,2);Ms=M[sel]
S=np.zeros((4,2,6))
for c in range(4):
    i=cl==c
    for j in range(2):
        y=Yc[i][...,j].astype(np.float64);p=Pc[i][...,j].astype(np.float64);w=np.abs(y)*Ms[i]
        S[c,j]=[w.sum(),(w*y).sum(),(w*p).sum(),(w*y*y).sum(),(w*p*p).sum(),(w*y*p).sum()]
vm=np.bincount(cl,minlength=4)/len(cl)
def score(s3,k,keep):
    mix=np.append(vm[:3]/vm[:3].sum()*(1-s3),s3);out=[]
    for j in range(2):
        T=np.zeros(6)
        for c in range(4):
            W,sy,sp,syy,spp,syp=S[c,j]*mix[c]/vm[c]
            kc=k[1] if c==3 else k[0]
            syp=sy*sp/W+kc*(syp-sy*sp/W)          # degrade within-class covariance
            if not keep[c]: sp=spp=syp=0.0        # zeroed predictions
            T+=[W,sy,sp,syy,spp,syp]
        W,sy,sp,syy,spp,syp=T;my,mp=sy/W,sp/W
        out.append((syp/W-my*mp)/np.sqrt((syy/W-my*my)*(spp/W-mp*mp)))
    return np.mean(out)
obs=np.array([0.6639,0.5473,0.4017])
f=lambda x:[score(x[0],x[1:],[1,1,1,1])-obs[0],score(x[0],x[1:],[1,1,1,0])-obs[1],score(x[0],x[1:],[0,0,0,1])-obs[2]]
r=least_squares(f,[0.4,1,1],bounds=([0.05,0.3,0.3],[0.95,1.5,1.5]))
print("share3 %.3f  k012 %.3f  k3 %.3f  resid"%tuple(r.x),np.round(r.fun,5))
print("local valid-mix:",round(score(vm[3],[1,1],[1]*4),4),round(score(vm[3],[1,1],[1,1,1,0]),4),round(score(vm[3],[1,1],[0,0,0,1]),4))
