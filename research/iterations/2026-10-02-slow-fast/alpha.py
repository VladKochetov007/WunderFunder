import numpy as np
from scipy.optimize import least_squares
exec(open("invert.py").read().split("obs=np.array")[0])
def score2(s3,k,keep,alpha):
    mix=np.append(vm[:3]/vm[:3].sum()*(1-s3),s3);out=[]
    for j in range(2):
        T=np.zeros(6)
        for c in range(4):
            W,sy,sp,syy,spp,syp=S[c,j]*mix[c]/vm[c]
            kc=k[1] if c==3 else k[0]
            syp=sy*sp/W+kc*(syp-sy*sp/W)
            a=alpha if c==3 else 1.0
            if not keep[c]:a=0.0
            T+=[W,sy,a*sp,syy,a*a*spp,a*syp]
        W,sy,sp,syy,spp,syp=T;my,mp=sy/W,sp/W
        out.append((syp/W-my*mp)/np.sqrt((syy/W-my*my)*(spp/W-mp*mp)))
    return np.mean(out)
obs=np.array([0.6639,0.5473,0.4017])
alphas=[0.5,0.7,0.85,1.0,1.2,1.4]
# param A: share + k012 + k3 (exact)
fits={}
def fitA(x):return [score2(x[0],x[1:],[1,1,1,1],1)-obs[0],score2(x[0],x[1:],[1,1,1,0],1)-obs[1],score2(x[0],x[1:],[0,0,0,1],1)-obs[2]]
r=least_squares(fitA,[0.4,1,1],bounds=([0.05,0.3,0.3],[0.95,1.5,1.5]));fits["share+k012+k3"]=(r.x[0],r.x[1:],r.fun)
# param B: valid share fixed, k012 + k3 (least squares)
def fitB(x):return [score2(vm[3],x,[1,1,1,1],1)-obs[0],score2(vm[3],x,[1,1,1,0],1)-obs[1],score2(vm[3],x,[0,0,0,1],1)-obs[2]]
r=least_squares(fitB,[1,1],bounds=([0.3,0.3],[1.5,1.5]));fits["validshare,k012+k3"]=(vm[3],r.x,r.fun)
# param C: share + k3 only (k012=1)
def fitC(x):return [score2(x[0],[1,x[1]],[1,1,1,1],1)-obs[0],score2(x[0],[1,x[1]],[1,1,1,0],1)-obs[1],score2(x[0],[1,x[1]],[0,0,0,1],1)-obs[2]]
r=least_squares(fitC,[0.4,1],bounds=([0.05,0.3],[0.95,1.5]));fits["share+k3"]=(r.x[0],np.array([1,r.x[1]]),r.fun)
for n,(s3,k,res) in fits.items():
    curve=[score2(s3,k,[1]*4,a) for a in alphas]
    print(f"{n:22s} s3={s3:.3f} k={np.round(k,3)} resid={np.round(res,4)}")
    print("   alpha:",alphas);print("   score:",np.round(curve,4))
print("local valid alpha curve:",np.round([score2(vm[3],[1,1],[1]*4,a) for a in alphas],4))
