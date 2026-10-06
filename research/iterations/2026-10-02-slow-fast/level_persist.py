import numpy as np
exec(open("mix.py").read().split("def wp(")[0])
P,sel=load("d256")
Yc=np.clip(Y,-2,2)
# use ALL rows after warmup for realized level (labels mature causally), and scored rows for evaluation
A=np.zeros(M.shape,bool);A[:,99:]=True
def lev(mask,lo,hi):
    m=mask.copy();m[:,:lo]=False;m[:,hi:]=False
    W=np.abs(Yc)*m[...,None]
    return (W*(Yc-P)).sum(1)/np.maximum(W.sum(1),1e-9),W.sum(1)
for c in (0,1,2,3):
    g=sel&(cls==c)
    for name,mask in(("allrows",A),("scored",M)):
        r1,w1=lev(mask,99,10000);r2,w2=lev(mask,10000,20000)
        ok=g&(w1.min(1)>0)&(w2.min(1)>0)
        cc=[np.corrcoef(r1[ok,j],r2[ok,j])[0,1] for j in range(2)]
        print(f"class{c} {name:7s} n={ok.sum()} corr(resid level 1st half, 2nd half) t0 {cc[0]:.3f} t1 {cc[1]:.3f}")
# finer: corr between residual level over rows [s-2000,s) all rows and scored residual over [s, s+2000)
for c in (3,0):
    g=sel&(cls==c);cs=[]
    for s in range(4000,20000,4000):
        r1,w1=lev(A,s-2000,s);r2,w2=lev(M,s,s+2000);ok=g&(w1.min(1)>0)&(w2.min(1)>0)
        cs.append(np.mean([np.corrcoef(r1[ok,j],r2[ok,j])[0,1] for j in range(2)]))
    print(f"class{c}: corr(past-2000-row residual level, next-2000 scored residual level) by position",np.round(cs,3))
