import numpy as np
exec(open("mix.py").read().split("def wp(")[0])
P,sel=load("d256")
def wp(P,g):
    out=[]
    for j in range(2):
        y=np.clip(Y[g][...,j],-2,2)[M[g]].astype(np.float64);p=np.clip(P[g][...,j],-2,2)[M[g]].astype(np.float64);w=np.abs(y)
        my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum();out.append((w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum()))
    return np.mean(out)
for c in ("all",0,1,2,3):
    g=sel if c=="all" else sel&(cls==c)
    ys=(Y[g][...,0]+Y[g][...,1])[M[g]];ps=(P[g][...,0]+P[g][...,1])[M[g]]
    yd=(Y[g][...,0]-Y[g][...,1])[M[g]];pd=(P[g][...,0]-P[g][...,1])[M[g]]
    S=Y[...,0]+Y[...,1];D=P[...,0]-P[...,1]
    Q=np.stack([(S+D)/2,(S-D)/2],-1)  # oracle sum, model difference
    D2=Y[...,0]-Y[...,1];Ps=P[...,0]+P[...,1];Q2=np.stack([(Ps+D2)/2,(Ps-D2)/2],-1)  # model sum, oracle diff
    print(f"{str(c):4s} corr(sum) {np.corrcoef(ys,ps)[0,1]:.3f} corr(diff) {np.corrcoef(yd,pd)[0,1]:.3f} | WP base {wp(P,g):.4f} oracle-sum {wp(Q,g):.4f} oracle-diff {wp(Q2,g):.4f} | var share sum {ys.var()/(ys.var()+yd.var()):.2f}")
