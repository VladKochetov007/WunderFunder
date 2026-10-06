import numpy as np
exec(open("mix.py").read().split("def wp(")[0])
P,sel=load("ctxC")
def wp(P,rowmask,seqmask):
    out=[]
    for j in range(2):
        m=M[seqmask]&rowmask[None,:];y=np.clip(Y[seqmask][...,j],-2,2)[m].astype(np.float64);p=np.clip(P[seqmask][...,j],-2,2)[m].astype(np.float64);w=np.abs(y)
        my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum();out.append((w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum()))
    return np.mean(out)
edges=[99,2000,5000,10000,15000,20000];pos=np.arange(20000)
for name,sm in(("all",sel),("c012",sel&(cls!=3)),("c3",sel&(cls==3))):
    r=[wp(P,(pos>=a)&(pos<b),sm) for a,b in zip(edges[:-1],edges[1:])]
    print(name,"WP by row range",list(zip(edges[:-1],np.round(r,4))))
# share of scored weight by position
W=(np.abs(np.clip(Y[sel],-2,2))*M[sel][...,None]).sum(-1).sum(0)
print("scored weight share by range",[round(W[a:b].sum()/W.sum(),3) for a,b in zip(edges[:-1],edges[1:])])
