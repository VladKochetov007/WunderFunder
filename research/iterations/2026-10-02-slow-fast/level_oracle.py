import numpy as np
exec(open("mix.py").read().split("def wp(")[0])
P,sel=load("d256")
def wp(P,g):
    out=[]
    for j in range(2):
        y=np.clip(Y[g][...,j],-2,2)[M[g]].astype(np.float64);p=np.clip(P[g][...,j],-2,2)[M[g]].astype(np.float64);w=np.abs(y)
        my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum();out.append((w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum()))
    return np.mean(out)
W=np.abs(np.clip(Y,-2,2))*M[...,None]
smY=(W*np.clip(Y,-2,2)).sum(1)/np.maximum(W.sum(1),1e-9);smP=(W*P).sum(1)/np.maximum(W.sum(1),1e-9)
base=wp(P,sel);print("base",round(base,4))
for lam in (0.25,0.5,1.0):
    Q=P+lam*(smY-smP)[:,None,:];print(f"level shifted {lam:.2f} toward oracle seq level:",round(wp(Q,sel),4))
for c in range(4):
    g=sel&(cls==c);Q=P+(smY-smP)[:,None,:];print(f" class{c}: base {wp(P,g):.4f} oracle-level {wp(Q,g):.4f}")
# causal proxy: running weighted mean of preds is what model has; how predictable is seq level from first 2000 rows of predictions?
