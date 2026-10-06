import numpy as np
src=open("tailrank.py").read().split("runs=sorted")[0]
exec(src)
members=["d256","d256s4","h256s4","ref","refs4","c500"]
sel=np.isin(fk,[0,1])
Ps=[]
for r in members:
    P=np.zeros_like(Y)
    for k in (0,1):
        p=np.load(R+f"cache/pred_{r}_f{k}_valid.npy",mmap_mode="r");i=np.where(fk==k)[0];P[i]=p[i]
    Ps.append(P)
Ps=np.stack(Ps);mu=Ps.mean(0);sd=Ps.std(0)
print("member pred corr",np.round(np.corrcoef(Ps[:,sel][:, :, ::50].reshape(len(members),-1))[0,1:],4))
print("mean sd/|mu| on scored", (sd[sel][M[sel]].mean()/np.abs(mu[sel][M[sel]]).mean()).round(4))
def show(name,P):print(name.ljust(18)," ".join(f"{k}={wp(P,g&sel):.4f}" for k,g in groups.items()))
show("ens mean",mu)
for k in (1,3,10,30):
    show(f"shrink k={k}",mu/(1+k*sd**2))
# relative: shrink by sd relative to typical level
for k in (0.5,1,2):
    show(f"rel k={k}",mu/(1+k*(sd/(np.abs(mu)+0.05))**2))
