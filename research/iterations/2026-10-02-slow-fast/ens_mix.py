import numpy as np
exec(open("tailrank.py").read().split("runs=sorted")[0])
sel=np.isin(fk,[0,1])
cache={}
def L(r):
    if r not in cache:
        P=np.zeros_like(Y)
        for k in (0,1):
            p=np.load(R+f"cache/pred_{r}_f{k}_valid.npy",mmap_mode="r");i=np.where(fk==k)[0];P[i]=p[i]
        cache[r]=P
    return cache[r]
def ens(spec):
    tot=sum(w for _,w in spec);return sum(L(r)*w for r,w in spec)/tot
cands={
 "raw4  d256+qw0+qw0s4+c500":[("d256",1),("qw0",1),("qw0s4",1),("c500",1)],
 "dm4   d256+adem+adems4+c500":[("d256",1),("adem",1),("adems4",1),("c500",1)],
 "mix6  d256x2+qw0+qw0s4+adem+adems4+c500":[("d256",2),("qw0",1),("qw0s4",1),("adem",1),("adems4",1),("c500",1)],
 "mix6b d256x2+d256s4... ":[("d256",1),("d256s4",1),("qw0",1),("qw0s4",1),("adem",1),("adems4",1),("c500",1)],
 "dm4+d256s4":[("d256",1),("d256s4",1),("adem",1),("adems4",1),("c500",1)],
}
for n,s in cands.items():
    P=ens(s);print(n.ljust(40)," ".join(f"{k}={wp(P,g&sel):.4f}" for k,g in groups.items() if k!="c3active"))
