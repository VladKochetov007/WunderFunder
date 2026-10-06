"""Composite eval at PAIR level: base = mean(cmp_fullft s3,s4); add members (each a seed-pair mean or single pred) with weights."""
import sys,numpy as np
exec(open("comp_eval.py").read().split("public={")[0])
L=lambda n,sp:np.load(R+f"cache/pred_{n}_{sp}.npy",mmap_mode="r")
def comp(spec):  # spec: list of (weight, [names], scale)
    Sv=np.zeros((2,6));St=np.zeros((2,6))
    srcs=[(w*sc/len(ns),[ (L(n,"valid"),L(n,"train")) for n in ns]) for w,ns,sc in spec]
    for i in iv:
        P=sum(c*sum(np.asarray(v[i]) for v,_ in arr) for c,arr in srcs)
        for j in range(2):Sv[j]+=sums(Yv[i,:,j],P[:,j],Mv[i].astype(float))
    for i in it:
        q=np.asarray(prop[rgt[i]]).astype(float);q[:99]=0
        P=sum(c*sum(np.asarray(t[i]) for _,t in arr) for c,arr in srcs)
        for j in range(2):St[j]+=sums(Yt[i,:,j],P[:,j],q)
    return np.array([wp(Sv),wp(St)])
pair=["cmp_fullft_s3","cmp_fullft_s4"];b0=comp([(1,pair,1)]);print("pair fullft valid %.4f trainHO %.4f"%tuple(b0),flush=True)
def rep(name,spec):r=comp(spec)-b0;print(f"{name:42s} d_valid {r[0]:+.4f} d_trainHO {r[1]:+.4f}",flush=True)
pr=lambda z:[f"cmp_{z}_s3",f"cmp_{z}_s4"]
rep("+ 4th/5th seed control: base pair .5",[(0.5,pair,1),(0.5,pr("base"),1)])
for z in ("rawwmse","rawcorr","za","zvol","zpxev","zi1"):
    if z=="rawwmse":rep("+ rawwmse(s3) .25",[(0.75,pair,1),(0.25,["cmp_rawwmse_s3"],1)])
    else:rep(f"+ {z} pair .25",[(0.75,pair,1),(0.25,pr(z),1)])
rep("+ baseline .25*.25",[(0.75,pair,1),(0.25,["orgbase"],0.25)])
rep("+ base + rawwmse",[(0.5,pair,1),(0.25,["orgbase"],0.25),(0.25,["cmp_rawwmse_s3"],1)])
rep("+ base + za + zvol + zpxev",[(0.4,pair,1),(0.25,["orgbase"],0.25),(0.12,pr("za"),1),(0.12,pr("zvol"),1),(0.11,pr("zpxev"),1)])
rep("+ base + rawwmse + za + zvol",[(0.4,pair,1),(0.25,["orgbase"],0.25),(0.15,["cmp_rawwmse_s3"],1),(0.1,pr("za"),1),(0.1,pr("zvol"),1)])
