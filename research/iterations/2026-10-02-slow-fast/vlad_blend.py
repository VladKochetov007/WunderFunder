import numpy as np
exec(open("tailrank.py").read().split("runs=sorted")[0])
H=np.load(R+"cache/vlad_holdout_mask.npy");V=np.load(R+"cache/pred_vlad_s4_valid.npy",mmap_mode="r")
B=np.load(R+"cache/pred_orgbase_valid.npy",mmap_mode="r")
def oof(run):
    P=np.zeros(Y.shape,np.float32)
    for k in (0,1):
        p=np.load(R+f"cache/pred_{run}_f{k}_valid.npy",mmap_mode="r");i=np.where(fk==k)[0];P[i]=p[i]
    return P
def L(n):return np.asarray(np.load(R+f"cache/pred_{n}_valid.npy",mmap_mode="r"))
sel01=H&np.isin(fk,[0,1]);sel0=H&(fk==0)
Vv=np.asarray(V);Bv=np.asarray(B)
D=oof("d256boots4");Dm=oof("d256mu6");F3=L("cmp_fullft_s3");F4=L("cmp_fullft_s4");RC=L("cmp_rawcorr_s3")
def rep(name,P,sel):print(f"{name:40s} " + " ".join(f"{k}={wp(P,g&sel):.4f}" for k,g in groups.items() if k in ("all","c012","c3","c3quiet")),flush=True)
print("== folds0+1 holdout (n=%d)"%sel01.sum())
for nm,P in [("vlad",Vv),("our d256boots4 OOF",D),("vlad+.25*.25base",0.75*Vv+0.0625*Bv),("vlad+d256boots4 .5/.5",0.5*Vv+0.5*D),("vlad .7 + d256boots4 .3",0.7*Vv+0.3*D),("vlad+d256boots4+d256mu6 .5/.25/.25",0.5*Vv+0.25*D+0.25*Dm),("vlad .6+boots4 .25 + base .0375",0.6*Vv+0.25*D+0.15*0.25*Bv)]:rep(nm,P,sel01)
print("== fold0 holdout (n=%d)"%sel0.sum())
Fp=0.5*(F3+F4)
for nm,P in [("vlad",Vv),("cmp_fullft pair",Fp),("vlad .5 + fullft pair .5",0.5*Vv+0.5*Fp),("vlad+base(.25,.25)",0.75*Vv+0.0625*Bv),("vlad+rawcorr .15",0.85*Vv+0.15*RC),("vlad .6+base .0625+rawcorr .15",0.6/0.8125*0.8125*Vv+0.0625*Bv+0.15*RC),("vlad .45+pair .3+base+rawcorr",0.4375*Vv+0.35*Fp+0.0625*Bv+0.15*RC)]:rep(nm,P,sel0)
