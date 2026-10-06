import numpy as np
exec(open("comp_eval.py").read().split("public={")[0])
def lvl(name):
    Pv=np.load(R+f"cache/pred_{name}_valid.npy",mmap_mode="r");Pt=np.load(R+f"cache/pred_{name}_train.npy",mmap_mode="r")
    Sv=np.zeros((2,6));St=np.zeros((2,6))
    for i in iv:
        for j in range(2):Sv[j]+=sums(Yv[i,:,j],np.asarray(Pv[i,:,j]),Mv[i].astype(float))
    for i in it:
        q=np.asarray(prop[rgt[i]]).astype(float);q[:99]=0
        for j in range(2):St[j]+=sums(Yt[i,:,j],np.asarray(Pt[i,:,j]),q)
    return wp(Sv),wp(St)
for n in ("orgbase32","cmp_baseft35_s3","cmp_baseft4_s3"):
    v,t=lvl(n);print(f"{n:18s} valid fold0 {v:.4f} train-HO {t:.4f} ratio {t/v:.3f}",flush=True)
