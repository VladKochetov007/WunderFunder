import numpy as np
R="/home/grinchenkoeg/WunderFunder-creative/"
for split in ("valid","train"):
    t=np.load(R+f"cache/{split}_targets.npz");Y=t["y"];M=t["mask"].copy();cl=t["market_class"]
    A=np.ones(M.shape,bool);A[:,:99]=False;M[:,:99]=False
    y=np.clip(Y,-2,2)
    print(split,"mask frac",M.mean().round(4))
    for k in range(4):
        i=cl==k;yi=y[i]
        for name,MM in (("scored",M[i]),("all",A[i])):
            w=np.abs(yi)*MM[...,None]
            print(f"  c{k} {name:6s} wmean {((w*yi).sum((0,1))/w.sum((0,1))).round(3)} mean {yi[MM].mean(0).round(3)} frac>0 {(yi[MM]>0).mean(0).round(3)} frac0 {(yi[MM]==0).mean(0).round(3)}")
