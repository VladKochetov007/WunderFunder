import numpy as np
R="/home/grinchenkoeg/WunderFunder-creative/"
c=np.load(R+"research/artifacts/market-classes-v1.npz")
for split in ("valid","train"):
    t=np.load(R+f"cache/{split}_targets.npz");keys=t.files;Y=t["y"]
    M=t["mask"] if "mask" in keys else np.ones(Y.shape[:2],bool)
    M=M.copy();M[:,:99]=False;cl=c[split]
    y=np.clip(Y,-2,2);W=np.abs(y)*M[...,None]
    sm=(W*y).sum(1)/np.maximum(W.sum(1),1e-9);um=(y*M[...,None]).sum(1)/M.sum(1)[:,None]
    print(split,keys)
    for k in range(4):
        i=cl==k
        print(f"  class{k} n={i.sum():5d} wmean y t0 {sm[i,0].mean():+.3f}±{sm[i,0].std():.3f} t1 {sm[i,1].mean():+.3f}±{sm[i,1].std():.3f} | plain mean t0 {um[i,0].mean():+.4f} t1 {um[i,1].mean():+.4f} | mean|y| {np.abs(y[i][M[i]]).mean(0).round(3)}")
