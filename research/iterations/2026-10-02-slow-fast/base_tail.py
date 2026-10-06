import numpy as np
exec(open("tailrank.py").read().split("runs=sorted")[0])
B=np.load(R+"cache/pred_orgbase_valid.npy",mmap_mode="r");sel=np.isin(fk,[0,1])
for run in ("d256boots4","d256mu6"):
    P=np.zeros_like(Y,dtype=np.float32)
    for k in (0,1):
        p=np.load(R+f"cache/pred_{run}_f{k}_valid.npy",mmap_mode="r");i=np.where(fk==k)[0];P[i]=p[i]
    b={k:wp(P,g&sel) for k,g in groups.items()}
    for w in (0.1,0.25,0.35):
        Q=(1-w)*P+w*0.25*np.asarray(B)
        print(run,f"w={w}"," ".join(f"{k}={wp(Q,g&sel)-b[k]:+.4f}" for k,g in groups.items()),flush=True)
