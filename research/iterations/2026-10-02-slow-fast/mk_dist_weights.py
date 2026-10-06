import numpy as np
R="/home/grinchenkoeg/WunderFunder-creative/";D=R+"research/iterations/2026-10-02-slow-fast/"
cls=np.load(R+"research/artifacts/market-classes-v1.npz");Z=np.load(D+"quiet_z.npz")
# A: quiet class-3 emphasis (train only; valid stays x1)
w=np.ones(10607);c3=cls["train"]==3;w[c3]=np.clip(np.exp(-2*Z["train"][c3]),0.25,3);w[c3]/=w[c3].mean()
np.savez(D+"rep_quiet.npz",train=w,valid=np.ones(1873))
# B: train class mix -> valid class mix
tv=np.bincount(cls["train"],minlength=4)/len(cls["train"]);vv=np.bincount(cls["valid"],minlength=4)/len(cls["valid"])
mult=vv/tv;print("train mix",tv.round(3),"valid mix",vv.round(3),"mult",mult.round(3))
np.savez(D+"rep_classmix.npz",train=mult[cls["train"]],valid=np.ones(1873))
print("quiet weight pct",np.percentile(w[c3],[5,50,95]).round(2))
