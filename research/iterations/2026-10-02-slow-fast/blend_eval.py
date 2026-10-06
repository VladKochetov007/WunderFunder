"""Composite eval of blending cmp_<arm> preds with another pred (e.g. orgbase): P=(1-w)*P_arm + w*scale*P_other."""
import sys,numpy as np
exec(open("comp_eval.py").read().split("public={")[0])
arm,other,scale=sys.argv[1],sys.argv[2],float(sys.argv[3])
Bv=np.load(R+f"cache/pred_{other}_valid.npy",mmap_mode="r");Bt=np.load(R+f"cache/pred_{other}_train.npy",mmap_mode="r")
def score_blend(name,w):
    Pv=np.load(R+f"cache/pred_{name}_valid.npy",mmap_mode="r");Pt=np.load(R+f"cache/pred_{name}_train.npy",mmap_mode="r")
    Sv=np.zeros((2,6));St=np.zeros((2,6))
    for i in iv:
        for j in range(2):Sv[j]+=sums(Yv[i,:,j],(1-w)*np.asarray(Pv[i,:,j])+w*scale*np.asarray(Bv[i,:,j]),Mv[i].astype(float))
    for i in it:
        q=np.asarray(prop[rgt[i]]).astype(float);q[:99]=0
        for j in range(2):St[j]+=sums(Yt[i,:,j],(1-w)*np.asarray(Pt[i,:,j])+w*scale*np.asarray(Bt[i,:,j]),q)
    return np.array([wp(Sv),wp(St),wp(Sv+St)])
for s in (3,4):
    b=score_blend(f"cmp_{arm}_s{s}",0.0)
    for w in [float(x) for x in sys.argv[4:]]:
        r=score_blend(f"cmp_{arm}_s{s}",w);print(f"s{s} w={w}: d_valid {r[0]-b[0]:+.4f} d_trainHO {r[1]-b[1]:+.4f} d_comp {r[2]-b[2]:+.4f}",flush=True)
