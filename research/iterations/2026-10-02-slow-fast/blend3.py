"""Composite eval of P=(1-a-b)*arm + a*other1*s1 + b*other2*s2 vs arm."""
import sys,numpy as np
exec(open("comp_eval.py").read().split("public={")[0])
arm=sys.argv[1];o1,s1,o2,s2=sys.argv[2],float(sys.argv[3]),sys.argv[4],float(sys.argv[5]);grid=[tuple(map(float,x.split(","))) for x in sys.argv[6:]]
L=lambda n,sp:np.load(R+f"cache/pred_{n}_{sp}.npy",mmap_mode="r")
O1v,O1t,O2v,O2t=L(o1,"valid"),L(o1,"train"),L(o2,"valid"),L(o2,"train")
def sc(name,a,b):
    Pv,Pt=L(name,"valid"),L(name,"train");Sv=np.zeros((2,6));St=np.zeros((2,6))
    for i in iv:
        for j in range(2):Sv[j]+=sums(Yv[i,:,j],(1-a-b)*np.asarray(Pv[i,:,j])+a*s1*np.asarray(O1v[i,:,j])+b*s2*np.asarray(O2v[i,:,j]),Mv[i].astype(float))
    for i in it:
        q=np.asarray(prop[rgt[i]]).astype(float);q[:99]=0
        for j in range(2):St[j]+=sums(Yt[i,:,j],(1-a-b)*np.asarray(Pt[i,:,j])+a*s1*np.asarray(O1t[i,:,j])+b*s2*np.asarray(O2t[i,:,j]),q)
    return np.array([wp(Sv),wp(St),wp(Sv+St)])
for s in (3,4):
    b0=sc(f"cmp_{arm}_s{s}",0,0)
    for a,b in grid:
        r=sc(f"cmp_{arm}_s{s}",a,b);print(f"s{s} a={a} b={b}: d_valid {r[0]-b0[0]:+.4f} d_trainHO {r[1]-b0[1]:+.4f} d_comp {r[2]-b0[2]:+.4f}",flush=True)
