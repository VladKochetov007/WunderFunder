"""Composite eval of FT variants vs cmp_fullft (same parents). Usage: comp_eval2.py arm1 arm2 ... [--blend name:w]"""
import sys,os,numpy as np
exec(open("comp_eval.py").read().split("public={")[0])
def S(name,P_override=None):
    return score(name)
arms=[a for a in sys.argv[1:]]
base=np.array([score(f"cmp_fullft_s{s}") for s in (3,4)]);print("fullft      valid",base[:,0].round(4),"train",base[:,1].round(4),flush=True)
for a in arms:
    ss=[s for s in (3,4) if os.path.exists(R+f"cache/pred_cmp_{a}_s{s}_train.npy")]
    if not ss:continue
    r=np.array([score(f"cmp_{a}_s{s}") for s in ss]);d=r-base[[s-3 for s in ss]]
    print(f"{a:10s} seeds {ss} d_valid {d[:,0].round(4)} mean {d[:,0].mean():+.4f} | d_trainHO {d[:,1].round(4)} mean {d[:,1].mean():+.4f} | d_comp {d[:,2].mean():+.4f}",flush=True)
