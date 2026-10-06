import os,sys
import numpy as np,torch
exec(open(os.path.expanduser("~/WunderFunder-creative/research/iterations/2026-10-02-slow-fast/build_v1dm.py")).read().split("for r in (")[0])
BEM=BE+"/members"
def export_raw(path,out):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    assert (ck["args"].get("a_mode") or "none")=="none",path
    a={k:v.detach().cpu().float().numpy() for k,v in ck["model"].items()}
    a["head.2.weight"]=a["head.2.weight"][:2].copy();a["head.2.bias"]=a["head.2.bias"][:2].copy();np.savez(out,**a)
export_raw(H+"/WunderFunder-creative/artifacts/runs/find256r_s4h2/model_ep1.pt",f"{M}/find256r_s4h2.npz")
pkg("par256r",[(f"{M}/find256r_s4h2.npz","none",1.0)])
pkg("final_mix5",[(f"{BEM}/d256_s3h2.npz","none",.2),(f"{M}/find256r_s4h2.npz","none",.2),
                  (f"{M}/findm128_s3h2.npz","demean",.2),(f"{M}/findm128_s4h2.npz","demean",.2),(f"{M}/findmc500_s3h2.npz","demean",.2)])
