import os
exec(open(os.path.expanduser("~/WunderFunder-creative/research/iterations/2026-10-02-slow-fast/build_v1dm.py")).read().split("for r in (")[0])
import numpy as np,torch
def export_raw(run,out):
    ck=torch.load(H+f"/WunderFunder-creative/artifacts/runs/{run}/model_ep1.pt",map_location="cpu",weights_only=False)
    assert (ck["args"].get("a_mode") or "none")=="none"
    a={k:v.detach().cpu().float().numpy() for k,v in ck["model"].items()}
    a["head.2.weight"]=a["head.2.weight"][:2].copy();a["head.2.bias"]=a["head.2.bias"][:2].copy();np.savez(out,**a)
names=["fmu256_s3h2","fmu256_s4h2","fmu128_s3h2","fmu128_s4h2","fmuc500_s3h2"]
for n in names:export_raw(n,f"{M}/{n}.npz")
pkg("par_mu",[(f"{M}/fmu128_s3h2.npz","none",1.0)])
pkg("final_mu5",[(f"{M}/{n}.npz","none",.2) for n in names])
