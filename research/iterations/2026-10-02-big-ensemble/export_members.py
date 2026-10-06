"""Export full-data checkpoints to wnn_cpp npz (same transform as midpoint package_student.py)."""
import json,sys,hashlib
from pathlib import Path
import numpy as np,torch
H=Path.home()
M={
 "d256_s3h2":"WunderFunder/artifacts/runs/find256_s3h2/model_ep1.pt",
 "f256_s3h":"WunderFunder/artifacts/runs/fin256_s3h/model_ep0.pt",
 "f256_s4h":"WunderFunder/artifacts/runs/fin256_s4h/model_ep0.pt",
 "h128_s3h2":"WunderFunder/artifacts/runs/fin128_s3h2/model_ep1.pt",
 "h128_s4h2":"WunderFunder/artifacts/runs/fin128_s4h2/model_ep1.pt",
 "c500_s3h2":"WunderFunder/artifacts/runs/finc500b_s3h2/model_ep1.pt",
 "wm_s3h2":"WunderFunder-creative/artifacts/runs/oct02_wm_final_s3h2/model_ep1.pt",
 "ow_s3h2":"WunderFunder-creative/artifacts/runs/oct02_ow_final_s3h2/model_ep1.pt",
 "sh256_s3h2":"WunderFunder-creative/artifacts/runs/cont_sh256_final_s3h2/model_ep1.pt",
}
out=Path("members");out.mkdir(exist_ok=True);rec={}
for name,p in M.items():
    p=H/p
    if not p.exists(): print("MISSING",name,p);continue
    ck=torch.load(p,map_location="cpu",weights_only=False)
    a={k:v.detach().cpu().float().numpy() for k,v in ck["model"].items()}
    a["head.2.weight"]=a["head.2.weight"][:2].copy();a["head.2.bias"]=a["head.2.bias"][:2].copy()
    np.savez(out/f"{name}.npz",**a)
    args=ck.get("args",{})
    rec[name]=dict(src=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),hidden=args.get("hidden"),layers=args.get("layers"),keys=len(a))
    print(name,rec[name]["hidden"],rec[name]["layers"],len(a))
json.dump(rec,open("members.json","w"),indent=1)
