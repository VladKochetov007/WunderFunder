import json,shutil,os,zipfile,hashlib
import numpy as np,torch
H=os.path.expanduser("~");D=H+"/WunderFunder-creative/research/iterations/2026-10-02-big-ensemble"
runs={"find256_s3":"WunderFunder/artifacts/runs/find256_s3/model_ep3.pt","fin128_s3":"WunderFunder/artifacts/runs/fin128_s3/model_ep3.pt",
      "fin128_s4":"WunderFunder/artifacts/runs/fin128_s4/model_ep3.pt","finc500b_s3":"WunderFunder/artifacts/runs/finc500b_s3/model_ep3.pt"}
d=f"{D}/pkg_v1par";shutil.rmtree(d,ignore_errors=True);os.makedirs(d)
for sub in ("v2","v3","avx2"):shutil.copytree(f"{D}/pkg_e7/{sub}",f"{d}/{sub}")
shutil.copy(f"{D}/pkg_e7/solution.py",d);mem=[]
for i,(n,p) in enumerate(runs.items()):
    ck=torch.load(f"{H}/{p}",map_location="cpu",weights_only=False)
    assert not ck["args"].get("a_mode") or ck["args"]["a_mode"]=="none"
    a={k:v.detach().cpu().float().numpy() for k,v in ck["model"].items()}
    print(n,"head rows",a["head.2.weight"].shape)
    a["head.2.weight"]=a["head.2.weight"][:2].copy();a["head.2.bias"]=a["head.2.bias"][:2].copy()
    np.savez(f"{d}/weights_{i}.npz",**a);mem.append({"weights":f"weights_{i}.npz","coef":0.25,"name":n+"@ep3"})
json.dump(mem,open(f"{d}/members.json","w"),indent=1)
with zipfile.ZipFile(f"{D}/v1par.zip","w",zipfile.ZIP_DEFLATED) as z:
    for root,_,fs in os.walk(d):
        for f in fs:p=os.path.join(root,f);z.write(p,os.path.relpath(p,d))
print(hashlib.sha256(open(f"{D}/v1par.zip","rb").read()).hexdigest()[:16])
