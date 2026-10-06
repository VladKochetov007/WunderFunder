import json,shutil,os,zipfile,hashlib
import numpy as np,torch
H=os.path.expanduser("~");D=H+"/WunderFunder-creative/research/iterations/2026-10-02-big-ensemble"
def exp(path,out):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    a={k:v.detach().cpu().float().numpy() for k,v in ck["model"].items()}
    a["head.2.weight"]=a["head.2.weight"][:2].copy();a["head.2.bias"]=a["head.2.bias"][:2].copy();np.savez(out,**a);return ck["args"]
def pkg(tag,paths):
    d=f"{D}/pkg_{tag}";shutil.rmtree(d,ignore_errors=True);os.makedirs(d)
    for sub in ("v2","v3","avx2"):shutil.copytree(f"{D}/pkg_e7/{sub}",f"{d}/{sub}")
    shutil.copy(f"{D}/pkg_e7/solution.py",d);mem=[]
    for i,p in enumerate(paths):
        a=exp(p,f"{d}/weights_{i}.npz");mem.append({"weights":f"weights_{i}.npz","coef":1/len(paths),"name":p.split("/")[-2]})
        print(tag,p.split("/")[-2],a.get("hidden"),a.get("valid_fold"),a.get("train_source"),a.get("init"))
    json.dump(mem,open(f"{d}/members.json","w"),indent=1)
    with zipfile.ZipFile(f"{D}/{tag}.zip","w",zipfile.ZIP_DEFLATED) as z:
        for root,_,fs in os.walk(d):
            for f in fs:q=os.path.join(root,f);z.write(q,os.path.relpath(q,d))
    print(tag,hashlib.sha256(open(f"{D}/{tag}.zip","rb").read()).hexdigest()[:16])
R=H+"/WunderFunder/artifacts/runs"
pkg("kf_h128",[f"{R}/{n}_f{k}h/model.pt" for n in ("ft128","ft128s4") for k in range(4)])
pkg("full_h128",[f"{R}/fin128_s3h/model.pt",f"{R}/fin128_s4h/model.pt"])
