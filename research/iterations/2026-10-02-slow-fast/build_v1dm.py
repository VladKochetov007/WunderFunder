import json,shutil,os,zipfile,hashlib
import numpy as np,torch
H=os.path.expanduser("~");D=H+"/WunderFunder-creative/research/iterations/2026-10-02-slow-fast"
BE=H+"/WunderFunder-creative/research/iterations/2026-10-02-big-ensemble";M=D+"/members"
def export(run,ep,out):
    ck=torch.load(f"{H}/WunderFunder-creative/artifacts/runs/{run}/model_ep{ep}.pt",map_location="cpu",weights_only=False)
    assert ck["args"].get("a_mode")=="demean",run
    a={k:v.detach().cpu().float().numpy() for k,v in ck["model"].items()}
    a["head.2.weight"]=a["head.2.weight"][:2].copy();a["head.2.bias"]=a["head.2.bias"][:2].copy()
    np.savez(out,**a)
def pkg(tag,members):
    d=f"{D}/pkg_{tag}";shutil.rmtree(d,ignore_errors=True);os.makedirs(d)
    for sub in ("v2","v3","avx2"):shutil.copytree(f"{BE}/pkg_e7/{sub}",f"{d}/{sub}")
    shutil.copy(f"{D}/solution_dm.py",f"{d}/solution.py");mem=[]
    for i,(src,mode,coef) in enumerate(members):
        shutil.copy(src,f"{d}/weights_{i}.npz");mem.append({"weights":f"weights_{i}.npz","coef":coef,"a_mode":mode,"name":os.path.basename(src)})
    json.dump(mem,open(f"{d}/members.json","w"),indent=1)
    with zipfile.ZipFile(f"{D}/{tag}.zip","w",zipfile.ZIP_DEFLATED) as z:
        for root,_,fs in os.walk(d):
            for f in fs:p=os.path.join(root,f);z.write(p,os.path.relpath(p,d))
    print(tag,hashlib.sha256(open(f"{D}/{tag}.zip","rb").read()).hexdigest()[:16])
for r in ("findm256_s3h2","findm256_s4h2"):export(r,1,f"{M}/{r}.npz")
pkg("par256",[(f"{M}/findm256_s3h2.npz","demean",1.0)])
pkg("v1_dm",[(f"{M}/findm256_s3h2.npz","demean",.25),(f"{M}/findm128_s3h2.npz","demean",.25),(f"{M}/findm128_s4h2.npz","demean",.25),(f"{M}/findmc500_s3h2.npz","demean",.25)])
pkg("v1_dm5",[(f"{M}/findm256_s3h2.npz","demean",.2),(f"{M}/findm256_s4h2.npz","demean",.2),(f"{M}/findm128_s3h2.npz","demean",.2),(f"{M}/findm128_s4h2.npz","demean",.2),(f"{M}/findmc500_s3h2.npz","demean",.2)])
