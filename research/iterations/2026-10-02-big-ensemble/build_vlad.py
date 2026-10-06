exec(open("build_kf.py").read().split("R=H+")[0])
import numpy as np
from raw_export import exp_raw
R=H+"/WunderFunder-creative/artifacts/runs";VL=H+"/dsjobs/vlad_s4/weights_0.npz"
def pkgN(tag,members):
    d=f"{D}/pkg_{tag}";shutil.rmtree(d,ignore_errors=True);os.makedirs(d)
    for sub in ("v2","v3","avx2"):shutil.copytree(f"{D}/pkg_e7/{sub}",f"{d}/{sub}")
    shutil.copy(f"{D}/solution_mix.py",f"{d}/solution.py");mem=[]
    for i,(kind,path,coef) in enumerate(members):
        f=f"weights_{i}.npz"
        if kind=="run":exp(path,f"{d}/{f}")
        elif kind=="raw":exp_raw(path,f"{d}/{f}")
        else:np.savez(f"{d}/{f}",**dict(np.load(path)))
        m={"weights":f,"coef":coef,"name":kind+":"+path.split("/")[-2]}
        if kind=="base":m["fp32"]=True
        mem.append(m)
    json.dump(mem,open(f"{d}/members.json","w"),indent=1)
    with zipfile.ZipFile(f"{D}/{tag}.zip","w",zipfile.ZIP_DEFLATED) as z:
        for root,_,fs in os.walk(d):
            for f in fs:q=os.path.join(root,f);z.write(q,os.path.relpath(q,d))
    print(tag,hashlib.sha256(open(f"{D}/{tag}.zip","rb").read()).hexdigest()[:16],[(m["name"],round(m["coef"],4)) for m in mem],flush=True)
BASE=("base",H+"/WunderFunder-creative/artifacts/orgbase.npz",0.0625)
RAW=lambda c:[("raw",f"{R}/frawc_s3ft/model_ep1.pt",c/2),("raw",f"{R}/frawc_s4ft/model_ep1.pt",c/2)]
PAIR=lambda c:[("run",f"{R}/ffull_s3/model.pt",c/2),("run",f"{R}/ffull_s4/model.pt",c/2)]
pkgN("vlad_br",[("vlad",VL,0.6)]+RAW(0.15)+[BASE])
pkgN("vlad_pair",[("vlad",VL,0.5)]+PAIR(0.5))
pkgN("vlad_pair_br",[("vlad",VL,0.35)]+PAIR(0.25)+RAW(0.15)+[BASE])
