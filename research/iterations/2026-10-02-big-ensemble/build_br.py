exec(open("build_kf.py").read().split("R=H+")[0])
import numpy as np
from raw_export import exp_raw
def pkg3(tag,pair,raws,cp,cr,cb):
    d=f"{D}/pkg_{tag}";shutil.rmtree(d,ignore_errors=True);os.makedirs(d)
    for sub in ("v2","v3","avx2"):shutil.copytree(f"{D}/pkg_e7/{sub}",f"{d}/{sub}")
    shutil.copy(f"{D}/solution_mix.py",f"{d}/solution.py");mem=[]
    for i,p in enumerate(pair):exp(p,f"{d}/weights_{i}.npz");mem.append({"weights":f"weights_{i}.npz","coef":cp/len(pair),"name":p.split("/")[-2]})
    for i,p in enumerate(raws):exp_raw(p,f"{d}/weights_raw{i}.npz");mem.append({"weights":f"weights_raw{i}.npz","coef":cr/len(raws),"name":p.split("/")[-2]})
    np.savez(f"{d}/weights_base.npz",**dict(np.load(H+"/WunderFunder-creative/artifacts/orgbase.npz")))
    mem.append({"weights":"weights_base.npz","coef":cb,"name":"organizer_baseline","fp32":True})
    json.dump(mem,open(f"{d}/members.json","w"),indent=1)
    with zipfile.ZipFile(f"{D}/{tag}.zip","w",zipfile.ZIP_DEFLATED) as z:
        for root,_,fs in os.walk(d):
            for f in fs:q=os.path.join(root,f);z.write(q,os.path.relpath(q,d))
    print(tag,hashlib.sha256(open(f"{D}/{tag}.zip","rb").read()).hexdigest()[:16],[(m["name"],round(m["coef"],4)) for m in mem])
R=H+"/WunderFunder-creative/artifacts/runs"
raws=[f"{R}/frawc_s3ft/model_ep1.pt",f"{R}/frawc_s4ft/model_ep1.pt"]
pkg3("pair_fullft_br",[f"{R}/ffull_s3/model.pt",f"{R}/ffull_s4/model.pt"],raws,0.6,0.15,0.0625)
pkg3("pair_wise05_br",[f"{R}/wise05full_s3/model.pt",f"{R}/wise05full_s4/model.pt"],raws,0.6,0.15,0.0625)
