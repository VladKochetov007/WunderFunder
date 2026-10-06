import sys
exec(open("build_kf.py").read().split("R=H+")[0])
import numpy as np
def pkgmix(tag,paths,coef_pair,coef_base):
    d=f"{D}/pkg_{tag}";shutil.rmtree(d,ignore_errors=True);os.makedirs(d)
    for sub in ("v2","v3","avx2"):shutil.copytree(f"{D}/pkg_e7/{sub}",f"{d}/{sub}")
    shutil.copy(f"{D}/solution_mix.py",f"{d}/solution.py");mem=[]
    for i,p in enumerate(paths):
        exp(p,f"{d}/weights_{i}.npz");mem.append({"weights":f"weights_{i}.npz","coef":coef_pair/len(paths),"name":p.split("/")[-2]})
    b=dict(np.load(H+"/WunderFunder-creative/artifacts/orgbase.npz"));np.savez(f"{d}/weights_base.npz",**b)
    mem.append({"weights":"weights_base.npz","coef":coef_base,"name":"organizer_baseline","fp32":True})
    json.dump(mem,open(f"{d}/members.json","w"),indent=1)
    with zipfile.ZipFile(f"{D}/{tag}.zip","w",zipfile.ZIP_DEFLATED) as z:
        for root,_,fs in os.walk(d):
            for f in fs:q=os.path.join(root,f);z.write(q,os.path.relpath(q,d))
    print(tag,hashlib.sha256(open(f"{D}/{tag}.zip","rb").read()).hexdigest()[:16],mem)
R=H+"/WunderFunder-creative/artifacts/runs"
w=0.25
pkgmix("pair_fullft_base",[f"{R}/ffull_s3/model.pt",f"{R}/ffull_s4/model.pt"],1-w,w*0.25)
pkgmix("pair_wise05_base",[f"{R}/wise05full_s3/model.pt",f"{R}/wise05full_s4/model.pt"],1-w,w*0.25)
