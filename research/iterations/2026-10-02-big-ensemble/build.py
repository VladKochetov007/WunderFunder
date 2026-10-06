import json,shutil,sys,os,zipfile
tag=sys.argv[1];names=sys.argv[2].split(",")
d=f"pkg_{tag}";shutil.rmtree(d,ignore_errors=True);os.makedirs(d)
for sub in ["v2","v3","avx2"]: shutil.copytree(f"pkg_e7/{sub}",f"{d}/{sub}")
shutil.copy("pkg_e7/solution.py",d)
mem=[]
for i,n in enumerate(names):
    shutil.copy(f"members/{n}.npz",f"{d}/weights_{i}.npz");mem.append({"weights":f"weights_{i}.npz","coef":1/len(names),"name":n})
json.dump(mem,open(f"{d}/members.json","w"),indent=1)
with zipfile.ZipFile(f"{tag}.zip","w",zipfile.ZIP_DEFLATED) as z:
    for root,_,fs in os.walk(d):
        for f in fs:
            p=os.path.join(root,f);z.write(p,os.path.relpath(p,d))
