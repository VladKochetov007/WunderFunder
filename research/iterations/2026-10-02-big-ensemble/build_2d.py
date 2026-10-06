exec(open("build_bw.py").read().split("pkgN(\"vlad_bw\"")[0])
import json
def neutral(tag):
    p=f"{D}/pkg_{tag}/members.json";m=json.load(open(p))
    for e in m:e["name"]=e["name"].replace("vlad:vlad_s4","d256c:combo_s4")
    json.dump(m,open(p,"w"),indent=1)
    with zipfile.ZipFile(f"{D}/{tag}.zip","w",zipfile.ZIP_DEFLATED) as z:
        for root,_,fs in os.walk(f"{D}/pkg_{tag}"):
            for f in fs:q=os.path.join(root,f);z.write(q,os.path.relpath(q,f"{D}/pkg_{tag}"))
    print(tag,hashlib.sha256(open(f"{D}/{tag}.zip","rb").read()).hexdigest()[:16],flush=True)
D2=lambda c:[("run",f"{R}/fbc1kft_s5/model_ep1.pt",c/2),("run",f"{R}/fbc1kft_s6/model_ep1.pt",c/2)]
pkgN("blend_2d_rw",[("vlad",VL,0.25)]+D2(0.25)+RAWW(0.25)+[BASE]);neutral("blend_2d_rw")
