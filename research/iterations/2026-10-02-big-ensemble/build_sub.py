exec(open("build_vlad.py").read().split("pkgN(\"vlad_br\"")[0])
import json
_pkgN=pkgN
def pkgN(tag,members):
    _pkgN(tag,members)
    p=f"{D}/pkg_{tag}/members.json";m=json.load(open(p))
    for e in m:e["name"]=e["name"].replace("vlad:","d256c:")
    json.dump(m,open(p,"w"),indent=1)
    with zipfile.ZipFile(f"{D}/{tag}.zip","w",zipfile.ZIP_DEFLATED) as z:
        for root,_,fs in os.walk(f"{D}/pkg_{tag}"):
            for f in fs:q=os.path.join(root,f);z.write(q,os.path.relpath(q,f"{D}/pkg_{tag}"))
    print(tag,hashlib.sha256(open(f"{D}/{tag}.zip","rb").read()).hexdigest()[:16])
SUB=lambda c:[("raw",f"{R}/f{z}_s3ft/model_ep1.pt",c) for z in ("za","zvol","zpxev")]
pkgN("blend_pair_sub",[("vlad",VL,0.25)]+PAIR(0.15)+SUB(0.1)+[BASE])
