
import json,glob
for p in glob.glob("/root/run/refill/pending/*w14_*.json")+glob.glob("/root/run/refill/pending/*w13*.json"):
    d=json.load(open(p))
    for s in d["steps"]:
        a=s["argv"]
        if "--teacher-epoch" not in a and any("pipeline" in x for x in a): a+=["--teacher-epoch","2"]
    json.dump(d,open(p+".tmp","w"),indent=1); import os; os.replace(p+".tmp",p); print("patched",p)
