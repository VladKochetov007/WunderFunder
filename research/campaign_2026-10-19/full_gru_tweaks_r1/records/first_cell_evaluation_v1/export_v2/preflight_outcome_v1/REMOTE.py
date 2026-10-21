import json,os,subprocess
from datetime import UTC,datetime
unit="computer-vision-homework-fg19-first-cell-export-v1.service"
d="/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_evaluation_v1"
args=["systemctl","--user","show","--all",unit]
for k in ("Id","LoadState","ActiveState","SubState","MainPID","InvocationID"):args += ["-p",k]
r=subprocess.run(args,capture_output=True,text=True,timeout=12,check=False)
if r.returncode:raise RuntimeError("new-unit outcome query failed")
props=dict(x.split("=",1) for x in r.stdout.splitlines() if "=" in x)
print(json.dumps({"schema":"fg19.first-cell-export-preflight-outcome.v2","captured_utc":datetime.now(UTC).isoformat(),"unit":unit,"unit_properties":props,"evidence_path":d,"evidence_lexists":os.path.lexists(d),"no_mutations":True,"payloads_read_or_hashed":False},sort_keys=True))
