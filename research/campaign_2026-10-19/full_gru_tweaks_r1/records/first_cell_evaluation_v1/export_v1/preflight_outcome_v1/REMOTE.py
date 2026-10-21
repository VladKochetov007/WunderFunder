from __future__ import annotations
import json,os,subprocess
from datetime import UTC,datetime
from pathlib import Path
unit="computer-vision-homework-fg19-first-cell-export-v1.service"
d=Path("/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_evaluation_v1")
support=d.parent/"fg19_child_allthree_verify_v2"
args=["systemctl","--user","show","--all",unit]
for k in ("Id","LoadState","ActiveState","SubState","MainPID","InvocationID"): args += ["-p",k]
r=subprocess.run(args,capture_output=True,text=True,timeout=12,check=False)
if r.returncode: raise RuntimeError("exact new-unit diagnostic failed")
props=dict(x.split("=",1) for x in r.stdout.splitlines() if "=" in x)
files={}
for name in ("parent_fit_terminal_check_v3.py","propensity_terminal_support.py","teacher_s5_terminal_check_v1.py"):
 f=support/name
 s=f.lstat()
 files[name]={"path":str(f),"islink":f.is_symlink(),"bytes":s.st_size,"device":s.st_dev,"inode":s.st_ino,"mtime_ns":s.st_mtime_ns}
print(json.dumps({"schema":"fg19.first-cell-export-preflight-outcome.v1","captured_utc":datetime.now(UTC).isoformat(),"unit":unit,"unit_properties":props,"evidence_path":str(d),"evidence_lexists":os.path.lexists(d),"support_flat_code_stat_only":files,"nested_preparation_lexists":os.path.lexists(support/"preparation"),"no_mutations":True,"model_state_array_payloads_read_or_hashed":False},sort_keys=True))
