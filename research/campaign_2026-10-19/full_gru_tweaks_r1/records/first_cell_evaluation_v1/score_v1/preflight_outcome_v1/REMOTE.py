import json,os,subprocess,stat
from datetime import datetime,timezone
from pathlib import Path
UNIT="computer-vision-homework-fg19-first-cell-score-v1.service"
B=Path("/home/grinchenkoeg/computer-vision-homework-20261018")
paths={"score_directory":B/"research_tmp/fg19_first_cell_f0_s1001_score_v1","result":B/"research_tmp/fg19_first_cell_f0_s1001_evaluation_v3/RESULT.json"}
observed={}
for k,p in paths.items():
 try:
  s=p.lstat();observed[k]={"path":str(p),"exists":True,"mode":s.st_mode,"bytes":s.st_size,"device":s.st_dev,"inode":s.st_ino}
 except FileNotFoundError:observed[k]={"path":str(p),"exists":False}
props="Id,LoadState,ActiveState,SubState,MainPID,InvocationID,ExecMainCode,ExecMainStatus,Result,ControlGroup"
p=subprocess.run(["systemctl","--user","show","--all","--property="+props,UNIT],capture_output=True,timeout=10,check=False)
if len(p.stdout)>32768 or len(p.stderr)>32768:raise RuntimeError("bounded exact-unit response exceeded cap")
print(json.dumps({"schema":"fg19.first-cell-score-preflight-outcome.v1","captured_utc":datetime.now(timezone.utc).isoformat(),"unit":UNIT,"systemctl_returncode":p.returncode,"stdout":p.stdout.decode(),"stderr":p.stderr.decode(),"paths":observed,"read_only":True,"old_units_queried":False,"payloads_read_or_hashed":False}))
