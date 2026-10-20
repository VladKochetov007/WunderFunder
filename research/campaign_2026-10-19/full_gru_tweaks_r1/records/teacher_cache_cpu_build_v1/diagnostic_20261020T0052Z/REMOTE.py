import json,subprocess,os,datetime
from pathlib import Path
u='computer-vision-homework-fg19-teacher-cache-cpu-build-v1.service';keys=['InvocationID','MainPID','ActiveState','SubState','ExecMainCode','ExecMainStatus','Result','CPUUsageNSec','MemoryCurrent','MemoryPeak','ControlGroup','ExecMainExitTimestamp']
r=subprocess.run(['systemctl','--user','show','--all',u,*sum((['-p',k] for k in keys),[])],capture_output=True,text=True,timeout=10);assert r.returncode==0;x=dict(v.split('=',1) for v in r.stdout.splitlines() if '=' in v);assert x['InvocationID']=='b131699c99e541c583f050b01a286f03';pid=int(x['MainPID']);d={};fds=[]
if pid:
 p=Path('/proc')/str(pid)
 for name in ['io','status','wchan']:
  q=p/name
  try:
   with q.open('rb') as f:raw=f.read(8193)
   assert len(raw)<=8192;d[name]=raw.decode(errors='replace')
  except FileNotFoundError:d[name]='process exited during observation'
 try:
  entries=list((p/'fd').iterdir());assert len(entries)<64
  for q in entries:
   try:fds.append({'fd':q.name,'target':os.readlink(q)})
   except FileNotFoundError:pass
 except FileNotFoundError:pass
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':x,'own_process_metadata':d,'own_fd_links_only':fds,'scope':'Exact owned PID read-only bounded proc metadata/fd links; no descriptor payload read, stack attach, signals, model/inference/scorer work or mutation.'}))
