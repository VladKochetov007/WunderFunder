import subprocess,json,hashlib,base64,datetime
from pathlib import Path
B=Path('/home/grinchenkoeg/computer-vision-homework-20261018');D=B/'research_tmp/fg19_teacher_cache_profile_v1';U='computer-vision-homework-fg19-teacher-cache-profile-v1.service'
def show(u):
 r=subprocess.run(['systemctl','--user','show','--all',u,*sum((['-p',k] for k in ['LoadState','MainPID','ExecMainStartTimestampMonotonic','ExecMainExitTimestampMonotonic','InvocationID','ActiveState','SubState','ExecMainCode','ExecMainStatus','ControlGroup','CPUUsageNSec','MemoryPeak','MemoryCurrent','ExecStart','ExecMainStartTimestamp','ExecMainExitTimestamp','Result']),[])],capture_output=True,text=True,timeout=15);assert r.returncode==0;return dict(a.split('=',1) for a in r.stdout.splitlines())
u=show(U);assert u['InvocationID']=='983d65b923c54f05a25aa98d61e0bb1d'
f={}
if u['SubState'] in ['exited','failed']:
 for n in ['PROFILE.json','ADMISSION.json','LAUNCH.json','unit.log']:
  p=D/n
  if p.exists():
   assert p.stat().st_size<1000000;v=p.read_bytes();f[n]={'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest(),'base64':base64.b64encode(v).decode()}
parent=show('wnn-campaign.slice');cp=parent['ControlGroup'];assert cp and cp!='/'
cg=Path('/sys/fs/cgroup')/cp.lstrip('/');paths=[cg/'cgroup.procs',*cg.rglob('cgroup.procs')];assert len(paths)<128
pop={str(p):v for p in paths if (v:=p.read_text().strip())}
events={n:(cg/n).read_text() for n in ['memory.current','memory.events','memory.events.local']}
own=None
if u['ControlGroup']:
 g=Path('/sys/fs/cgroup')/u['ControlGroup'].lstrip('/');assert g!=Path('/sys/fs/cgroup');own={n:(g/n).read_text() for n in ['memory.current','memory.events','memory.events.local'] if (g/n).exists()}
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),unit=u,parent=parent,parent_events=events,populated=pop,cgroup_count=len(paths),own_files=own,files=f)))
