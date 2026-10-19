import json,base64,subprocess,hashlib,datetime
from pathlib import Path
B=Path('/home/grinchenkoeg/computer-vision-homework-20261018');O=B/'research_tmp/fg19_teacher_profile_v3';R=B/'data/artifacts/runs/fg19_t3_h384x2_profile_s3_v3';U='computer-vision-homework-fg19-teacher-profile-v3.service'
def show(u):
 p=subprocess.run(['systemctl','--user','show','--all',u,*sum((['-p',k] for k in ['InvocationID','LoadState','ActiveState','SubState','ExecMainCode','ExecMainStatus','ControlGroup','MainPID','CPUUsageNSec','MemoryCurrent','MemoryPeak','ExecStart','ExecMainStartTimestamp','ExecMainExitTimestamp']),[])],capture_output=True,text=True,timeout=15);assert p.returncode==0;return dict(a.split('=',1) for a in p.stdout.splitlines())
u=show(U);assert u['InvocationID']=='68b2f648a5c24dc28bd5ee65aabc4a45';p=show('wnn-campaign.slice');assert p['ControlGroup'] and p['ControlGroup']!='/'
g=Path('/sys/fs/cgroup')/p['ControlGroup'].lstrip('/');paths=[g/'cgroup.procs',*g.rglob('cgroup.procs')];assert len(paths)<128
f={}
for n in ['ADMISSION.json','LAUNCH.json','unit.log']:
 q=O/n
 if q.exists():
  assert q.stat().st_size<2000000;a=q.read_bytes();f[n]={'bytes':len(a),'sha256':hashlib.sha256(a).hexdigest(),'base64':base64.b64encode(a).decode()}
run_inventory={str(q.relative_to(R)):{'bytes':q.stat().st_size} for q in R.iterdir() if q.is_file()} if R.exists() else None
own=None
if u['ControlGroup']:
 q=Path('/sys/fs/cgroup')/u['ControlGroup'].lstrip('/');own={n:(q/n).read_text() if (q/n).exists() else None for n in ['memory.current','memory.events','memory.events.local','cgroup.procs']}
print(json.dumps(dict(utc=datetime.datetime.now(datetime.UTC).isoformat(),unit=u,parent=p,events={n:(g/n).read_text() for n in ['memory.current','memory.events','memory.events.local']},populated={str(q):q.read_text() for q in paths if q.read_text().strip()},own=own,files=f,run_inventory_stat_only=run_inventory)))
