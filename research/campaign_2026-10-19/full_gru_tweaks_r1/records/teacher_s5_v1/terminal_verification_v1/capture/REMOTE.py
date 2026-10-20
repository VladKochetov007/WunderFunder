import json,subprocess,base64,hashlib,datetime
from pathlib import Path
B=Path('/home/grinchenkoeg/computer-vision-homework-20261018');D=B/'research_tmp/fg19_teacher_s5_verify_v1';U='computer-vision-homework-fg19-teacher-s5-verify-v1.service'
def show(u):
 r=subprocess.run(['systemctl','--user','show','--all',u,*sum((['-p',k] for k in ['LoadState','MainPID','ExecMainStartTimestampMonotonic','ExecMainExitTimestampMonotonic','InvocationID','ActiveState','SubState','ExecMainCode','ExecMainStatus','ControlGroup','CPUUsageNSec','MemoryPeak','ExecStart','ExecMainStartTimestamp','ExecMainExitTimestamp']),[])],capture_output=True,text=True,timeout=15);assert r.returncode==0;return dict(a.split('=',1) for a in r.stdout.splitlines())
u=show(U);assert u['InvocationID']=='6594891a6fea423f907e7c478da0af31'
f={}
for n in ['VERIFICATION.json','unit.log','VERIFIER_ADMISSION.json','VERIFIER_LAUNCH.json','TERMINAL_EVIDENCE.json']:
 p=D/n
 if p.exists():
  assert p.stat().st_size<1000000;v=p.read_bytes();f[n]={'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest(),'base64':base64.b64encode(v).decode()}
parent=show('wnn-campaign.slice')
cp=parent['ControlGroup'];assert cp and cp!='/'
cg=Path('/sys/fs/cgroup')/cp.lstrip('/')
parent_events={name:(cg/name).read_text() for name in ('memory.current','memory.events','memory.events.local')}
print(json.dumps(dict(parent=parent,parent_events=parent_events,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),unit=u,native=show('computer-vision-homework-fg19-teacher-s5-v1.service'),files=f)))
