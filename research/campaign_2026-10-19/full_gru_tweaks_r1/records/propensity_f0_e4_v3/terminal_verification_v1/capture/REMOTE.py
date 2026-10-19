import json,subprocess,base64,hashlib,datetime
from pathlib import Path
B=Path('/home/grinchenkoeg/computer-vision-homework-20261018');D=B/'research_tmp/fg19_propensity_verify_v1';U='computer-vision-homework-fg19-propensity-verify-v1.service'
def show(u):
 r=subprocess.run(['systemctl','--user','show','--all',u,*sum((['-p',k] for k in ['InvocationID','ActiveState','SubState','ExecMainCode','ExecMainStatus','ControlGroup','CPUUsageNSec','MemoryPeak','ExecStart','ExecMainStartTimestamp','ExecMainExitTimestamp']),[])],capture_output=True,text=True,timeout=15);assert r.returncode==0;return dict(a.split('=',1) for a in r.stdout.splitlines())
u=show(U);assert u['InvocationID']=='5a4d868aaa054d349a9dcbd99e5ed576'
f={}
for n in ['VERIFICATION.json','unit.log','ADMISSION.json','LAUNCH.json']:
 p=D/n
 if p.exists():
  assert p.stat().st_size<1000000;v=p.read_bytes();f[n]={'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest(),'base64':base64.b64encode(v).decode()}
print(json.dumps(dict(utc=datetime.datetime.now(datetime.UTC).isoformat(),unit=u,native=show('computer-vision-homework-fg19-propensity-f0-e4-v3.service'),files=f)))
