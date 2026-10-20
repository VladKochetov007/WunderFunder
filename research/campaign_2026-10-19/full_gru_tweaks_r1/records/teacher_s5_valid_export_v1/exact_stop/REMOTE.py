import subprocess,json,datetime
from pathlib import Path
units=[('computer-vision-homework-fg19-teacher-s5-valid-export-v1.service','48d87b0c40514a18adbecf80e5a7d1db')]
def show(u):
 r=subprocess.run(['systemctl','--user','show','--all',u,*sum((['-p',k] for k in ['InvocationID','LoadState','ActiveState','SubState','ExecMainCode','ExecMainStatus','ControlGroup','MemoryCurrent','MemoryPeak','CPUUsageNSec']),[])],capture_output=True,text=True,timeout=15);assert r.returncode==0;return dict(a.split('=',1) for a in r.stdout.splitlines())
before={u:show(u) for u,_ in units}
for u,inv in units:
 x=before[u];assert x['InvocationID']==inv and x['SubState']=='exited' and int(x['ExecMainCode'])==1 and int(x['ExecMainStatus'])==0
p=show('wnn-campaign.slice');assert p['ControlGroup'] and p['ControlGroup']!='/'
g=Path('/sys/fs/cgroup')/p['ControlGroup'].lstrip('/');paths=[g/'cgroup.procs',*g.rglob('cgroup.procs')];assert len(paths)<128
pop={str(x):x.read_text() for x in paths if x.read_text().strip()};events={k:(g/k).read_text() for k in ['memory.current','memory.events','memory.events.local']}
stop={}
for u,_ in units:
 r=subprocess.run(['systemctl','--user','stop',u],capture_output=True,text=True,timeout=30);stop[u]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before=before,stop=stop,after={u:show(u) for u,_ in units},parent=events,populated=pop)))
