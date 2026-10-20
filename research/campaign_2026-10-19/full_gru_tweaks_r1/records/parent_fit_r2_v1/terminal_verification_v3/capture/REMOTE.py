from pathlib import Path
import json,subprocess,base64,hashlib,datetime
B=Path('/home/grinchenkoeg/computer-vision-homework-20261018');D=B/'research_tmp/fg19_parent_fit_r2_verify_v3';U='computer-vision-homework-fg19-parent-fit-r2-verify-v3.service'
def show(u):
 keys=['LoadState','ActiveState','SubState','MainPID','InvocationID','Result','ExecMainCode','ExecMainStatus','ControlGroup','CPUUsageNSec','MemoryCurrent','MemoryPeak','MemoryHigh','MemoryMax','MemorySwapMax','CPUQuotaPerSecUSec','ExecMainStartTimestamp','ExecMainExitTimestamp','ExecMainStartTimestampMonotonic','ExecMainExitTimestampMonotonic']
 r=subprocess.run(['systemctl','--user','show','--all',u,*sum((['-p',k] for k in keys),[])],capture_output=True,text=True,timeout=10);assert r.returncode==0;return dict(x.split('=',1) for x in r.stdout.splitlines() if '=' in x)
assert D.is_dir() and not D.is_symlink();launch=json.loads((D/'VERIFIER_LAUNCH.json').read_bytes());inv=launch['invocation_id'];assert isinstance(inv,str) and len(inv)==32
u=show(U);assert u['InvocationID']==inv
parent=show('wnn-campaign.slice');assert parent['ControlGroup'] not in ('','/');cg=Path('/sys/fs/cgroup')/parent['ControlGroup'].lstrip('/');ps=[cg/'cgroup.procs',*cg.rglob('cgroup.procs')];assert len(ps)<128;pop={str(p):v for p in ps if (v:=p.read_text().strip())}
own=None
if u['ControlGroup'] not in ('','/'):
 g=Path('/sys/fs/cgroup')/u['ControlGroup'].lstrip('/');own={n:(g/n).read_text() for n in ('memory.current','memory.peak','memory.events','memory.events.local') if (g/n).exists()}
files={}
def add(n):
 p=D/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size<1000000;b=p.read_bytes();files[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
for n in ('VERIFIER_ADMISSION.json','VERIFIER_LAUNCH.json','unit.log'):add(n)
terminal=(u['ActiveState'],u['SubState'],u['MainPID'],u['ExecMainCode'],u['ExecMainStatus'])==('active','exited','0','1','0')
if terminal:add('VERIFICATION.json')
assert sum(r['bytes'] for r in files.values())<4000000
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':u,'invocation_id':inv,'terminal_success':terminal,'parent':parent,'parent_events':{n:(cg/n).read_text() for n in ('memory.current','memory.events','memory.events.local')},'own_files':own,'populated':pop,'files':files,'scope':'Bounded verifier metadata/log capture only; no observer model or array payload read/hash; empty own cgroup is unavailable, never host root.'}))
