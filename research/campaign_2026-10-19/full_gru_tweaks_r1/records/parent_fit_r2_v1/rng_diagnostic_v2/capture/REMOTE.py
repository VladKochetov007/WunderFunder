from pathlib import Path
import base64,datetime,hashlib,json,subprocess

B=Path('/home/grinchenkoeg/computer-vision-homework-20261018');D=B/'research_tmp/fg19_parent_rng_diag_v2'
U='computer-vision-homework-fg19-parent-rng-diag-v2.service';N='computer-vision-homework-fg19-parent-fit-r2-v1.service';NI='5186b3016ff54b1fa754c10a1f87b3e4'
K=('LoadState','ActiveState','SubState','MainPID','InvocationID','Result','ExecMainCode','ExecMainStatus','ControlGroup','CPUUsageNSec','MemoryCurrent','MemoryPeak','MemoryHigh','MemoryMax','MemorySwapMax','CPUQuotaPerSecUSec','ExecMainStartTimestamp','ExecMainExitTimestamp','ExecMainStartTimestampMonotonic','ExecMainExitTimestampMonotonic')
def show(unit):
 r=subprocess.run(['systemctl','--user','show','--all',unit,*sum((['-p',k] for k in K),[])],capture_output=True,text=True,timeout=15);assert r.returncode==0;return dict(x.split('=',1) for x in r.stdout.splitlines() if '=' in x)
def read(p,cap):
 a=p.lstat();assert p.is_file() and not p.is_symlink() and a.st_size<=cap
 b=p.read_bytes();z=p.stat();assert len(b)==a.st_size and (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns);return b,z
assert D.is_dir() and not D.is_symlink()
files={}
def add(name,cap,required=True):
 p=D/name
 if not p.exists():
  assert not required;return None
 b,z=read(p,cap);files[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()};return b
admission_raw=add('RNG_DIAGNOSTIC_ADMISSION.json',1_000_000);launch_raw=add('RNG_DIAGNOSTIC_LAUNCH.json',1_000_000);add('unit.log',1_000_000);diagnostic_raw=add('RNG_DIAGNOSTIC.json',256_000,False)
assert sum(x['bytes'] for x in files.values())<=2_256_000
a=json.loads(admission_raw);l=json.loads(launch_raw);assert a['unit']==U and a['native_unit']==N and a['native_invocation_id']==NI and l['unit']==U
inv=l['invocation_id'];assert isinstance(inv,str) and len(inv)==32
u=show(U);assert u['InvocationID']==inv
n=show(N);assert n['InvocationID']==NI and (n['ActiveState'],n['SubState'],n['MainPID'],n['ExecMainCode'],n['ExecMainStatus'])==('active','exited','0','1','0')
p=D/'RNG_DIAGNOSTIC.py';src,z=read(p,12_000);sha=hashlib.sha256(src).hexdigest();stage=a['staged_metadata_and_source']['RNG_DIAGNOSTIC.py'];assert (stage['bytes'],stage['sha256'])==(len(src),sha)
source={'path':str(p),'bytes':len(src),'device':z.st_dev,'inode':z.st_ino,'mtime_ns':z.st_mtime_ns,'sha256':sha}
cg=show('wnn-campaign.slice');assert cg['ControlGroup'] not in ('','/');g=Path('/sys/fs/cgroup')/cg['ControlGroup'].lstrip('/');ps=[g/'cgroup.procs',*g.rglob('cgroup.procs')];assert len(ps)<128
pop={str(x):v for x in ps if x.is_file() and (v:=x.read_text().strip())};pe={k:(g/k).read_text() for k in ('memory.current','memory.events','memory.events.local') if (g/k).is_file()}
own=None
if u['ControlGroup'] not in ('','/'):
 h=Path('/sys/fs/cgroup')/u['ControlGroup'].lstrip('/');own={k:(h/k).read_text() for k in ('memory.current','memory.peak','memory.events','memory.events.local') if (h/k).is_file()}
status=json.loads(diagnostic_raw).get('status') if diagnostic_raw else None
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit_name':U,'invocation_id':inv,'unit':u,'native_unit_name':N,'native_invocation_id':NI,'native':n,'diagnostic_status':status,'diagnostic_source':source,'parent':cg,'parent_events':pe,'own_cgroup':own,'populated':pop,'files':files,'scope':'Bounded diagnostic metadata/log and source stat/hash only; no model, checkpoint, training-state, or array payload read or hashed. Empty own cgroup is unavailable, never host root.'},sort_keys=True))
