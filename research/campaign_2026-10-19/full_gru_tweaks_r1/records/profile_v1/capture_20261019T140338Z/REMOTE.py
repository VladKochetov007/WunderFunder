
import base64,datetime,json,subprocess,pathlib,hashlib
UNIT='computer-vision-homework-fg19-propensity-profile-v1.service'
OUT=pathlib.Path('/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_propensity_profile_v1')
expected='0698555969d348eebb9b51d7dc39f545'
p=subprocess.run(['systemctl','--user','show','--all',UNIT,'-p','InvocationID','-p','ExecStart','-p','MainPID','-p','ControlGroup','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainCode','-p','ExecMainStatus','-p','CPUUsageNSec','-p','MemoryPeak','-p','ExecMainStartTimestamp','-p','ExecMainExitTimestamp'],capture_output=True,text=True,timeout=15)
assert p.returncode==0,(p.returncode,p.stderr)
props=dict(line.split('=',1) for line in p.stdout.splitlines())
assert props['InvocationID']==expected
parent=subprocess.run(['systemctl','--user','show','--all','wnn-campaign.slice','-p','ControlGroup','-p','MemoryCurrent'],capture_output=True,text=True,timeout=15)
assert parent.returncode==0
pp=dict(line.split('=',1) for line in parent.stdout.splitlines());assert pp['ControlGroup']
cg=pathlib.Path('/sys/fs/cgroup')/pp['ControlGroup'].lstrip('/')
paths=[cg/'cgroup.procs',*cg.rglob('cgroup.procs')];assert len(paths)<128
children={str(x):x.read_text() for x in paths if x.read_text().strip()}
files={}
for name in ['PROFILE_RESULT.json','UPDATE_PROGRESS.jsonl','unit.log']:
 f=OUT/name
 if f.exists():
  assert f.stat().st_size<2000000
  b=f.read_bytes();files[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
record={'created_utc':datetime.datetime.now(datetime.UTC).isoformat(),'expected_invocation':expected,'properties':props,'raw_systemctl':p.stdout,'parent_properties':pp,'parent_events':(cg/'memory.events').read_text(),'parent_events_local':(cg/'memory.events.local').read_text(),'populated_descendants':children,'own_cgroup_files':None,'files':files}
if props.get('ControlGroup'):
 own=pathlib.Path('/sys/fs/cgroup')/props['ControlGroup'].lstrip('/')
 record['own_cgroup_files']={n:(own/n).read_text() if (own/n).exists() else None for n in ['memory.current','memory.events','memory.events.local','cgroup.procs']}
print(json.dumps(record))
