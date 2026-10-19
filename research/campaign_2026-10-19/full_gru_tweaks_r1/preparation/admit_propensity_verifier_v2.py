"""Submit one capped CPU terminal verifier after the exact native zero exit."""
import base64,datetime,hashlib,json,subprocess
from pathlib import Path
C=Path(__file__).resolve().parents[1];P=C/'preparation';R=C/'records/propensity_f0_e4_v3/terminal_verification_v2'
names=['verify_propensity_terminal.py','propensity_terminal_checks.py','propensity_terminal_support.py']
files={n:(P/n).read_bytes() for n in names};files['BINDING.json']=(C/'records/propensity_f0_e4_v3/BINDING.json').read_bytes()
t=json.loads((P/'PROPENSITY_TERMINAL_TEST_RECEIPT.json').read_bytes())
assert all(hashlib.sha256(files[n]).hexdigest()==t['implementation_files'][n] for n in names)
assert hashlib.sha256(files['BINDING.json']).hexdigest()==t['binding_sha256']
payload={k:{'sha256':hashlib.sha256(v).hexdigest(),'base64':base64.b64encode(v).decode()} for k,v in files.items()}
body=r'''import base64,json,hashlib,subprocess,os,datetime
from pathlib import Path
P=PAYLOAD
B=Path('/home/grinchenkoeg/computer-vision-homework-20261018');D=B/'research_tmp/fg19_propensity_verify_v2';U='computer-vision-homework-fg19-propensity-verify-v2.service';N='computer-vision-homework-fg19-propensity-f0-e4-v3.service'
def show(u):
 p=subprocess.run(['systemctl','--user','show','--all',u,*sum((['-p',k] for k in ['LoadState','InvocationID','SubState','ActiveState','ExecMainCode','ExecMainStatus','ControlGroup','MemoryCurrent','CPUQuotaPerSecUSec','MemoryHigh','MemoryMax','MemorySwapMax']),[])],capture_output=True,text=True,timeout=15);assert p.returncode==0;return dict(a.split('=',1) for a in p.stdout.splitlines())
n=show(N);assert n['InvocationID']=='85b87a83f96945f4b78908f124435a2b' and n['SubState']=='exited' and int(n['ExecMainCode'])==1 and int(n['ExecMainStatus'])==0
assert show(U)['LoadState']=='not-found' and not D.exists() and not (B/'runtime/STOP_ADMISSION.json').exists()
p=show('wnn-campaign.slice');assert p['ControlGroup'] and p['ControlGroup']!='/' and p['CPUQuotaPerSecUSec']=='2s' and p['MemoryHigh']=='18000000000' and p['MemoryMax']=='20000000000' and p['MemorySwapMax']=='0'
g=Path('/sys/fs/cgroup')/p['ControlGroup'].lstrip('/');paths=[g/'cgroup.procs',*g.rglob('cgroup.procs')];assert len(paths)<128
pop={str(x):x.read_text() for x in paths if x.read_text().strip()};assert not pop,pop
assert int(p['MemoryCurrent'])+1500000000<17000000000
assert os.statvfs(B).f_bavail*os.statvfs(B).f_frsize>20000000000
fstype=subprocess.check_output(['findmnt','-T',str(B/'research_tmp'),'-n','-o','FSTYPE'],text=True).strip();assert fstype not in ['tmpfs','ramfs']
D.mkdir()
for name,item in P.items():
 data=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(data).hexdigest()==item['sha256'];(D/name).write_bytes(data)
a={'utc':datetime.datetime.now(datetime.UTC).isoformat(),'native':n,'parent':p,'populated':pop,'events':(g/'memory.events').read_text(),'events_local':(g/'memory.events.local').read_text(),'files':{k:v['sha256'] for k,v in P.items()}}
(D/'ADMISSION.json').write_text(json.dumps(a,indent=2)+'\n')
cmd=['systemd-run','--user','--unit='+U,'--property=Type=exec','--property=RemainAfterExit=yes','--property=Slice=wnn-campaign.slice','--property=CPUQuota=50%','--property=MemoryHigh=1400000000','--property=MemoryMax=1500000000','--property=MemorySwapMax=0','--property=RuntimeMaxSec=120','--property=StandardOutput=append:'+str(D/'unit.log'),'--property=StandardError=append:'+str(D/'unit.log'),'--working-directory='+str(D)]
for k,v in {'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TMPDIR':str(B/'research_tmp')}.items():cmd+=['--setenv='+k+'='+v]
cmd+=['/home/grinchenkoeg/WunderFunder/.venv/bin/python','-u',str(D/'verify_propensity_terminal.py'),'--run-dir',str(B/'data/artifacts/runs/fg19_propensity_clean_f0_seed0_e4'),'--source-root',str(B/'source-fg19-propensity-235db75'),'--binding',str(D/'BINDING.json'),'--split',str(B/'splits/fold0.json'),'--output',str(D/'VERIFICATION.json'),'--invocation-id','85b87a83f96945f4b78908f124435a2b','--native-exit-status','0','--expected-binding-sha256','babfd92939eb5580933929a591872dec9056aaacd5002979a26ca2de8ed6ff00','--expected-split-sha256','c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7']
r=subprocess.run(cmd,capture_output=True,text=True,timeout=20);launch={'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'post':show(U)};(D/'LAUNCH.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps({'admission':a,'launch':launch}))
'''.replace('P=PAYLOAD','P='+repr(payload))
compile(body,'remote','exec');R.mkdir(exist_ok=False);(R/'REMOTE.py').write_text(body)
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','beleriand','python3 -B -'],input=body.encode(),capture_output=True,timeout=60);(R/'STDOUT.raw.json').write_bytes(r.stdout);(R/'STDERR.raw').write_bytes(r.stderr)
assert r.returncode==0,(r.returncode,r.stderr)
x=json.loads(r.stdout)
for k,v in x.items():(R/(k.upper()+'.json')).write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({'post':x['launch']['post'],'launch_returncode':x['launch']['returncode']}))
