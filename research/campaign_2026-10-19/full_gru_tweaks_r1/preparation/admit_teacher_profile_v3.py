"""Reuse immutable source and reviewed gates on one available identical A4000."""
import base64,datetime,hashlib,json,subprocess,zlib
from pathlib import Path
C=Path(__file__).resolve().parents[1];R=C/'records/teacher_profile_v3';P=C/'preparation';old=C/'records/teacher_profile_v2';gpu='GPU-ec74220d-78d7-cc47-69fc-0de1fb580944'
a={'schema':'fg19.single_gpu_reallocation.v1','utc':datetime.datetime.now(datetime.UTC).isoformat(),'human_authority':'одну гпу можно занять','gpu_worker_max':1,'previous_gpu':'GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c','selected_gpu_uuid':gpu,'inventory':'records/GPU_INVENTORY_20261019T1525Z/STDOUT.raw','reason':'One other identical RTX A4000 has10924MiB free. Previous capped profile is exited/failed with no owned GPU process; preserve foreign jobs and select a single available card rather than wait on the crowded card.','resource_policy':'Return to the original4GiB allocator+1GBreserve envelope declared BEFORE the2.5GB attempt, on externally sufficient headroom. No source/model/math/hyperparameter/steps change and no progressive cap search. One worker, sameB provider/runtime,2aggregateCPU/8e9Bworker/20e9Bparent; futurepairedstudentarms remain on same selecteddevice.'}
(C/'GPU_REALLOCATION_20261019T1525Z.json').write_text(json.dumps(a,indent=2)+'\n')
raw=(old/'REMOTE_SUBMISSION.py').read_bytes();launch=json.loads((old/'LAUNCH.json').read_bytes());payload={'reuse_zlib':base64.b64encode(zlib.compress(raw)).decode(),'reuse_sha256':hashlib.sha256(raw).hexdigest(),'launch':launch,'allocation':a,'allocation_sha256':hashlib.sha256((C/'GPU_REALLOCATION_20261019T1525Z.json').read_bytes()).hexdigest()}
b=r'''import zlib,base64,json,hashlib,subprocess,datetime,os
from pathlib import Path
P=PAYLOAD
raw=zlib.decompress(base64.b64decode(P['reuse_zlib'],validate=True));assert hashlib.sha256(raw).hexdigest()==P['reuse_sha256']
G={'__name__':'reused_readonly_profile_gates'};exec(compile(raw,'reviewed_profile_gates','exec'),G)
G['GPU']=P['allocation']['selected_gpu_uuid'];G['MIN_GPU_FREE']=4294967296+1000000000
B=G['B'];S=G['SOURCE'];U='computer-vision-homework-fg19-teacher-profile-v3.service';O=B/'research_tmp/fg19_teacher_profile_v3';RUN=B/'data/artifacts/runs/fg19_t3_h384x2_profile_s3_v3'
assert not O.exists() and not RUN.exists() and not (B/'runtime/STOP_ADMISSION.json').exists();assert G['_unit'](U,['LoadState'])['LoadState']=='not-found'
expected={v['path']:v['sha256'] for v in G['PAYLOAD']['source_manifest']['files']};actual={str(x.relative_to(S)) for x in S.rglob('*') if x.is_file()};assert len(expected)==164 and actual==set(expected)
for n,h in expected.items():
 q=S/n;assert not q.is_symlink() and G['_sha'](q)==h
prop=G['_check_terminal_propensity']();data=G['_data_identity']();parent,pop,cgroup=G['_parent']();gpu,foreign=G['_gpu']();assert not pop
scratch=G['_run'](['findmnt','-T',str(B/'research_tmp'),'-n','-o','FSTYPE']).strip();assert scratch not in ['tmpfs','ramfs'];assert os.statvfs(B).f_bavail*os.statvfs(B).f_frsize>20000000000
old=P['launch']['command'];cmd=[]
for arg in old:
 if arg.startswith('--unit='):arg='--unit='+U
 elif arg.startswith('--property=StandardOutput='):arg='--property=StandardOutput=append:'+str(O/'unit.log')
 elif arg.startswith('--property=StandardError='):arg='--property=StandardError=append:'+str(O/'unit.log')
 elif arg.startswith('--setenv=CUDA_VISIBLE_DEVICES='):arg='--setenv=CUDA_VISIBLE_DEVICES='+G['GPU']
 elif arg.startswith('--setenv=WNN_GPU_MEMORY_BYTES='):arg='--setenv=WNN_GPU_MEMORY_BYTES=4294967296'
 elif arg=='fg19_t3_h384x2_profile_s3_v2':arg='fg19_t3_h384x2_profile_s3_v3'
 cmd.append(arg)
assert '--setenv=WNN_GPU_MEMORY_BYTES=4294967296' in cmd and '--strict-muon-fp32' in cmd and cmd[cmd.index('--max-batches')+1]=='6'
O.mkdir();admission={'utc':datetime.datetime.now(datetime.UTC).isoformat(),'unit':U,'source_commit':'af21d6d8409c46b36830f0bdddd50e67ee17025f','source_files':expected,'reused_guard_sha256':P['reuse_sha256'],'allocation':P['allocation'],'allocation_sha256':P['allocation_sha256'],'propensity':prop,'data':data,'parent':parent,'populated':pop,'gpu':gpu,'foreign_preserved':foreign,'events':(cgroup/'memory.events').read_text(),'events_local':(cgroup/'memory.events.local').read_text(),'command':cmd}
G['_write'](O/'ADMISSION.json',admission)
r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);launch={'utc':datetime.datetime.now(datetime.UTC).isoformat(),'unit':U,'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'post':G['_unit'](U,['InvocationID','ActiveState','SubState','MainPID','ControlGroup','ExecStart'])};G['_write'](O/'LAUNCH.json',launch);print(json.dumps({'admission':admission,'launch':launch}))
'''.replace('P=PAYLOAD','P='+repr(payload))
compile(b,'profilev3','exec');R.mkdir(exist_ok=False);(R/'REMOTE_SUBMISSION.py').write_text(b)
try:r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','beleriand','python3 -B -'],input=b.encode(),capture_output=True,timeout=90)
except subprocess.TimeoutExpired as e:
 (R/'STDOUT.raw.json').write_bytes(e.stdout or b'');(R/'STDERR.raw').write_bytes(e.stderr or b'');(R/'TRANSPORT.json').write_text('{"outcome":"unknown","never_retry":true}\n');raise
(R/'STDOUT.raw.json').write_bytes(r.stdout);(R/'STDERR.raw').write_bytes(r.stderr);(R/'TRANSPORT.json').write_text(json.dumps({'returncode':r.returncode})+'\n');assert r.returncode==0,(r.returncode,r.stderr)
x=json.loads(r.stdout)
for k,v in x.items():(R/(k.upper()+'.json')).write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({'returncode':x['launch']['returncode'],'post':x['launch']['post'],'selected_gpu':gpu}))
