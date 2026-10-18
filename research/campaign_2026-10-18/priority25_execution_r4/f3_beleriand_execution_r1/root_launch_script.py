"""Root-owned exact F3 admission; requires sealed result support SHA arguments."""
import argparse,base64,hashlib,json,pathlib,subprocess,datetime
ap=argparse.ArgumentParser();ap.add_argument('--result-manifest-sha',required=True);ap.add_argument('--result-review-sha',required=True);a=ap.parse_args()
R=pathlib.Path('/Users/nb/Documents/wunder/WunderFunder-maintenance');rel=pathlib.Path('research/campaign_2026-10-18/priority25_execution_r4');P=R/rel
B='/home/grinchenkoeg/computer-vision-homework-20261018';py='/home/grinchenkoeg/WunderFunder/.venv/bin/python'
def raw(n):return (P/n).read_bytes()
def digest(b):return hashlib.sha256(b).hexdigest()
def check(n,h):
 b=raw(n);assert digest(b)==h,(n,digest(b));return json.loads(b)
plan=check('f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json','b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8')
check('f3_beleriand_preparation_r1/INDEPENDENT_REVIEW.json','d2951d9abdf54481dbb7debba3cd9c68d24c9ee9d85e1ceff31201b28c1369d1')
cp=check('f3_checkpoint_support_r1/PREPARATION_MANIFEST.json','7e4fce0d7232a487a769e12d611050bc608f9cc7fcea300bc2063525abfa38dc')
check('f3_checkpoint_support_r1/INDEPENDENT_REVIEW.json','de709897397d7cd775d2aba84fb71f8f7d91504802814f5bcfc26e48e203763b')
result=check('f3_result_collection_r1/PREPARATION_MANIFEST.json',a.result_manifest_sha)
resultreview=check('f3_result_collection_r1/INDEPENDENT_REVIEW.json',a.result_review_sha)
assert resultreview['schema']=='priority25.f3_result_collection_independent_review.v1'
assert resultreview['status']=='passed' and resultreview['decision']=='PASS_STATIC_REVIEW_READY_FOR_ROOT_GATED_COLLECTION'
assert resultreview['manifest_sha256']==a.result_manifest_sha and resultreview['findings']==[]
assert resultreview['test_receipt_sha256']==result['test_receipt']['sha256']
preflight=check('f3_preflight_r1/F3_BELERIAND_PREFLIGHT_V1.json','3d2ec9ba9cb760e1f60888e4608290c311a35398bff372e9e06f1436f476b347');assert preflight['status']=='passed'
assert json.loads(raw('f3_preflight_r1/STOP_V1.json'))['exit_status']==0
# Stage only explicit hash-bound packet files; no source-tree or frozen-plan edits.
files={}
for folder,m in [('f3_checkpoint_support_r1',cp),('f3_result_collection_r1',result)]:
 items=m.get('artifacts',m.get('files'));assert isinstance(items,dict)
 for n,spec in items.items():
  name=folder+'/'+n;b=raw(name);expected=spec['sha256'] if isinstance(spec,dict) else spec;assert digest(b)==expected
  files[str(rel/name)]={'sha256':digest(b),'base64':base64.b64encode(b).decode()}
 for n in ['PREPARATION_MANIFEST.json','INDEPENDENT_REVIEW.json']:
  name=folder+'/'+n;b=raw(name);files[str(rel/name)]={'sha256':digest(b),'base64':base64.b64encode(b).decode()}
for n in ['F3_BELERIAND_PLAN.json','F3_SPLIT_METADATA_V1.json','PREPARATION_MANIFEST.json','INDEPENDENT_REVIEW.json','README.md','INDEPENDENT_REVIEW_REQUEST.md']:
 name='f3_beleriand_preparation_r1/'+n;b=raw(name);files[str(rel/name)]={'sha256':digest(b),'base64':base64.b64encode(b).decode()}
assert sum(len(x['base64']) for x in files.values())<2000000
for n in files:
 q=pathlib.PurePosixPath(n);assert not q.is_absolute() and '..' not in q.parts and str(q)==n and not any(x in n for x in ('\\','\x00'))
paths=set()
for s in plan['steps']:
 for key in ['archive_directory','collection_receipt_path','verification_receipt_path','receipt_path','run_output_directory','grade_output','prediction_path','sidecar_path']:
  if key in s:paths.add(s[key])
 paths.update(s.get('must_not_exist',[]));paths.update(s.get('required_outputs',[]))
paths.update([B+'/runtime/receipts/RANK01_F3_S1001_SEQUENCE.json',B+'/result_snapshots/rr18_r01_f3_s1001_e6',B+'/receipts/F3_BELERIAND_ADMISSION_V1.json',B+'/receipts/F3_BELERIAND_LAUNCH_V1.json'])
evidence={n:digest(raw(n)) for n in ['f3_preflight_r1/F3_BELERIAND_PREFLIGHT_V1.json','f3_preflight_r1/ACCEPTANCE_V1.json','f3_preflight_r1/FRESH_METADATA_RESOURCE_V1.json','f3_preflight_r1/SOURCE_INVENTORY_V1.json','f3_preflight_r1/ARCHIVE_RECHECK_V1.json','f3_preflight_r1/PROVIDER_CELL_RECONCILIATION_V1.json','f3_preflight_r1/DATASPHERE_OBSERVER_V1.json','f3_preflight_r1/OFFICIAL_PROVIDER_STATUS_V1.json','f3_checkpoint_support_r1/PREPARATION_MANIFEST.json','f3_checkpoint_support_r1/INDEPENDENT_REVIEW.json','f3_result_collection_r1/PREPARATION_MANIFEST.json','f3_result_collection_r1/INDEPENDENT_REVIEW.json']}
# All remote freshness checks precede the first admission/staging mutation.
script='files='+repr(files)+'\npaths='+repr(sorted(paths))+'\nplan='+repr(plan)+'\nevidence='+repr(evidence)+'\n'+r'''
import pathlib,json,subprocess,hashlib,base64,datetime,os
B=pathlib.Path('/home/grinchenkoeg/computer-vision-homework-20261018');unit='computer-vision-homework-r01-f3-sequence.service'
def command(args,timeout=15):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout);assert p.returncode==0,(args,p.stderr);return p.stdout
for path in paths:assert not os.path.lexists(path),path
for target in [unit]+['wnn-bandit-'+s['attempt']+'.service' for s in plan['steps']]:
 p=subprocess.run(['systemctl','--user','show',target,'-p','LoadState','--value'],capture_output=True,text=True,timeout=5);assert p.stdout.strip()=='not-found',(target,p.stdout)
assert not (B/'runtime/STOP_ADMISSION.json').exists()
properties=dict(x.split('=',1) for x in command(['systemctl','--user','show','wnn-campaign.slice','-p','ControlGroup','-p','MemoryCurrent','-p','MemoryHigh','-p','MemoryMax','-p','MemorySwapMax','-p','CPUQuotaPerSecUSec']).splitlines())
assert properties['CPUQuotaPerSecUSec']=='2s' and properties['MemoryHigh']=='18000000000' and properties['MemoryMax']=='20000000000' and properties['MemorySwapMax']=='0'
assert int(properties['MemoryCurrent'])+8000000000+512000000+1500000000<17000000000
cg=pathlib.Path('/sys/fs/cgroup')/properties['ControlGroup'].lstrip('/')
proc_files=[cg/'cgroup.procs']+list(cg.rglob('cgroup.procs'));assert len(proc_files)<128
active_cgroup_pids={str(p):p.read_text().strip() for p in proc_files if p.exists() and p.read_text().strip()};assert not active_cgroup_pids,active_cgroup_pids
assert os.statvfs(B).f_bavail*os.statvfs(B).f_frsize>=15000000000
scratch=command(['findmnt','-T',str(B/'research_tmp'),'-no','FSTYPE,SOURCE,TARGET']);assert scratch.split()[0] not in ['tmpfs','ramfs']
gpu=command(['nvidia-smi','-i','GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c','--query-gpu=uuid,name,memory.free','--format=csv,noheader']);assert int(gpu.strip().split(',')[-1].split()[0])>=5000
gpu_processes=[]
for line in command(['nvidia-smi','-i','GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader']).splitlines():
 cols=[v.strip() for v in line.split(',')];assert len(cols)==3;pid=int(cols[0]);pd=pathlib.Path('/proc')/str(pid)
 try:
  uid=pd.stat().st_uid;pcg=(pd/'cgroup').read_text();ours=uid==os.getuid() and ('wnn-campaign.slice' in pcg or str(B) in (pd/'cmdline').read_bytes().decode(errors='replace'))
 except FileNotFoundError:
  gpu_processes.append({'pid':pid,'exited_during_observation':True});continue
 assert not ours,('active campaign GPU process',pid)
 gpu_processes.append({'pid':pid,'uid':uid,'gpu_uuid':cols[1],'memory':cols[2],'campaign_process':ours,'preserved':True})

assert hashlib.sha256((B/'splits/fold3.json').read_bytes()).hexdigest()=='21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f'
assert hashlib.sha256((B/'receipts/F3_BELERIAND_PREFLIGHT_V1.json').read_bytes()).hexdigest()=='3d2ec9ba9cb760e1f60888e4608290c311a35398bff372e9e06f1436f476b347'
for folder in ['f3_beleriand_preparation_r1','f3_checkpoint_support_r1','f3_result_collection_r1']:
 destination=B/'research/campaign_2026-10-18/priority25_execution_r4'/folder
 assert not os.path.lexists(destination),('staging directory must be new',str(destination))
for name in files:
 q=pathlib.PurePosixPath(name);assert not q.is_absolute() and '..' not in q.parts and str(q)==name
 dest=B/name;assert dest.resolve().is_relative_to(B.resolve()),name
 for ancestor in dest.parents:
  if ancestor==B:break
  assert not ancestor.is_symlink(),('symlink ancestor',str(ancestor))
 assert not os.path.lexists(dest),name
admission={'schema':'priority25.f3_beleriand_admission.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'ADMITTED','cell':{'fold':3,'seed':1001,'arms':['wide','deep'],'provider':'Beleriand'},'plan_sha256':'b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8','bound_evidence':evidence,'fresh_parent_properties':properties,'fresh_memory_events':(cg/'memory.events').read_text(),'scratch':scratch,'selected_gpu':gpu,'selected_gpu_processes':gpu_processes,'active_campaign_cgroup_pids':active_cgroup_pids,'collision_paths_checked_absent':paths,'foreign_jobs':'preserved; empty physical GPU not required','ownership_scope':'B-only F3 claim; no F3 was dispatched by us to another provider; fresh DataSphere recent100 nonterminal0 and current remote integration has only unadmittedF3draft. Cluster API404 is access failure; cluster-wide absence unverified. F2 issue16 remains separate.','automatic_sequence_and_acceptance':'Frozen plan documentary continuation_gate is implemented as receipt/acceptance condition. Native trainer rejects source/data/runtime/schedule contract drift; sequence advances exact successful stage1 commands automatically. Collect and verify stage1 counters/ancestry while peer runs. Independent checkpoint acceptance gates accepted results, not an added global stop.','limits':{'gpu_uuid':'GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c','max_our_gpu_workers':1,'worker_cpu':1,'worker_memory_bytes':8000000000,'gpu_allocator_bytes':4000000000,'controller_cpu':.1,'controller_memory_bytes':512000000,'controller_runtime_seconds':370800,'fit_seconds':86400,'export_seconds':7200,'grade_seconds':3600,'admission_seconds':300},'checkpoint_acceptance_requirements':['Exact actual working_directory is B/source-a266b036 and native guard inventory matches fresh source binding','Stage1 next_epoch2/29760updates/2976batches and terminal next_epoch6/89280updates/8928batches,FP32Adam,138 expected canonical modules','Both own and peer stage1 archives and receipts; exact resumed_from path/SHA/next_epoch2','Full seven-step result acceptance and arrival terminal seal using reviewed F3 helper; no F0/F1 counts','No quality claim until actual independent official grade and accepted outputs']}
with (B/'receipts/F3_BELERIAND_ADMISSION_V1.json').open('x') as f:json.dump(admission,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
for name,item in files.items():
 p=B/name;p.parent.mkdir(parents=True,exist_ok=True);data=base64.b64decode(item['base64']);assert hashlib.sha256(data).hexdigest()==item['sha256']
 with p.open('xb') as f:f.write(data)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
cmd=['systemd-run','--user','--unit='+unit,'--property=Type=exec','--property=Slice=wnn-campaign.slice','--property=CPUQuota=10%','--property=MemoryMax=512000000','--property=MemorySwapMax=0','--property=RuntimeMaxSec=370800','--property=TimeoutStopSec=45','--working-directory='+str(B/'source-a266b036')]
for k,v in {'WNN_SLICE':'wnn-campaign.slice','WNN_TMP_ROOT':str(B/'research_tmp'),'TMPDIR':str(B/'research_tmp'),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}.items():cmd+=['--setenv='+k+'='+v]
cmd+=['/home/grinchenkoeg/WunderFunder/.venv/bin/python',str(B/'source-a266b036/scripts/bandit_sequence.py'),'--plan',str(B/'research/campaign_2026-10-18/priority25_execution_r4/f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json'),'--receipt',str(B/'runtime/receipts/RANK01_F3_S1001_SEQUENCE.json')]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
launch={'schema':'priority25.f3_beleriand_launch.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':unit,'command':cmd,'exit_status':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'staged_files':{k:v['sha256'] for k,v in files.items()},'admission_sha256':hashlib.sha256((B/'receipts/F3_BELERIAND_ADMISSION_V1.json').read_bytes()).hexdigest()}
with (B/'receipts/F3_BELERIAND_LAUNCH_V1.json').open('x') as f:json.dump(launch,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({n:base64.b64encode((B/'receipts'/n).read_bytes()).decode() for n in ['F3_BELERIAND_ADMISSION_V1.json','F3_BELERIAND_LAUNCH_V1.json']}));assert p.returncode==0
'''
D=P/'f3_beleriand_execution_r1';D.mkdir(exist_ok=False)
(D/'root_launch_script.py').write_text(pathlib.Path(__file__).read_text())
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','beleriand','python3 -B -'],input=script.encode(),capture_output=True,timeout=100)
with (D/'transport_stdout.json').open('xb') as f:f.write(p.stdout)
with (D/'transport_stderr.txt').open('xb') as f:f.write(p.stderr)
assert p.returncode==0,(p.returncode,p.stderr.decode())
for n,data in json.loads(p.stdout).items():
 with (D/n).open('xb') as f:f.write(base64.b64decode(data))
print(json.dumps({'status':'launched','unit':'computer-vision-homework-r01-f3-sequence.service','receipt_dir':str(D)}))
