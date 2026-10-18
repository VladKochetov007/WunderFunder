"""Hash two exact fin256 parent inputs and launch one reviewed bounded metadata extraction."""
import argparse,base64,datetime,hashlib,json,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--review-sha',required=True);p.add_argument('--manifest-sha',required=True);a=p.parse_args()
R=pathlib.Path('/Users/nb/Documents/wunder/WunderFunder-maintenance');Q=R/'research/campaign_2026-10-18/quality_gap_r1';D=Q/'tl3_fin256_metadata_r1';E=Q/'tl3_fin256_metadata_execution_r1'
def sha(b):return hashlib.sha256(b).hexdigest()
mraw=(D/'PREPARATION_MANIFEST.json').read_bytes();assert sha(mraw)==a.manifest_sha
m=json.loads(mraw)
for name,spec in m['artifacts'].items():assert sha((D/name).read_bytes())==spec['sha256']
reviewraw=(D/'INDEPENDENT_REVIEW.json').read_bytes();assert sha(reviewraw)==a.review_sha
review=json.loads(reviewraw)
assert review["schema"]=="quality_gap.tl3_fin256_metadata_independent_review.v1"
assert review["decision"]=="PASS_STATIC_REVIEW_READY_FOR_ROOT_ADMISSION_WITH_FRESH_HASH_BINDING"
assert review["preparation_manifest_sha256"]==sha(mraw) and review["blockers"]==[]
template=json.loads((D/'INPUT_MANIFEST_TEMPLATE.json').read_bytes());helper=(D/'extract_fin256_metadata.py').read_bytes()
script='template='+repr(template)+'\nhelper64='+repr(base64.b64encode(helper).decode())+'\nreview_sha='+repr(a.review_sha)+'\nmanifest_sha='+repr(a.manifest_sha)+'\n'+r'''
import pathlib,subprocess,json,hashlib,datetime,os,base64,stat,time
B=pathlib.Path('/home/grinchenkoeg/computer-vision-homework-20261018');unit='computer-vision-homework-tl3-fin256-metadata-v1.service';E=B/'quality-gap-r1/tl3-fin256-metadata-v1';records=B/'receipts';start=time.monotonic()
names=['TL3_FIN256_METADATA_ADMISSION_V1.json','TL3_FIN256_METADATA_INPUTS_R1.json','TL3_FIN256_METADATA_R1.json','TL3_FIN256_METADATA_LAUNCH_V1.json']
for n in names:assert not os.path.lexists(records/n),n
assert not os.path.lexists(E)
for ancestor in E.parents:
 if ancestor==B:break
 assert not ancestor.is_symlink()
p=subprocess.run(['systemctl','--user','show',unit,'-p','LoadState','--value'],capture_output=True,text=True,timeout=5);assert p.stdout.strip()=='not-found'
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);assert p.returncode==0,(args,p.stderr);return p.stdout
props=dict(line.split('=',1) for line in cmd(['systemctl','--user','show','wnn-campaign.slice','-p','ControlGroup','-p','MemoryCurrent','-p','MemoryHigh','-p','MemoryMax','-p','MemorySwapMax','-p','CPUQuotaPerSecUSec']).splitlines())
assert props['CPUQuotaPerSecUSec']=='2s' and props['MemoryMax']=='20000000000' and props['MemoryHigh']=='18000000000' and props['MemorySwapMax']=='0'
cg=pathlib.Path('/sys/fs/cgroup')/props['ControlGroup'].lstrip('/');active=[];child_usage=0;reserved=0;quota=0
expected={'computer-vision-homework-r01-f3-sequence.service':(0.1,512000000),'wnn-bandit-computer-vision-homework-r01-wide-f3s1001-ep01-v1.service':(1.0,8000000000)}
for child in cg.iterdir():
 if not child.is_dir():continue
 if (child/'cgroup.procs').exists() and (child/'cgroup.procs').read_text().strip():
  assert child.name in expected,child.name
  cap,period=(child/'cpu.max').read_text().split();assert cap!='max'
  cores=int(cap)/int(period);memory=int((child/'memory.max').read_text());usage=int((child/'memory.current').read_text())
  assert (cores,memory)==expected[child.name]
  assert (child/'memory.swap.max').read_text().strip()=='0'
  active.append({'unit':child.name,'cpu_cores':cores,'memory_max':memory,'memory_current':usage})
  child_usage+=usage;reserved+=memory;quota+=cores
assert {x['unit'] for x in active}==set(expected)
assert quota+0.5<=2
committed=max(0,int(props['MemoryCurrent'])-child_usage)+reserved+1500000000
assert committed<17000000000,committed
scratch=cmd(['findmnt','-T',str(B/'research_tmp'),'-no','FSTYPE,SOURCE,TARGET']);assert scratch.split()[0] not in ['tmpfs','ramfs']
assert os.statvfs(B).f_bavail*os.statvfs(B).f_frsize>2000000000
bound=[];aliases=[]
for item in template['inputs']:
 p=pathlib.Path(item['path']);assert p.resolve()==p
 logical=pathlib.Path('/home/grinchenkoeg/WunderFunder-creative/artifacts/runs')/p.parent.name/p.name
 assert logical.parent.is_symlink() and logical.resolve()==p
 aliases.append({'logical_path':str(logical),'resolved_path':str(logical.resolve()),'parent_link_target':os.readlink(logical.parent)})
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_size==item['expected_bytes'] and s.st_size<=32*1024*1024
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 e=p.lstat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(e.st_dev,e.st_ino,e.st_size,e.st_mtime_ns)
 item['expected_sha256']=h.hexdigest();bound.append({'id':item['id'],'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest(),'mtime_ns':s.st_mtime_ns})
assert time.monotonic()-start<60

admission={'schema':'quality_gap.tl3_fin256_metadata_admission.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'ADMITTED_METADATA_ONLY','manifest_sha256':manifest_sha,'review_sha256':review_sha,'helper_sha256':hashlib.sha256(base64.b64decode(helper64)).hexdigest(),'bound_inputs':bound,'logical_alias_bindings':aliases,'fresh_parent_properties':props,'active_children':active,'committed_with_support_bytes':committed,'total_reserved_cpu_with_support':quota+0.5,'memory_events_before':(cg/'memory.events').read_text(),'physical_scratch':scratch,'limits':template['resource_contract'],'no_training_or_inference':True,'restricted_loader_only':True}
def write(n,obj):
 data=json.dumps(obj,indent=2,sort_keys=True).encode()+b'\n'
 with (records/n).open('xb') as f:f.write(data)
 return data
ad=write(names[0],admission)
template['status']='ROOT_HASH_BOUND';template['root_admission_receipt_sha256']=hashlib.sha256(ad).hexdigest();inp=write(names[1],template)
E.mkdir()
with (E/'extract_fin256_metadata.py').open('xb') as f:f.write(base64.b64decode(helper64))
with (E/names[1]).open('xb') as f:f.write(inp)
assert sorted(p.name for p in E.iterdir())==sorted(['extract_fin256_metadata.py',names[1]])
command=['systemd-run','--user','--unit='+unit,'--property=Type=exec','--property=RemainAfterExit=yes','--property=Slice=wnn-campaign.slice','--property=CPUQuota=50%','--property=MemoryMax=1500000000','--property=MemorySwapMax=0','--property=RuntimeMaxSec=120','--property=TimeoutStopSec=15','--working-directory='+str(E)]
for k,v in {'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TMPDIR':str(B/'research_tmp')}.items():command+=['--setenv='+k+'='+v]
command+=['/home/grinchenkoeg/WunderFunder/.venv/bin/python',str(E/'extract_fin256_metadata.py'),'--manifest',str(E/names[1]),'--output',str(records/names[2])]
p=subprocess.run(command,capture_output=True,text=True,timeout=20)
write(names[3],{'schema':'quality_gap.tl3_fin256_metadata_launch.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':unit,'command':command,'exit_status':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'admission_sha256':hashlib.sha256(ad).hexdigest(),'input_manifest_sha256':hashlib.sha256(inp).hexdigest()})
print(json.dumps({n:base64.b64encode((records/n).read_bytes()).decode() for n in [names[0],names[1],names[3]]}));assert p.returncode==0
'''
E.mkdir(exist_ok=False);(E/'root_launch_script.py').write_bytes(pathlib.Path(__file__).read_bytes())
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','beleriand','python3 -B -'],input=script.encode(),capture_output=True,timeout=100)
(E/'transport_stdout.json').write_bytes(p.stdout);(E/'transport_stderr.txt').write_bytes(p.stderr);assert p.returncode==0,(p.returncode,p.stderr.decode())
for name,value in json.loads(p.stdout).items():
 with (E/name).open('xb') as f:f.write(base64.b64decode(value))
print(json.dumps({'status':'launched','records':str(E)}))
