import base64, datetime, hashlib, json, os, pathlib, shutil, subprocess

B = pathlib.Path('/home/grinchenkoeg/computer-vision-homework-20261018')
P = B / 'research/campaign_2026-10-18/priority25_execution_r4'
UNIT = 'computer-vision-homework-r01-f3-result-collection-r1.service'
ADMISSION = B / 'receipts/F3_RESULT_COLLECTION_ROOT_ADMISSION_V1.json'
LAUNCH = B / 'receipts/F3_RESULT_COLLECTION_ROOT_LAUNCH_V1.json'
RECEIPT = B / 'receipts/RANK01_F3_S1001_RESULT_COLLECTION_R1.json'
SNAPSHOT = P / 'f3_result_collection_r1/records/F3_S1001_RESULT_R1'
LOG = B / 'receipts/F3_RESULT_COLLECTION_R1.log'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def props(unit):
    names = ['LoadState','ActiveState','SubState','ControlGroup','CPUQuotaPerSecUSec','MemoryHigh','MemoryMax','MemorySwapMax','MemoryCurrent']
    p = subprocess.run(['systemctl','--user','show',unit,*[x for n in names for x in ['-p',n]]],capture_output=True,text=True,timeout=10)
    assert p.returncode == 0
    return dict(x.split('=',1) for x in p.stdout.splitlines() if '=' in x)

def write(path, value):
    with path.open('x') as f:
        json.dump(value,f,indent=2); f.write('\n')

sequence_path = B / 'runtime/receipts/RANK01_F3_S1001_SEQUENCE.json'
sequence = json.loads(sequence_path.read_bytes())
assert sequence['status'] == 'finished' and sequence['index'] == 6
assert sequence['attempt'] == 'computer-vision-homework-r01-f3s1001-grade-v1'
assert sequence['plan_sha256'] == CFG['plan_sha256']
for name, digest in CFG['input_hashes'].items():
    path=pathlib.Path(name); assert path.is_relative_to(B) and sha(path)==digest, name
for name, digest in CFG['support_hashes'].items():
    assert sha(P/'f3_result_collection_r1'/name)==digest, name
assert sha(P/'f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json')==CFG['plan_sha256']
assert sha(B/'splits/fold3.json')==CFG['fold_sha256']
assert props(UNIT)['LoadState']=='not-found'
for path in [ADMISSION,LAUNCH,RECEIPT,SNAPSHOT,LOG,*[B/'receipts'/n for n in CFG['files']]]:
    assert not path.exists() and not path.is_symlink(), str(path)
parent=props('wnn-campaign.slice')
assert parent['CPUQuotaPerSecUSec']=='2s' and parent['MemoryHigh']=='18000000000' and parent['MemoryMax']=='20000000000' and parent['MemorySwapMax']=='0'
cg=pathlib.Path('/sys/fs/cgroup')/parent['ControlGroup'].lstrip('/')
assert 'wnn-campaign.slice' in str(cg)
assert all(not x.read_text().strip() for x in cg.rglob('cgroup.procs')), 'populated campaign descendant'
events=(cg/'memory.events').read_text(); local=(cg/'memory.events.local').read_text()
assert dict(x.split() for x in events.splitlines())['oom']=='0'
assert dict(x.split() for x in events.splitlines())['oom_kill']=='0'
assert int((cg/'memory.current').read_text())+512000000 < 17000000000
scratch=B/'research_tmp'; fs=subprocess.run(['findmnt','-n','-o','FSTYPE','-T',str(scratch)],capture_output=True,text=True,timeout=5)
assert fs.returncode==0 and fs.stdout.strip() not in ['tmpfs','ramfs'] and shutil.disk_usage(scratch).free>2000000000
mi=dict(x.replace(':','').split()[:2] for x in pathlib.Path('/proc/meminfo').read_text().splitlines())
assert int(mi['MemAvailable'])*1024>5000000000
for name, item in CFG['files'].items():
    assert pathlib.Path(name).name==name
    raw=base64.b64decode(item['base64'],validate=True)
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    with (B/'receipts'/name).open('xb') as f:f.write(raw)
    assert sha(B/'receipts'/name)==item['sha256']
assert json.loads(sequence_path.read_bytes())==sequence
assert all(not x.read_text().strip() for x in cg.rglob('cgroup.procs'))
cmd=['systemd-run','--user','--unit='+UNIT,'--slice=wnn-campaign.slice','--property=Type=exec','--property=CPUQuota=75%','--property=MemoryHigh=448000000','--property=MemoryMax=512000000','--property=MemorySwapMax=0','--property=RuntimeMaxSec=120','--property=RemainAfterExit=yes','--property=StandardOutput=append:'+str(LOG),'--property=StandardError=append:'+str(LOG),'--working-directory='+str(B),'/usr/bin/env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','TMPDIR='+str(scratch),'PYTHONPATH='+str(B/'source-a266b036/src'),'/home/grinchenkoeg/WunderFunder/.venv/bin/python','-B',str(P/'f3_result_collection_r1/collect_f3_results.py'),'--project-root',str(B),'--plan',str(P/'f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json'),'--terminal-seal',str(B/'receipts/F3_TERMINAL_EVIDENCE_SEAL_V1.json'),'--snapshot-dir',str(SNAPSHOT),'--receipt',str(RECEIPT)]
record={'schema':'priority25.f3_result_collection_root_admission.v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':UNIT,'sequence':sequence,'parent_properties':parent,'parent_events':events,'parent_events_local':local,'parent_current_bytes':int((cg/'memory.current').read_text()),'input_hashes':CFG['input_hashes'],'support_hashes':CFG['support_hashes'],'staged_files':{n:{k:v for k,v in x.items() if k!='base64'} for n,x in CFG['files'].items()},'snapshot':str(SNAPSHOT),'receipt':str(RECEIPT),'command':cmd,'scope':'Existing reviewed collector only; no training, export, grade or model deserialization.'}
write(ADMISSION,record)
p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
record.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
write(LAUNCH,record)
print(json.dumps(record,indent=2))
