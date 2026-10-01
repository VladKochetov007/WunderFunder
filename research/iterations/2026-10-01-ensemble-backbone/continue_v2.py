"""Controller: complete the single authorized cache -> paired fits -> grading queue.

No retries, quality-based branching or checkpoint selection. Every step is bounded.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,subprocess,time,uuid
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];B=str(HERE.relative_to(ROOT))
spec=importlib.util.spec_from_file_location('queue_helpers',HERE/'run_queue.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
sha=Q.sha
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def rel(p):return str(Path(p).relative_to(ROOT))
def verify(mp,mh):
 assert sha(mp)==mh
 m=json.loads(mp.read_text())
 for r,h in m['sources'].items():assert sha(ROOT/r)==h,('changed source',r)
 for r,h in m['inputs'].items():assert sha(ROOT/r)==h,('changed input binding',r)
 assert m['execution_authorized'] is True
 return m
def stat_record(p,h):
 s=p.stat();return dict(path=str(p),resolved=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,prior_sha256=h)
def prepare(m):
 cache=json.loads((HERE/'cache-execution-01/result.json').read_text())
 assert cache['status']=='COMPLETED_6_ANCHOR_STREAMS_NO_METRICS' and cache['failure'] is None and len(cache['jobs'])==6
 assert cache['allocation_sha256']==m['inputs'][B+'/cache-allocation.json']
 for j in cache['jobs']:assert j['exit_code']==0 and j['failure'] is None and not j['remaining_scope_pids']
 inp=HERE/'input-bindings.json'
 subprocess.run(['.venv/bin/python',B+'/prepare_inputs.py','--assemble','--inputs',str(inp),'--inputs-sha256',sha(inp)],check=True,cwd=ROOT)
 records=[]
 for fold in (0,1):
  p=HERE/f'partner-bindings_f{fold}.json';b=json.loads(p.read_text());r=stat_record(p,sha(p));r['sha256']=r.pop('prior_sha256');records.append(r)
  for member in b['members']:
   for key,hkey in [('path','sha256'),('receipt_path','receipt_sha256')]:
    p=ROOT/member['train'][key];records.append(stat_record(p,member['train'][hkey]))
 pb=HERE/'partner-input-bindings.json';write(pb,dict(records=records))
 commands=json.loads((HERE/'commands.json').read_text());jobs=[]
 for j in commands['jobs']:
  name=j['name'];d=f'artifacts/runs/{name}';epochs=j['resolved_args']['epochs']
  outputs=[f'{d}/model_ep{i}.pt' for i in range(epochs)]+[d+'/model.pt',d+'/history.json',d+'/trajectory-proof.jsonl',d+'/partner-bindings.json',f'{B}/execution-01/{name}.check.json']
  jobs.append(dict(name=name,stage=j['stage'],argv=j['argv'],post_argv=j['check_argv'],timeout_seconds=j['wall_limit_seconds'],require_absent=[d,f'{B}/execution-01/{name}.check.json'],required_outputs=outputs))
  if j['stage']=='H2':
   p=f'{B}/cache/pred_{name}_valid'
   jobs.append(dict(name=name,stage='OOF',argv=j['OOF_argv'],timeout_seconds=j['OOF_wall_limit_seconds'],require_absent=[p+'.npy',p+'.npy.partial',p+'.inputs.json'],required_outputs=[p+'.npy',p+'.inputs.json']))
 ca=json.loads((HERE/'cache-allocation.json').read_text())
 a=dict(execution_authorized=True,sources=m['sources'],input_bindings=[dict(path=rel(inp),sha256=sha(inp)),dict(path=rel(pb),sha256=sha(pb))],jobs=jobs,execution_directory=B+'/execution-01',success_status='COMPLETED_8_FITS_4_OOF_NO_METRICS',deadline_utc=ca['deadline_utc'],resources=ca['resources'],environment=ca['environment'])
 write(HERE/'training-allocation.json',a)
 print(json.dumps(dict(status='PREPARED_TRAINING_ALLOCATION',sha256=sha(HERE/'training-allocation.json'),jobs=len(jobs))),flush=True)
def paired_proof():
 checks=[]
 for fold in (0,1):
  for stage in ('parent','h2'):
   paths=[ROOT/f'artifacts/runs/eb_{arm}_s3_{stage}_f{fold}/trajectory-proof.jsonl' for arm in ('control','candidate')]
   rows=[[json.loads(line) for line in p.read_text().splitlines()] for p in paths];c,t=rows
   assert len(c)==len(t)
   assert c[0]==t[0],('initialization or row order mismatch',fold,stage)
   assert all(x['torch_cpu_rng_sha256']==y['torch_cpu_rng_sha256'] for x,y in zip(c,t)),('CPU RNG mismatch',fold,stage)
   if stage=='parent':assert c[1]==t[1],('normalization mismatch',fold)
   checks.append(dict(fold=fold,stage=stage,paths={rel(p):sha(p) for p in paths},identical_initialization_and_cpu_rng=True,identical_parent_normalization=stage=='parent'))
 write(HERE/'paired-trajectory-check.json',dict(status='PASS_PAIRED_INITIALIZATION_AND_CPU_RNG',checks=checks,caveat='CUDA arithmetic is not guaranteed byte deterministic; own trained H2 parents intentionally differ.'))
def grade(m):
 a=HERE/'training-allocation.json';done=json.loads((HERE/'execution-01/result.json').read_text())
 assert done['status']=='COMPLETED_8_FITS_4_OOF_NO_METRICS' and done['allocation_sha256']==sha(a)
 paired_proof()
 bp=HERE/'evaluation-bindings-01.json'
 subprocess.run(['.venv/bin/python',B+'/build_bindings.py','--allocation',rel(a),'--allocation-sha256',sha(a),'--source-sha256',m['sources'][B+'/build_bindings.py']],check=True,cwd=ROOT)
 subprocess.run(['.venv/bin/python',B+'/evaluate.py','--bindings',rel(bp),'--bindings-sha256',sha(bp),'--output-dir',B+'/evaluation-01'],check=True,cwd=ROOT)
 assert json.loads((HERE/'evaluation-01/receipt.json').read_text())['status']=='COMPLETE_FIXED_ENSEMBLE_BACKBONE_COMPARISONS'
def cpu_step(phase,mp,mh,out,seconds):
 unit='eb-finish-'+uuid.uuid4().hex;p=None
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'src'),WNN_THREADS='2',EB_CPU_UNIT=unit)
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS'):env[k]='2'
 argv=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=200%','-p',f'MemoryMax={10*2**30}','-p','MemorySwapMax=0','--','taskset','-c','22,23','nice','-n','10','timeout','--signal=TERM','--kill-after=5s',str(seconds)+'s','.venv/bin/python',str(Path(__file__).relative_to(ROOT)),'--manifest',str(mp),'--manifest-sha256',mh,'--phase',phase]
 r=dict(phase=phase,unit=unit,argv=argv)
 try:
  with (out/(phase+'.log')).open('x') as log:
   p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   rc=p.wait(timeout=seconds+20);r['exit_code']=rc;assert rc==0,(phase,rc)
 finally:
  try:
   fields,pids=Q.H.scope(unit)
   if pids or (p is not None and p.poll() is None):
    try:Q.H.command(['systemctl','--user','stop',unit+'.scope'])
    except subprocess.TimeoutExpired:pass
    _,pids=Q.H.scope(unit)
    if pids:Q.H.command(['systemctl','--user','kill','--signal=KILL',unit+'.scope'])
   if p is not None:
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:
     p.terminate()
     try:p.wait(timeout=5)
     except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
   fields,pids=Q.H.scope(unit);r.update(scope_state=fields,remaining_pids=sorted(pids))
   assert not pids
  except BaseException as exc:r['cleanup_error']=repr(exc);raise
  finally:write(out/(phase+'.result.json'),r)
def run(mp,mh):
 m=verify(mp,mh);out=HERE/'controller-02';out.mkdir();start=time.monotonic();r=dict(status='RUNNING',manifest_sha256=mh,started_utc=datetime.now(timezone.utc).isoformat())
 deadline=datetime.fromisoformat(json.loads((HERE/'cache-allocation.json').read_text())['deadline_utc'])
 try:
  cp=HERE/'cache-execution-01/result.json'
  while not cp.exists():
   assert datetime.now(timezone.utc)<deadline;time.sleep(5)
  # Completion is written once; wait briefly for that file write to finish.
  time.sleep(1);verify(mp,mh)
  remaining=int((deadline-datetime.now(timezone.utc)).total_seconds());assert remaining>30
  cpu_step('prepare',mp,mh,out,min(600,remaining-10))
  verify(mp,mh)
  with (out/'training.log').open('x') as log:
   rc=subprocess.run(['.venv/bin/python',B+'/run_queue.py','--allocation',B+'/training-allocation.json'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
  assert rc==0,('training queue failed',rc)
  verify(mp,mh)
  remaining=int((deadline-datetime.now(timezone.utc)).total_seconds());assert remaining>30
  cpu_step('grade',mp,mh,out,min(900,remaining))
  verify(mp,mh);r.update(status='COMPLETE_ONE_PAIRED_STUDY',evaluation_receipt_sha256=sha(HERE/'evaluation-01/receipt.json'))
 except BaseException as exc:r.update(status='FAILED_STOP_NO_RETRY',failure=repr(exc));raise
 finally:
  r.update(seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat());write(out/'result.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--phase',choices=['prepare','grade']);a=ap.parse_args()
 if a.phase:
  m=verify(a.manifest,a.manifest_sha256);u=os.environ['EB_CPU_UNIT'];fields,pids=Q.H.scope(u);assert os.getpid() in pids and os.environ['CUDA_VISIBLE_DEVICES']==''
  c=Path('/sys/fs/cgroup')/fields['ControlGroup'].lstrip('/');assert (c/'memory.max').read_text().strip()==str(10*2**30)
  quota,period=map(int,(c/'cpu.max').read_text().split());assert quota==2*period
  (prepare if a.phase=='prepare' else grade)(m)
 else:run(a.manifest,a.manifest_sha256)
