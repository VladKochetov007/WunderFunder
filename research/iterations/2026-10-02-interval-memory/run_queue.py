"""Bounded sequential cache or training queue; only fresh owned scopes may be stopped."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess, time, uuid
from datetime import datetime, timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
HELPER=ROOT/'research/iterations/2026-10-01-context-replay/run.py'
spec=importlib.util.spec_from_file_location('resource_helpers',HELPER)
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
sha=H.sha

def utc():return datetime.now(timezone.utc)
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def path(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 return ROOT/p

def contract(ap,expected):
 assert sha(ap)==expected,'allocation changed'
 a=json.loads(ap.read_text());assert a['execution_authorized'] is True
 assert a['sources'][str(Path(__file__).relative_to(ROOT))]==sha(__file__)
 assert a['sources'][str(HELPER.relative_to(ROOT))]==sha(HELPER)
 for rel,h in a['sources'].items():assert sha(path(rel))==h,('source changed',rel)
 for binding in a['input_bindings']:
  bp=path(binding['path']);assert sha(bp)==binding['sha256']
  for r in json.loads(bp.read_text())['records']:
   p=Path(r['path']);s=p.stat()
   d=dict(resolved=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
   assert all(r[k]==v for k,v in d.items()),('input changed',str(p))
   # Large arrays were fully hashed in inventory/producer receipts; preserve their
   # device/inode/size/mtime binding between jobs. Rehash small provenance files.
   if r.get('sha256') and r['size'] <= 20 * 2**20:assert sha(p)==r['sha256'],str(p)
 return a

def worker(ap,ah,index):
 a=contract(ap,ah);j=a['jobs'][index];out=path(a['execution_directory']);unit=os.environ['EB_OWN_UNIT']
 fields,pids=H.scope(unit);assert os.getpid() in pids
 base=Path('/sys/fs/cgroup')/fields['ControlGroup'].lstrip('/')
 assert (base/'memory.max').read_text().strip()==str(a['resources']['host_memory_bytes'])
 q,period=map(int,(base/'cpu.max').read_text().split());assert q*100==a['resources']['cpu_quota']*period
 r=dict(status='RUNNING',name=j['name'],stage=j['stage'],allocation_sha256=ah,argv=j['argv'],started_utc=utc().isoformat());t=time.monotonic()
 write(out/(j['name']+'.'+j['stage']+'.worker-started.json'),r)
 try:
  for rel in j.get('require_absent',[]):assert not path(rel).exists() and not path(rel).is_symlink(),('collision',rel)
  rc=subprocess.run(j['argv'],cwd=ROOT).returncode;r['returncode']=rc;assert rc==0,('job failed',rc)
  contract(ap,ah)
  if j.get('post_argv'):
   e=os.environ.copy();e['CUDA_VISIBLE_DEVICES']='';e['WNN_THREADS']='1'
   for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','ARROW_NUM_THREADS','NUMEXPR_NUM_THREADS'):e[k]='1'
   rc=subprocess.run(j['post_argv'],cwd=ROOT,env=e).returncode;r['post_returncode']=rc;assert rc==0,('postcheck failed',rc)
  r['outputs']={rel:dict(bytes=path(rel).stat().st_size,sha256=sha(path(rel))) for rel in j['required_outputs']}
  contract(ap,ah);r['status']='COMPLETE'
 except BaseException as exc:r.update(status='FAILED_STOP_NO_RETRY',failure=repr(exc));raise
 finally:
  r.update(seconds=time.monotonic()-t,finished_utc=utc().isoformat());write(out/(j['name']+'.'+j['stage']+'.worker-result.json'),r)

def main_run(ap):
 ah=sha(ap);a=contract(ap,ah);r=a['resources'];deadline=datetime.fromisoformat(a['deadline_utc'])
 assert deadline.utcoffset() is not None and deadline>utc()
 assert 0<r['allocator_fraction']<=.375 and r['owned_gpu_max_MiB']<=7168
 assert 0<r['cpu_quota']<=600 and 0<r['host_memory_bytes']<=20*2**30
 out=path(a['execution_directory']).resolve();assert out.is_relative_to(HERE) and out!=HERE and not out.exists();out.mkdir()
 write(out/'allocation.json',a)
 env=os.environ.copy();env.update(a['environment']);env.update(PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(r['gpu_id']),WNN_GPU_FRACTION=str(r['allocator_fraction']))
 start=time.monotonic();records=[];status='FAILED_STOP_NO_RETRY';failure=None;began=utc().isoformat()
 def emit(x):
  x['utc']=utc().isoformat()
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush()
 try:
  for j in a['jobs']:
   for rel in j.get('require_absent',[]):assert not path(rel).exists() and not path(rel).is_symlink(),('collision',rel)
  for index,j in enumerate(a['jobs']):
   contract(ap,ah);g=H.gpu(r['gpu_id']);assert g['uuid']==r['gpu_uuid'] and g['free_MiB']>=r['initial_free_MiB'],g
   assert g['total_MiB']*r['allocator_fraction']<=6144
   seconds=min(j['timeout_seconds'],(deadline-utc()).total_seconds())-10;assert seconds>1
   unit='eb-'+uuid.uuid4().hex;_,prior=H.scope(unit);assert not prior
   argv=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p',f'CPUQuota={r["cpu_quota"]}%','-p',f'MemoryMax={r["host_memory_bytes"]}','-p','MemorySwapMax=0','--','taskset','-c',','.join(map(str,r['cpu_ids'])),'nice','-n','10','/usr/bin/timeout','--signal=TERM','--kill-after=5s',f'{seconds:.3f}s','/usr/bin/time','-v','.venv/bin/python',str(Path(__file__).relative_to(ROOT)),'--allocation',str(ap.resolve()),'--allocation-sha256',ah,'--worker',str(index)]
   e=env.copy();e['EB_OWN_UNIT']=unit;rec=dict(name=j['name'],stage=j['stage'],unit=unit,argv=argv,gpu_before=g,peak_owned_GPU_MiB=0);p=None;err=None;t=time.monotonic()
   try:
    with (out/(j['name']+'.'+j['stage']+'.log')).open('x') as log:
     p=subprocess.Popen(argv,cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT);rec['pid']=p.pid;emit(dict(event='started',**rec))
     while p.poll() is None:
      fields,pids=H.scope(unit);g=H.gpu(r['gpu_id']);owned=0
      rows=H.command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],check=True).stdout.splitlines()
      for row in rows:
       if not row.strip():continue
       uid,pid,memory=[s.strip() for s in row.split(',')]
       if uid==r['gpu_uuid'] and int(pid) in pids:owned+=int(memory)
      rec['peak_owned_GPU_MiB']=max(rec['peak_owned_GPU_MiB'],owned)
      emit(dict(event='resource_sample',unit=unit,owned_GPU_MiB=owned,free_MiB=g['free_MiB']))
      assert g['uuid']==r['gpu_uuid'] and owned<=r['owned_gpu_max_MiB'] and g['free_MiB']>=r['running_free_MiB'],(owned,g)
      assert utc()<deadline and time.monotonic()-t<j['timeout_seconds'],'deadline'
      time.sleep(2)
     assert p.returncode==0,('step failed',p.returncode)
    wp=out/(j['name']+'.'+j['stage']+'.worker-result.json');w=json.loads(wp.read_text());assert w['status']=='COMPLETE' and w['allocation_sha256']==ah
    rec['worker_result']=dict(path=str(wp.relative_to(ROOT)),sha256=sha(wp));contract(ap,ah)
   except BaseException as exc:err=repr(exc);raise
   finally:
    try:
     fields,pids=H.scope(unit)
     if pids or (p is not None and p.poll() is None):
      try:H.command(['systemctl','--user','stop',unit+'.scope'])
      except subprocess.TimeoutExpired:pass
      fields,pids=H.scope(unit)
      if pids:H.command(['systemctl','--user','kill','--signal=KILL',unit+'.scope'])
      if p is not None:p.wait(timeout=15)
     fields,pids=H.scope(unit);rec.update(scope_state=fields,remaining_scope_pids=sorted(pids))
    except Exception as exc:rec['cleanup_error']=repr(exc)
    rec.update(exit_code=p.poll() if p else None,failure=err,seconds=time.monotonic()-t);records.append(rec);emit(dict(event='finished',**rec))
   assert not rec.get('cleanup_error') and not rec['remaining_scope_pids']
  status=a['success_status']
 except BaseException as exc:failure=repr(exc);raise
 finally:
  result=dict(status=status,failure=failure,allocation_sha256=ah,runner_sha256=sha(__file__),started_utc=began,finished_utc=utc().isoformat(),seconds=time.monotonic()-start,jobs=records,fits=sum(j['stage'] in ('parent','H2') for j in records),OOF_jobs=sum(j['stage']=='OOF' for j in records),metric_calls=0,retries=0)
  write(out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);ap.add_argument('--allocation-sha256');ap.add_argument('--worker',type=int);args=ap.parse_args()
 if args.worker is not None:
  assert args.allocation_sha256;worker(args.allocation,args.allocation_sha256,args.worker)
 else:main_run(args.allocation)
