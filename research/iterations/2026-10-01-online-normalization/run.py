"""One allocated attempt, six sequential scientific jobs; CPU validators inside fit scopes.

Stdlib only. Default --dry-run reads source/commands, never data/models/GPU.
GPU limits attribute NVML memory only to PIDs in the fresh owned UUID scope.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, time, uuid
from datetime import datetime, timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
COMMANDS=HERE/'commands.json';VALIDATOR=HERE/'check_checkpoint.py'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def utc():return datetime.now(timezone.utc)
def write(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def command(argv,**kw):return subprocess.run(argv,text=True,capture_output=True,timeout=10,**kw)
def relative(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 return ROOT/p

def steps_for(plan):
 steps=[]
 for j in plan['jobs']:
  f=j['fold'];name=j['name'];kind=j['stage'];args=j['resolved_args']
  assert f in (0,1) and kind in ('parent','H2')
  assert name==(f'cont_onsh256_s3_f{f}' if kind=='parent' else f'cont_onsh256_s3_h2_f{f}')
  assert j['argv'][:2]==['.venv/bin/python','research/iterations/2026-10-01-overnight/training/train_aux.py']
  assert args['online_norm'] and args['hidden']==256 and args['layers']==1 and args['seed']==3
  assert args['valid_fold']==f and j['expected_n_in']==323
  assert args['epochs']==(4 if kind=='parent' else 2)
  if kind=='H2':assert args['init']==f'cont_onsh256_s3_f{f}@ep3' and args['freeze']=='inp,gru'
  steps.append({'job':j,'stage':kind,'argv':j['argv']})
  if kind=='H2':
   expected=['.venv/bin/python',str((HERE/'predict_fixed.py').relative_to(ROOT)),name,'--fold',str(f),'--epoch','1','--batch','16','--output-dir',str((HERE/'cache').relative_to(ROOT))]
   assert j['OOF_argv']==expected;steps.append({'job':j,'stage':'OOF','argv':expected})
 assert [(s['job']['fold'],s['stage']) for s in steps]==[(0,'parent'),(0,'H2'),(0,'OOF'),(1,'parent'),(1,'H2'),(1,'OOF')]
 return steps

def contract(allocation_path,allocation_sha):
 assert sha(allocation_path)==allocation_sha
 a=json.loads(allocation_path.read_text());assert a['execution_authorized'] is True
 assert a['commands_sha256']==sha(COMMANDS) and a['runner_sha256']==sha(__file__)
 p=json.loads(COMMANDS.read_text());steps=steps_for(p)
 hashes=dict(p['source_sha256'])
 for rel,digest in a['source_sha256'].items():
  assert rel not in hashes or hashes[rel]==digest
  hashes[rel]=digest
 assert hashes.get(str(VALIDATOR.relative_to(ROOT)))==sha(VALIDATOR)
 assert hashes.get(str(Path(__file__).relative_to(ROOT)))==a['runner_sha256']
 for rel,digest in hashes.items():assert sha(relative(rel))==digest,('source changed',rel)
 bp=relative(a['input_bindings_path']);assert sha(bp)==a['input_bindings_sha256']
 bindings=json.loads(bp.read_text());assert len(bindings['records'])>0
 snapshots=[]
 for r in bindings['records']:
  q=Path(r['path']);st=q.stat()
  actual={'path':str(q),'resolved':str(q.resolve()),'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev}
  assert all(actual[k]==r[k] for k in ('resolved','size','mtime_ns','inode','device')),('input identity changed',str(q),actual,r)
  if r.get('sha256'):assert sha(q)==r['sha256'],('small input hash changed',str(q))
  snapshots.append(actual)
 return a,p,steps,snapshots

def gpu(gid):
 s=command(['nvidia-smi',f'--id={gid}','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],check=True).stdout.strip()
 uid,total,used,free=[v.strip() for v in s.split(',')]
 return {'uuid':uid,'total_MiB':int(total),'used_MiB':int(used),'free_MiB':int(free)}

def scope(unit):
 assert re.fullmatch('onorm-[0-9a-f]{32}',unit)
 s=command(['systemctl','--user','show',unit+'.scope','-p','ControlGroup','-p','ActiveState','-p','SubState','-p','LoadState'])
 fields=dict(v.split('=',1) for v in s.stdout.splitlines() if '=' in v);pids=set();limits={}
 if fields.get('ControlGroup'):
  base=Path('/sys/fs/cgroup')/fields['ControlGroup'].lstrip('/')
  for p in base.rglob('cgroup.procs'):
   try:pids.update(map(int,p.read_text().split()))
   except FileNotFoundError:pass
  for name in ('cpu.max','memory.max','memory.swap.max'):
   try:limits[name]=(base/name).read_text().strip()
   except FileNotFoundError:pass
 return fields,pids,limits

def proc_start(pid):
 try:return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
 except FileNotFoundError:return None

def owned_gpu(uid,pids):
 rows=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],check=True).stdout.splitlines();owned=0;members=[]
 for row in rows:
  if not row.strip():continue
  gu,pid,mem=[s.strip() for s in row.split(',')]
  if gu==uid and int(pid) in pids:
   n=int(mem);owned+=n;members.append({'pid':int(pid),'MiB':n})
 return owned,members

def check_limits(limits,r):
 if not limits:return
 assert limits.get('memory.max')==str(r['host_memory_bytes']) and limits.get('memory.swap.max')=='0',limits
 q,period=map(int,limits['cpu.max'].split());assert q*100==r['CPUQuota']*period,limits

def worker(a_path,a_sha,index):
 a,p,steps,before=contract(a_path,a_sha);s=steps[index];j=s['job'];stage=s['stage'];r=a['resources']
 unit=os.environ['WNN_OWN_UNIT'];fields,pids,limits=scope(unit)
 assert os.getpid() in pids,('worker is outside its owned resource scope',unit);check_limits(limits,r)
 execution=relative(a['execution_directory']).resolve();assert execution.is_relative_to(HERE) and execution!=HERE
 stem=f'{index:02d}-{j["name"]}-{stage}';start=time.monotonic();record={'status':'RUNNING','name':j['name'],'stage':stage,'argv':s['argv'],'pid':os.getpid(),'start_ticks':proc_start(os.getpid()),'started_utc':utc().isoformat(),'input_stat_before':before,'allocation_sha256':a_sha,'commands_sha256':a['commands_sha256'],'scope':unit,'scope_limits':limits,'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','WNN_GPU_FRACTION','WNN_THREADS','WNN_LOADER_WORKERS','PYTHONPATH')}}
 write(execution/(stem+'.worker-started.json'),record)
 try:
  if stage=='H2':
   parent=f'cont_onsh256_s3_f{j["fold"]}';pr=json.loads((execution/(parent+'.check.json')).read_text())
   assert pr['status']=='PASS_FIXED_ENDPOINT_FINITE_AND_LINEAGE'
   assert sha(ROOT/f'artifacts/runs/{parent}/model_ep3.pt')==pr['checkpoint']['sha256']
  if stage=='OOF':
   hr=json.loads((execution/(j['name']+'.check.json')).read_text());assert hr['status']=='PASS_FIXED_ENDPOINT_FINITE_AND_LINEAGE'
   assert sha(relative(j['endpoint']))==hr['checkpoint']['sha256']
  result=subprocess.run(s['argv'],cwd=ROOT);record['fit_or_OOF_returncode']=result.returncode
  assert result.returncode==0,('scientific job failed',result.returncode)
  contract(a_path,a_sha)
  if stage in ('parent','H2'):
   target=execution/(j['name']+'.check.json')
   validator=['.venv/bin/python',str(VALIDATOR.relative_to(ROOT)),j['name'],'--commands',str(COMMANDS.relative_to(ROOT)),'--output',str(target.relative_to(ROOT))]
   ve=os.environ.copy();ve['CUDA_VISIBLE_DEVICES']='';ve['WNN_THREADS']='1'
   for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','ARROW_NUM_THREADS','NUMEXPR_NUM_THREADS'):ve[k]='1'
   record['validator_argv']=validator;rv=subprocess.run(validator,cwd=ROOT,env=ve);record['validator_returncode']=rv.returncode
   assert rv.returncode==0,('checkpoint validator failed',rv.returncode)
   v=json.loads(target.read_text());assert v['status']=='PASS_FIXED_ENDPOINT_FINITE_AND_LINEAGE' and v['commands_sha256']==a['commands_sha256'] and v['validator_sha256']==sha(VALIDATOR)
   assert v['checkpoint']['n_in']==323 and v['checkpoint']['epoch']==(3 if stage=='parent' else 1)
   record['validator_receipt']={'path':str(target.relative_to(ROOT)),'sha256':sha(target),'checkpoint':v['checkpoint']}
  else:
   output=HERE/'cache'/f'pred_{j["name"]}_valid.npy';rp=output.with_suffix('.inputs.json');v=json.loads(rp.read_text())
   assert v['run']==j['name'] and v['fold']==j['fold'] and v['epoch']==1
   assert v['checkpoint_sha256']==hr['checkpoint']['sha256']==sha(relative(j['endpoint']))
   assert v['script_sha256']==sha(HERE/'predict_fixed.py') and v['prediction_sha256']==sha(output)
   record['prediction_receipt']={'path':str(rp.relative_to(ROOT)),'sha256':sha(rp),'prediction_sha256':v['prediction_sha256']}
  _,_,_,after=contract(a_path,a_sha);assert before==after
  record['status']='COMPLETE';record['input_stat_after']=after
 except BaseException as exc:
  record['status']='FAILED_STOP_NO_RETRY';record['failure']=repr(exc);raise
 finally:
  record['seconds']=time.monotonic()-start;record['finished_utc']=utc().isoformat();write(execution/(stem+'.worker-result.json'),record)

def run(a_path):
 a_sha=sha(a_path);a,p,steps,_=contract(a_path,a_sha);r=a['resources']
 assert type(r['gpu_id']) is int and r['gpu_uuid'].startswith('GPU-')
 assert 0<r['allocator_fraction']<=.375 and 0<r['allocator_max_MiB']<=6144
 assert 0<r['owned_GPU_MiB_max']<=7168 and r['minimum_initial_free_MiB']>=8192 and r['minimum_running_free_MiB']>=1024
 assert 0<r['CPUQuota']<=600 and 0<r['host_memory_bytes']<=20*2**30 and r['swap_bytes']==0
 assert 0<r['stage_seconds']<=7800 and r['cpu_ids'] and len(set(r['cpu_ids']))==len(r['cpu_ids'])
 for k,maximum in [('parent',3600),('H2',1200),('OOF',1200)]:assert 0<r['step_seconds'][k]<=maximum
 deadline=datetime.fromisoformat(a['deadline_utc']);assert deadline.utcoffset() is not None and deadline>utc()
 execution=relative(a['execution_directory']).resolve();assert execution.is_relative_to(HERE) and execution!=HERE and not execution.exists()
 for j in p['jobs']:
  q=ROOT/'artifacts/runs'/j['name'];assert not q.exists() and not q.is_symlink(),('run collision',q)
  if j['stage']=='H2':
   q=HERE/'cache'/f'pred_{j["name"]}_valid.npy'
   for f in (q,q.with_suffix('.npy.partial'),q.with_suffix('.inputs.json')):assert not f.exists() and not f.is_symlink(),('prediction collision',f)
 execution.mkdir();write(execution/'allocation.json',a)
 env=os.environ.copy();env.update(p['environment']);env.update(PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(r['gpu_id']),WNN_GPU_FRACTION=str(r['allocator_fraction']),WNN_GPU_DUTY='1.0')
 stage_started_utc=utc().isoformat();started=time.monotonic();end=started+min(r['stage_seconds'],(deadline-utc()).total_seconds());records=[];status='FAILED_STOP_NO_RETRY';failure=None
 def emit(v):
  v['utc']=utc().isoformat()
  with (execution/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n');f.flush()
 try:
  for index,s in enumerate(steps):
   contract(a_path,a_sha);j=s['job'];stage=s['stage'];before=gpu(r['gpu_id'])
   assert before['uuid']==r['gpu_uuid'] and before['free_MiB']>=r['minimum_initial_free_MiB'],before
   assert r['allocator_fraction']*before['total_MiB']<=r['allocator_max_MiB']
   seconds=min(r['step_seconds'][stage],end-time.monotonic(),(deadline-utc()).total_seconds())-7;assert seconds>1,'stage deadline exhausted'
   unit='onorm-'+uuid.uuid4().hex;prior,priorpids,_=scope(unit);assert not priorpids and prior.get('LoadState') in (None,'','not-found')
   child=['.venv/bin/python',str(Path(__file__).relative_to(ROOT)),'--allocation',str(a_path.resolve()),'--allocation-sha256',a_sha,'--worker-step',str(index)]
   argv=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p',f'CPUQuota={r["CPUQuota"]}%','-p',f'MemoryMax={r["host_memory_bytes"]}','-p','MemorySwapMax=0','--','taskset','-c',','.join(map(str,r['cpu_ids'])),'nice','-n','10','/usr/bin/timeout','--signal=TERM','--kill-after=5s',f'{seconds:.6f}s','/usr/bin/time','-v',*child]
   e=env.copy();e['WNN_OWN_UNIT']=unit;log=execution/f'{index:02d}-{j["name"]}-{stage}.log';t0=time.monotonic();rec={'name':j['name'],'stage':stage,'unit':unit,'argv':argv,'scientific_argv':s['argv'],'gpu_before':before,'peak_owned_GPU_MiB':0,'minimum_free_GPU_MiB':before['free_MiB'],'samples':0,'scope_limits_observed':False};process=None;step_failure=None
   try:
    with log.open('x') as stream:
     process=subprocess.Popen(argv,cwd=ROOT,env=e,stdout=stream,stderr=subprocess.STDOUT);rec.update(launcher_PID=process.pid,launcher_start_ticks=proc_start(process.pid));emit({'event':'started',**rec})
     while process.poll() is None:
      fields,pids,limits=scope(unit);check_limits(limits,r);rec['scope_limits_observed']|=bool(limits)
      g=gpu(r['gpu_id']);assert g['uuid']==r['gpu_uuid']
      owned,members=owned_gpu(g['uuid'],pids);rec['peak_owned_GPU_MiB']=max(rec['peak_owned_GPU_MiB'],owned);rec['minimum_free_GPU_MiB']=min(rec['minimum_free_GPU_MiB'],g['free_MiB']);rec['samples']+=1
      emit({'event':'resource_sample','unit':unit,'owned_GPU_MiB':owned,'owned_compute':members,'scope_pids':sorted(pids),'gpu':g})
      assert owned<=r['owned_GPU_MiB_max'],('owned GPU budget',owned)
      assert g['free_MiB']>=r['minimum_running_free_MiB'],('shared device free memory guard',g)
      assert time.monotonic()<end and utc()<deadline and time.monotonic()-t0<r['step_seconds'][stage],'deadline'
      time.sleep(1)
     assert process.returncode==0,('step exit',process.returncode)
    wr=execution/f'{index:02d}-{j["name"]}-{stage}.worker-result.json';w=json.loads(wr.read_text());assert w['status']=='COMPLETE' and w['allocation_sha256']==a_sha
    rec['worker_result']={'path':str(wr.relative_to(ROOT)),'sha256':sha(wr)}
    contract(a_path,a_sha)
   except BaseException as exc:step_failure=repr(exc);raise
   finally:
    cleanup=[]
    try:
     _,cleanup_pids,_=scope(unit)
     if process is not None and (process.poll() is None or step_failure is not None or cleanup_pids):
      # All signals target only this new UUID scope, never a PID or other unit.
      try:rv=command(['systemctl','--user','stop',unit+'.scope']);cleanup.append({'stop_returncode':rv.returncode})
      except Exception as exc:cleanup.append({'stop_error':repr(exc)})
      try:process.wait(timeout=5)
      except subprocess.TimeoutExpired:
       rv=command(['systemctl','--user','kill','--signal=KILL',unit+'.scope']);cleanup.append({'kill_returncode':rv.returncode});process.wait(timeout=10)
     fields,pids,limits=scope(unit)
     if pids:
      rv=command(['systemctl','--user','kill','--signal=KILL',unit+'.scope']);cleanup.append({'remaining_owned_pids_kill_returncode':rv.returncode});time.sleep(.2);fields,pids,limits=scope(unit)
     rec.update(scope_state=fields,remaining_scope_pids=sorted(pids),cleanup=cleanup)
    except Exception as exc:rec['cleanup_error']=repr(exc)
    rec.update(exit_code=process.poll() if process else None,failure=step_failure,seconds=time.monotonic()-t0);records.append(rec);emit({'event':'finished',**rec})
   assert rec.get('cleanup_error') is None and not rec['remaining_scope_pids'] and rec['scope_state'].get('ActiveState','') in ('','inactive','failed'),rec
   assert rec['scope_limits_observed'],'resource scope never observed'
  assert len(records)==6 and time.monotonic()<=end and utc()<=deadline
  status='COMPLETED_4_FITS_2_OOF_NO_METRICS'
 except BaseException as exc:failure=repr(exc);raise
 finally:
  result={'status':status,'failure':failure,'started_utc':stage_started_utc,'finished_utc':utc().isoformat(),'seconds':time.monotonic()-started,'commands_sha256':a['commands_sha256'],'allocation_sha256':a_sha,'runner_sha256':sha(__file__),'jobs':records,'fits':sum(s['stage'] in ('parent','H2') for s in records),'OOF_jobs':sum(s['stage']=='OOF' for s in records),'metric_calls':0,'retries':0}
  write(execution/'result.json',result);print(json.dumps(result),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path);ap.add_argument('--execute',action='store_true');ap.add_argument('--allocation-sha256');ap.add_argument('--worker-step',type=int);a=ap.parse_args()
 if a.worker_step is not None:
  assert a.allocation and a.allocation_sha256 and not a.execute and 0<=a.worker_step<6
  worker(a.allocation,a.allocation_sha256,a.worker_step)
 elif a.execute:
  assert a.allocation;run(a.allocation)
 else:
  p=json.loads(COMMANDS.read_text());ss=steps_for(p);print(json.dumps({'dry_run':True,'commands_sha256':sha(COMMANDS),'runner_sha256':sha(__file__),'steps':[{'name':s['job']['name'],'stage':s['stage'],'argv':s['argv']} for s in ss],'actual_data_GPU_model_execution':False},indent=2))
if __name__=='__main__':main()
