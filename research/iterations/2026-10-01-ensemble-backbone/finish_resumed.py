"""Wait for the unstarted suffix, verify paired trajectories, then grade once."""
from __future__ import annotations
import argparse,importlib.util,json,os,subprocess,time,uuid
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];B=str(HERE.relative_to(ROOT))
spec=importlib.util.spec_from_file_location('completed_controller',HERE/'continue_v2.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
sha=C.sha;write=C.write;H=C.Q.H
def verify(p,h):
 assert sha(p)==h;m=json.loads(p.read_text());assert m['execution_authorized'] is True
 for rel,digest in m['sources'].items():assert sha(ROOT/rel)==digest,rel
 for segment in ('original','suffix'):
  rec=m[segment]['allocation'];assert sha(ROOT/rec['path'])==rec['sha256']
 r=m['original']['result'];assert sha(ROOT/r['path'])==r['sha256']
 return m
def worker(p,h):
 m=verify(p,h);out=HERE/'finish-02';unit=os.environ['EB_GRADE_UNIT'];fields,pids=H.scope(unit)
 assert os.getpid() in pids and os.environ['CUDA_VISIBLE_DEVICES']==''
 cg=Path('/sys/fs/cgroup')/fields['ControlGroup'].lstrip('/');assert (cg/'memory.max').read_text().strip()==str(10*2**30)
 q,period=map(int,(cg/'cpu.max').read_text().split());assert q==2*period
 assert json.loads((ROOT/m['suffix']['result_path']).read_text())['status']=='COMPLETED_REMAINING_4_FITS_3_OOF_NO_METRICS'
 C.paired_proof();verify(p,h)
 with (out/'builder.log').open('x') as f:
  rc=subprocess.run(['.venv/bin/python',B+'/build_resumed_bindings.py','--plan',str(p.resolve()),'--plan-sha256',h],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
 assert rc==0,('binding failed',rc)
 bp=HERE/'evaluation-bindings-01.json';verify(p,h)
 with (out/'evaluate.log').open('x') as f:
  rc=subprocess.run(['.venv/bin/python',B+'/evaluate.py','--bindings',str(bp),'--bindings-sha256',sha(bp),'--output-dir',str(HERE/'evaluation-01')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
 assert rc==0,('evaluation failed',rc)
 verify(p,h)
 assert json.loads((HERE/'evaluation-01/receipt.json').read_text())['status']=='COMPLETE_FIXED_ENSEMBLE_BACKBONE_COMPARISONS'
 write(out/'worker-result.json',dict(status='COMPLETE_ONE_FIXED_EVALUATION',plan_sha256=h,evaluation_receipt_sha256=sha(HERE/'evaluation-01/receipt.json')))
def execute(p,h):
 m=verify(p,h);out=HERE/'finish-02';out.mkdir();t=time.monotonic();unit=None;proc=None
 a=json.loads((ROOT/m['suffix']['allocation']['path']).read_text());deadline=datetime.fromisoformat(a['deadline_utc'])
 r=dict(status='WAITING_FOR_REMAINING_JOBS',plan_sha256=h,started_utc=datetime.now(timezone.utc).isoformat(),metric_retries=0,scientific_fit_retries=0)
 write(out/'started.json',r)
 try:
  rp=ROOT/m['suffix']['result_path']
  while not rp.exists():
   assert datetime.now(timezone.utc)<deadline;time.sleep(5)
  time.sleep(1);done=json.loads(rp.read_text());assert done['status']=='COMPLETED_REMAINING_4_FITS_3_OOF_NO_METRICS',done['status']
  verify(p,h);assert not (HERE/'evaluation-01').exists() and not (HERE/'evaluation-bindings-01.json').exists()
  remaining=int((deadline-datetime.now(timezone.utc)).total_seconds());assert remaining>30;seconds=min(900,remaining-10)
  unit='eb-grade-'+uuid.uuid4().hex;_,pids=H.scope(unit);assert not pids
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'src'),WNN_THREADS='2',EB_GRADE_UNIT=unit)
  for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS'):env[k]='2'
  argv=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=200%','-p',f'MemoryMax={10*2**30}','-p','MemorySwapMax=0','--','taskset','-c','22,23','nice','-n','10','timeout','--signal=TERM','--kill-after=5s',f'{seconds}s','.venv/bin/python',B+'/finish_resumed.py','--plan',str(p.resolve()),'--plan-sha256',h,'--worker']
  r.update(unit=unit,argv=argv)
  with (out/'worker.log').open('x') as f:
   proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);rc=proc.wait(timeout=seconds+20);r['exit_code']=rc;assert rc==0,rc
  verify(p,h);r.update(status='COMPLETE_ONE_PAIRED_STUDY',evaluation_receipt_sha256=sha(HERE/'evaluation-01/receipt.json'))
 except BaseException as exc:r.update(status='FAILED_STOP_NO_RETRY',failure=repr(exc));raise
 finally:
  try:
   if unit:
    fields,pids=H.scope(unit)
    if pids or (proc is not None and proc.poll() is None):
     try:H.command(['systemctl','--user','stop',unit+'.scope'])
     except subprocess.TimeoutExpired:pass
     _,pids=H.scope(unit)
     if pids:H.command(['systemctl','--user','kill','--signal=KILL',unit+'.scope'])
    if proc is not None:
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:
      proc.terminate()
      try:proc.wait(timeout=5)
      except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
    fields,pids=H.scope(unit);r.update(scope_state=fields,remaining_scope_pids=sorted(pids));assert not pids
  except BaseException as exc:r.update(status='FAILED_CLEANUP',cleanup_error=repr(exc));raise
  finally:
   r.update(seconds=time.monotonic()-t,finished_utc=datetime.now(timezone.utc).isoformat());write(out/'result.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--worker',action='store_true');a=ap.parse_args()
 (worker if a.worker else execute)(a.plan,a.plan_sha256)
