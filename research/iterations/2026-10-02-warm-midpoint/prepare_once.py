"""Root-owned once-only CPU preparation; generated tests then fixed construction."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,os,subprocess,time,uuid
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sp=importlib.util.spec_from_file_location('q',HERE/'run_queue.py');Q=importlib.util.module_from_spec(sp);sp.loader.exec_module(Q)
def main():
 ap=HERE/'preparation-allocation.json';a=json.loads(ap.read_text());ah=Q.sha(ap)
 assert a['execution_authorized'] and a['CPU']==28 and a['wall_seconds']==60 and a['fits']==a['OOF']==0
 for p,h in a['sources'].items():assert Q.sha(ROOT/p)==h,p
 out=HERE/'prepare-01';out.mkdir(exist_ok=False);unit='warm-midpoint-prepare-'+uuid.uuid4().hex
 assert not Q.H.scope(unit)[1]
 script="import subprocess;from pathlib import Path;p=Path("+repr(str(HERE))+ ");steps=[('generated',['check_generated.py']),('construction',['construct.py']),('checkpoints',['check_checkpoint.py','--output',str(p/'prepare-01/checkpoints.json')])];\nfor name,args in steps:\n with (p/'prepare-01'/(name+'.log')).open('x') as f:\n  r=subprocess.run([str(p.parents[2]/'.venv/bin/python'),str(p/args[0]),*args[1:]],stdout=f,stderr=subprocess.STDOUT);assert r.returncode==0,(name,r.returncode)"
 argv=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=100%','-p','MemoryMax=4G','-p','MemorySwapMax=0','--','taskset','-c','28','timeout','--signal=TERM','--kill-after=5s','60s','.venv/bin/python','-c',script]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','ARROW_NUM_THREADS','WNN_THREADS'):env[k]='1'
 result=dict(status='RUNNING',allocation_sha256=ah,argv=argv,unit=unit,started_utc=datetime.now(timezone.utc).isoformat());started=time.monotonic()
 try:
  with (out/'outer.log').open('x') as log:rc=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=70).returncode
  result['exit_code']=rc;assert rc==0
  generated=json.loads((out/'generated.log').read_text());assert generated['status']=='PASS_GENERATED_FIXED_WARM_MIDPOINT'
  checked=json.loads((out/'checkpoints.json').read_text());assert checked['status']=='PASS_FIXED_WARM_MIDPOINT_CHECKPOINTS' and len(checked['checkpoints'])==4
  assert Q.sha(ap)==ah
  for p,h in a['sources'].items():assert Q.sha(ROOT/p)==h,p
  result.update(status='COMPLETE_FIXED_MIDPOINT_CPU_PREPARATION',checkpoints_sha256=Q.sha(out/'checkpoints.json'),generated_sha256=Q.sha(out/'generated.log'))
 except BaseException as exc:result.update(status='FAILED_NO_RETRY',failure=repr(exc));raise
 finally:
  fields,pids=Q.H.scope(unit)
  if pids:Q.H.command(['systemctl','--user','stop',unit+'.scope']);fields,pids=Q.H.scope(unit)
  result.update(scope_state=fields,remaining_scope_pids=sorted(pids),seconds=time.monotonic()-started)
  Q.write(out/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
