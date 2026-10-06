"""One immutable runtime attempt; one pinned CPU; hard wall and address-space bounds."""
import time
LAUNCH_START=time.perf_counter()
import argparse,datetime,hashlib,json,os
from pathlib import Path
import resource,signal,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);ap.add_argument('--allocation-sha256',required=True);a=ap.parse_args()
 assert sha(a.allocation)==a.allocation_sha256
 allocation=json.loads(a.allocation.read_text());plan_path=HERE/'runtime-plan.json';plan=json.loads(plan_path.read_text())
 assert allocation['midpoint_runtime_audit_authorized'] is True and allocation['attempts']==1
 assert allocation['plan_sha256']==sha(plan_path)
 for rel,expected in plan['source_sha256'].items():assert sha(ROOT/rel)==expected,rel
 cpu=int(plan['resources']['CPU']);hard=int(plan['resources']['hard_wall_seconds']);assert 3600<=hard<=4200
 os.sched_setaffinity(0,{cpu});cap=16*(1<<30);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
 rel=Path('/proc/self/cgroup').read_text().strip().split(':',2)[2];cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
 limits={name:(cg/name).read_text().strip() for name in ['cpu.max','memory.max','memory.swap.max']}
 assert limits=={'cpu.max':'100000 100000','memory.max':str(cap),'memory.swap.max':'0'}
 env=os.environ.copy()
 assert all(k not in env for k in ['WNN_NATIVE_LEVEL','WNN_EXPECTED_ROWS','WNN_BUDGET_S']),'fixed ensemble requires absent policy overrides'
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[key]='1'
 env['CUDA_VISIBLE_DEVICES']='';env['PYTHONUNBUFFERED']='1';env['ARROW_NUM_THREADS']='1'
 env['MALLOC_CONF']='background_thread:false';env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None)
 output=HERE/'execution';output.mkdir(exist_ok=False)
 argv=[str(Path(plan['scorer_environment'])/'run.sh'),str(HERE/'audit_runtime.py'),'--plan',str(plan_path),
       '--allocation',str(a.allocation.resolve()),'--allocation-sha256',a.allocation_sha256]
 receipt={'status':'STARTED','utc':now(),'argv':argv,'affinity':[cpu],'RLIMIT_AS':cap,
          'plan_sha256':sha(plan_path),'allocation_sha256':a.allocation_sha256,
          'attempts':1,'retry_allowed':False,'cgroup':rel,'cgroup_limits':limits,'environment':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','ARROW_NUM_THREADS','MALLOC_CONF']}}
 (output/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
 start=time.perf_counter();p=None;timed_out=False
 with (output/'worker.log').open('x') as log:
  p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   tail=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()
   process_receipt={'pid':p.pid,'process_group':p.pid,'start_ticks':int(tail[19]),'utc':now(),'argv':argv}
  except FileNotFoundError:process_receipt={'pid':p.pid,'process_group':p.pid,'start_ticks':None,'utc':now(),'argv':argv,'early_exit':True}
  (output/'worker-process.json').write_text(json.dumps(process_receipt,indent=2)+'\n')
  # 2s termination grace plus 3s budget for final status/receipt serialization.
  try:code=p.wait(timeout=max(.01,hard-5-(time.perf_counter()-LAUNCH_START)))
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(p.pid,signal.SIGTERM)
   try:code=p.wait(timeout=2)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
 elapsed=time.perf_counter()-LAUNCH_START
 receipt.update(ended_utc=now(),exit_code=code,hard_timeout=timed_out,process_wall_seconds=elapsed,
                actual_full_process_minutes_39_4M=elapsed/60,
                actual_full_process_at_most_60min=elapsed<=3600,
                log_sha256=sha(output/'worker.log'),status='COMPLETE' if code==0 else 'FAILED_OR_INTERRUPTED')
 result_path=output/'worker-result.json'
 if result_path.exists():
  worker=json.loads(result_path.read_text());receipt['worker_result_sha256']=sha(result_path)
  receipt['functional_complete']=worker['status']=='COMPLETE' and worker['callback_count']==39400000 and worker['emitted_count']==39204970 and worker['completed_sequences']==1970
  receipt['quiet_runtime_qualified']=worker.get('quiet_runtime_qualified',False)
  receipt['process_deadline_observed']=receipt['functional_complete'] and elapsed<=hard
 else:receipt.update(functional_complete=False,quiet_runtime_qualified=False,process_deadline_observed=False)
 receipt['delivery_runtime_qualified']=(receipt['functional_complete'] and receipt['actual_full_process_at_most_60min'] and receipt['process_deadline_observed'] and not timed_out)
 (output/'launch-result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
 if code!=0:raise SystemExit(code if code>0 else 1)
if __name__=='__main__':main()
