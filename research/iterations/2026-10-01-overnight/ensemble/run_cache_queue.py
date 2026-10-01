"""ENSHEAD cache/manifest runner: immutable inputs, bounded scopes, append-only ledger."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
 return h.hexdigest()


def stamp():return datetime.now(timezone.utc).isoformat()


def append(record):
 with (HERE/'job-ledger.jsonl').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())


def stop_own(process,unit,log):
 try:subprocess.run(['systemctl','--user','stop',unit+'.scope'],stdout=log,stderr=subprocess.STDOUT,timeout=15,check=False)
 except subprocess.TimeoutExpired:pass
 if process.poll() is None:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def own_gpu_memory(gpu,unit):
 group=subprocess.run(['systemctl','--user','show',unit+'.scope','-p','ControlGroup','--value'],capture_output=True,text=True,timeout=5)
 cg=group.stdout.strip()
 if not cg:return None
 procs=Path('/sys/fs/cgroup')/cg.lstrip('/')/'cgroup.procs'
 if not procs.exists():return None
 pids={int(p) for p in procs.read_text().split()}
 listing=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
 amounts=[]
 for line in listing.splitlines():
  pid,memory=[q.strip() for q in line.split(',')]
  if pid.isdigit() and int(pid) in pids and memory.isdigit():amounts.append(int(memory))
 return sum(amounts) if amounts else 0


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--stage',choices=['cache','manifest'],required=True);ap.add_argument('--gpu',type=int,choices=[6,7]);ap.add_argument('--lane',type=int,default=0,choices=[0,1]);a=ap.parse_args()
 if a.stage=='cache' and a.gpu is None:raise ValueError('cache stage requires GPU')
 if a.stage=='manifest' and (a.gpu is not None or a.lane!=0):raise ValueError('manifest stage is one sequential CPU lane')
 planpath=a.plan.resolve()
 if planpath.parent!=HERE:raise ValueError('plan must be in ENS directory')
 frozen=sha(planpath);plan=json.loads(planpath.read_text());deadline=datetime.fromisoformat(plan['deadline_utc'])
 env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),WNN_GPU_FRACTION='.30',WNN_THREADS='4',WNN_LOADER_WORKERS='2',WNN_LOADER_RAM_GB='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMBA_NUM_THREADS='4',ARROW_NUM_THREADS='4')
 env['CUDA_VISIBLE_DEVICES']=str(a.gpu) if a.gpu is not None else ''
 logs=HERE/'logs';logs.mkdir(exist_ok=True)
 jobs=plan['cache_jobs'][a.lane::2] if a.stage=='cache' else plan['manifest_jobs']
 if a.stage=='manifest':
  for job in plan['cache_jobs']:
   if not (ROOT/job['receipt']).exists():raise ValueError('manifest stage missing partner cache receipt')
 for job in jobs:
  if datetime.now(timezone.utc)>=deadline:raise RuntimeError('cache/manifest deadline reached')
  if sha(planpath)!=frozen:raise ValueError('execution plan changed')
  for path,expected in plan['input_sha256'].items():
   if sha(ROOT/path)!=expected:raise ValueError(f'input changed: {path}')
  for path,expected in job.get('input_sha256',{}).items():
   if sha(ROOT/path)!=expected:raise ValueError(f'job input changed: {path}')
  for output in job['new_outputs']:
   if (ROOT/output).exists():raise FileExistsError(ROOT/output)
  free=None
  if a.gpu is not None:
   free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip())
   if free<6200:
    append({'event':'resource_blocked','utc':stamp(),'name':job['name'],'stage':a.stage,'gpu':a.gpu,'free_MiB':free});raise RuntimeError('insufficient GPU headroom')
  logpath=logs/f"{job['name']}.{a.stage}.log";unit=f"wunder-ens-{job['name']}-{a.stage}"
  command=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','MemoryMax=20G','-p','MemorySwapMax=0','-p','CPUQuota=400%','--','nice','-n','10','/usr/bin/time','-v',*job['argv']]
  started=time.monotonic();timed_out=False;gpu_limit=False;peak=0;polls=0
  with logpath.open('x') as log:
   proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   append({'event':'started','utc':stamp(),'name':job['name'],'stage':a.stage,'gpu':a.gpu,'pid':proc.pid,'plan_sha256':frozen,'argv':command,'free_MiB':free,'timeout_s':600})
   try:
    while proc.poll() is None:
     remaining=600-(time.monotonic()-started)
     if remaining<=0:timed_out=True;stop_own(proc,unit,log);break
     try:proc.wait(timeout=min(5,remaining));break
     except subprocess.TimeoutExpired:pass
     if a.gpu is not None:
      usage=own_gpu_memory(a.gpu,unit)
      if usage is not None:
       polls+=1;peak=max(peak,usage)
       if usage>5632:gpu_limit=True;stop_own(proc,unit,log);break
   except BaseException:
    stop_own(proc,unit,log);append({'event':'runner_exception','utc':stamp(),'name':job['name'],'stage':a.stage});raise
   code=proc.wait(timeout=10)
  ok=code==0 and not timed_out and not gpu_limit
  append({'event':'completed' if ok else 'failed','utc':stamp(),'name':job['name'],'stage':a.stage,'gpu':a.gpu,'exit_code':code,'timed_out':timed_out,'gpu_limit_exceeded':gpu_limit,'sampled_peak_GPU_MiB':peak,'GPU_monitor_samples':polls,'seconds':time.monotonic()-started,'log_sha256':sha(logpath),'output_sha256':{path:sha(ROOT/path) for path in job['new_outputs'] if (ROOT/path).is_file()}})
  if not ok:raise RuntimeError(f"failed {job['name']}; immutable logs retained")
if __name__=='__main__':main()
