"""Single allocated generated probe controller; static GPU3/CPU17, no retries."""
from pathlib import Path
import hashlib,json,subprocess,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stage=json.loads((P/'gpu-stage-source.json').read_text())
assert sha(P/'minimal_rnn.py')==stage['model_sha256'] and sha(P/'gpu_generated_probe.py')==stage['probe_sha256']
def smi():
 r=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5)
 total,used,free=map(int,r.strip().split(','));return {'total_MiB':total,'used_MiB':used,'free_MiB':free}
before=smi();assert before['used_MiB']<128 and before['free_MiB']>=12000,before
out=P/'gpu-generated-01';out.mkdir()
unit='cont-mingru-gpu-generated-01'
state=subprocess.run(['systemctl','--user','show',unit+'.scope','-p','LoadState','--value'],capture_output=True,text=True,timeout=5)
assert state.stdout.strip() in ('','not-found')
cmd=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','MemoryMax=4G','-p','MemorySwapMax=0','-p','CPUQuota=100%','--','taskset','-c','17','nice','-n','10','env','CUDA_VISIBLE_DEVICES=3','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','NUMBA_NUM_THREADS=1','/usr/bin/timeout','--signal=TERM','--kill-after=5s','180s','/usr/bin/time','-v',str(ROOT/'.venv/bin/python'),str(P/'gpu_generated_probe.py')]
start=time.monotonic();failure=None;samples=[]
with (out/'probe.jsonl').open('x') as stdout,(out/'probe.stderr').open('x') as stderr:
 process=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
 try:
  while process.poll() is None:
   sample={'seconds':time.monotonic()-start,**smi()};samples.append(sample)
   with (out/'NVML.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
   if sample['used_MiB']>9216:raise RuntimeError('device-total memory envelope exceeded')
   if time.monotonic()-start>190:raise RuntimeError('outer containment timeout')
   time.sleep(.5)
  if process.returncode!=0:failure='probe exited '+str(process.returncode)
 except BaseException as exc:
  failure=repr(exc)
  stopped=subprocess.run(['systemctl','--user','stop',unit+'.scope'],capture_output=True,text=True,timeout=10)
  if stopped.returncode:subprocess.run(['systemctl','--user','kill','--signal=KILL',unit+'.scope'],capture_output=True,timeout=10)
  process.wait(timeout=10)
released=subprocess.run(['systemctl','--user','show',unit+'.scope','-p','ActiveState','-p','SubState'],capture_output=True,text=True,timeout=5)
result={'status':'PASS' if failure is None else 'FAILED_NO_RETRY','failure':failure,'exit_code':process.returncode,
        'seconds':time.monotonic()-start,'GPU_before':before,'GPU_after':smi(),'peak_sampled_total_used_MiB':max(s['used_MiB'] for s in samples),
        'scope':unit,'scope_release':released.stdout,'argv':cmd,'stage_sha256':sha(P/'gpu-stage-source.json'),
        'controller_sha256':sha(Path(__file__)),'model_sha256':stage['model_sha256'],'probe_sha256':stage['probe_sha256'],
        'real_data_or_checkpoints_read':False,'forecast_fit_or_metric':False}
(out/'execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
