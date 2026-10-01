"""Wait for four inference jobs, then one bounded CPU diagnosis."""
import os,json,time,subprocess,uuid,importlib.util
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('queue',HERE/'run_queue.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
a=json.loads((HERE/'allocation.json').read_text());deadline=datetime.fromisoformat(a['deadline_utc']);start=time.monotonic();unit='stage-grade-'+uuid.uuid4().hex
r=dict(status='WAITING',unit=unit);(HERE/'finish-started.json').write_text(json.dumps(r)+'\n')
try:
 p=HERE/'execution-01/result.json'
 while not p.exists():
  assert datetime.now(timezone.utc)<deadline;time.sleep(2)
 assert json.loads(p.read_text())['status']=='TWO_ATOM_VARIANT_PASSES'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'))
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','ARROW_NUM_THREADS']:env[k]='2'
 argv=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=200%','-p',f'MemoryMax={8*2**30}','-p','MemorySwapMax=0','--','taskset','-c','22,23','nice','-n','10','timeout','--signal=TERM','--kill-after=5s','180s','.venv/bin/python',str(HERE/'evaluate.py'),'--output-dir',str(HERE/'evaluation-01')]
 with (HERE/'evaluate.log').open('x') as log:rc=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 r['exit_code']=rc;assert rc==0
 assert json.loads((HERE/'evaluation-01/receipt.json').read_text())['status']=='COMPLETE_FIXED_ATOM_DECODER_PROBE'
 r['status']='COMPLETE_FIXED_ATOM_DECODER_PROBE'
except BaseException as e:r.update(status='FAILED_NO_RETRY',failure=repr(e));raise
finally:
 fields,pids=Q.H.scope(unit)
 if pids:
  Q.H.command(['systemctl','--user','stop',unit+'.scope']);fields,pids=Q.H.scope(unit)
 r.update(scope_state=fields,remaining_scope_pids=sorted(pids),seconds=time.monotonic()-start)
 Q.write(HERE/'finish-result.json',r)
