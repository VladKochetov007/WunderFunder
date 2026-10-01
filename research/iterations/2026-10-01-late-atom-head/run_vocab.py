"""One bounded CPU-only training dictionary build; no automatic retry."""
import os,json,subprocess,uuid,time,hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('queue',HERE/'run_queue.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
plan=json.loads((HERE/'commands.json').read_text())['vocab'];unit='late-vocab-'+uuid.uuid4().hex
assert not (HERE/'vocab-run.json').exists()
for rel in plan['required_outputs']:assert not (ROOT/rel).exists(),rel
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','ARROW_NUM_THREADS','WNN_THREADS'):env[k]='2'
argv=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=200%','-p',f'MemoryMax={8*2**30}','-p','MemorySwapMax=0','--','taskset','-c','22,23','nice','-n','10','timeout','--signal=TERM','--kill-after=5s','600s',*plan['argv']]
r=dict(status='RUNNING',unit=unit,argv=argv,source_sha256={p.name:Q.sha(p) for p in HERE.glob('*.py')});t=time.monotonic()
try:
 with (HERE/'vocab.log').open('x') as log:rc=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 r['returncode']=rc;assert rc==0
 r['outputs']={rel:dict(bytes=(ROOT/rel).stat().st_size,sha256=Q.sha(ROOT/rel)) for rel in plan['required_outputs']};r['status']='COMPLETE'
except BaseException as exc:r.update(status='FAILED_NO_RETRY',failure=repr(exc));raise
finally:
 fields,pids=Q.H.scope(unit)
 if pids:Q.H.command(['systemctl','--user','stop',unit+'.scope']);fields,pids=Q.H.scope(unit)
 r.update(scope_state=fields,remaining_scope_pids=sorted(pids),seconds=time.monotonic()-t);Q.write(HERE/'vocab-run.json',r)
