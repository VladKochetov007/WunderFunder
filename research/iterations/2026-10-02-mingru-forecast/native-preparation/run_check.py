"""Root-owned single generated check; training, scoring and old artifacts untouched."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,time,uuid
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
HP=ROOT/'research/iterations/2026-10-01-context-replay/run.py'
spec=importlib.util.spec_from_file_location('helpers',HP);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=HERE/'check-01';out.mkdir();unit='mingru-adapter-'+uuid.uuid4().hex
 paths=[*HERE.glob('*.py'),HERE/'contract.md',HP];sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'))
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','ARROW_NUM_THREADS','WNN_THREADS'):env[k]='1'
 argv=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=100%','-p',f'MemoryMax={4*2**30}','-p','MemorySwapMax=0','--','taskset','-c','35','nice','-n','10','timeout','--signal=TERM','--kill-after=5s','30s','.venv/bin/python',str((HERE/'adapter_check.py').relative_to(ROOT)),'--output',str((out/'result.json').relative_to(ROOT))]
 r=dict(status='RUNNING',unit=unit,argv=argv,source_sha256=sources,real_data=False,forecast_fits=0);t=time.monotonic();(out/'started.json').write_text(json.dumps(r,indent=2)+'\n')
 try:
  with (out/'stdout.log').open('x') as o,(out/'stderr.log').open('x') as e:rc=subprocess.run(argv,cwd=ROOT,env=env,stdout=o,stderr=e).returncode
  r['exit_code']=rc;assert rc==0,('generated check failed',rc)
  for name,h in sources.items():assert sha(ROOT/name)==h,name
  r.update(status='COMPLETE_GENERATED_ADAPTER_CHECK',result_sha256=sha(out/'result.json'))
 except BaseException as exc:r.update(status='FAILED_NO_RETRY',failure=repr(exc));raise
 finally:
  fields,pids=H.scope(unit)
  if pids:H.command(['systemctl','--user','stop',unit+'.scope']);fields,pids=H.scope(unit)
  r.update(scope_state=fields,remaining_scope_pids=sorted(pids),seconds=time.monotonic()-t);(out/'execution.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
