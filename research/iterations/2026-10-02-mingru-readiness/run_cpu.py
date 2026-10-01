"""Bounded CPU-only stage; preserve failures and release only the owned scope."""
import argparse,hashlib,importlib.util,json,os,re,subprocess,time,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
HP=ROOT/'research/iterations/2026-10-01-context-replay/run.py'
spec=importlib.util.spec_from_file_location('helpers',HP);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--cpus',default='34');ap.add_argument('--seconds',type=int,required=True);ap.add_argument('--source-dir',type=Path,required=True);ap.add_argument('program',nargs=argparse.REMAINDER);a=ap.parse_args()
 assert re.fullmatch('[a-z0-9-]+',a.name) and 0<a.seconds<=900 and a.program
 if a.program[0]=='--':a.program=a.program[1:]
 src=a.source_dir.resolve();assert src.is_relative_to(HERE)
 out=HERE/(a.name+'-execution');out.mkdir();unit='mingru-'+uuid.uuid4().hex
 paths=[Path(__file__),HP,HERE/'contract.md',*[p for p in src.rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.hpp')]]
 sources={str(p):sha(p) for p in paths}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'))
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','ARROW_NUM_THREADS','WNN_THREADS'):env[k]='1'
 cmd=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=200%','-p',f'MemoryMax={4*2**30}','-p','MemorySwapMax=0','--','taskset','-c',a.cpus,'nice','-n','10','timeout','--signal=TERM','--kill-after=5s',str(a.seconds)+'s',*a.program]
 r=dict(status='RUNNING',argv=cmd,unit=unit,source_sha256=sources,real_data=False);t=time.monotonic()
 (out/'started.json').write_text(json.dumps(r,indent=2)+'\n')
 try:
  with (out/'stdout.log').open('x') as o,(out/'stderr.log').open('x') as e:rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=o,stderr=e).returncode
  r['exit_code']=rc;assert rc==0,('stage failed',rc)
  for p,h in sources.items():assert sha(p)==h,('source changed',p)
  r['status']='COMPLETE'
 except BaseException as e:r.update(status='FAILED_NO_RETRY',failure=repr(e));raise
 finally:
  fields,pids=H.scope(unit)
  if pids:H.command(['systemctl','--user','stop',unit+'.scope']);fields,pids=H.scope(unit)
  r.update(scope_state=fields,remaining_scope_pids=sorted(pids),seconds=time.monotonic()-t)
  (out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
