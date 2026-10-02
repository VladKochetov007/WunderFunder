"""Root-owned generated native check, CPU37/4GiB/60s, no forecast data."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,os,subprocess,time,uuid
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('resources',ROOT/'research/iterations/2026-10-01-context-replay/run.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
assert H.sha(HERE/'native-probe/prepare.py')=='3ff1ad87f9887cd2f35366e9a641b46d5d7efa10c91c70d345b350a8c30960b8'
assert H.sha(HERE/'native-probe/check_benchmark.py')=='f86aee3faf151c28e9e348d2bb9d66e9f3b1b81cb727001d61ed192d4f1e4c14'
out=HERE/'native-execution-01';out.mkdir()
unit='interval-native-'+uuid.uuid4().hex;assert not H.scope(unit)[1]
argv=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=100%','-p',f'MemoryMax={4*2**30}','-p','MemorySwapMax=0','--','taskset','-c','37','nice','-n','10','timeout','--signal=TERM','--kill-after=5s','60s','.venv/bin/python',str((HERE/'native-probe/prepare.py').relative_to(ROOT)),'--output-dir',str((HERE/'native-probe/execution-01').relative_to(ROOT)),'--cpu','37']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
env.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','WNN_THREADS')})
r=dict(status='RUNNING',unit=unit,argv=argv,started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=H.sha(__file__),forecast_data=False,metrics=0,retries=0)
(out/'started.json').write_text(json.dumps(r,indent=2)+'\n');start=time.monotonic()
try:
 with (out/'run.log').open('x') as f:rc=subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 r['exit_code']=rc;assert rc==0,rc
 p=HERE/'native-probe/execution-01/execution.json';v=json.loads(p.read_text());assert v['status']=='COMPLETE_GENERATED_NATIVE_PREPARATION'
 r.update(status='COMPLETE_GENERATED_NATIVE_SCOPE',result_sha256=H.sha(p))
except BaseException as e:r.update(status='FAILED_NO_RETRY',failure=repr(e));raise
finally:
 fields,pids=H.scope(unit)
 if pids:H.command(['systemctl','--user','stop',unit+'.scope']);fields,pids=H.scope(unit)
 r.update(scope_state=fields,remaining_scope_pids=sorted(pids),seconds=time.monotonic()-start)
 (out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
