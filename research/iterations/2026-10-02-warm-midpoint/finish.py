"""Wait for this fixed queue, then one bounded CPU readout. Never retry a fit/readout."""
import hashlib,importlib.util,json,os,subprocess,time,uuid
from datetime import datetime,timezone,timedelta
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('queue',HERE/'run_queue.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 freeze=HERE/'evaluation-source.json';fh=Q.sha(freeze);f=json.loads(freeze.read_text());ap=HERE/'training-allocation.json';assert Q.sha(ap)==f['training_allocation_sha256'];a=json.loads(ap.read_text())
 for rel,h in f['source_sha256'].items():assert Q.sha(ROOT/rel)==h,rel
 deadline=datetime.fromisoformat(a['deadline_utc'])+timedelta(minutes=15)
 out=HERE/'finish-01';out.mkdir();result=HERE/'execution-01/result.json';unit='warm-midpoint-score-'+uuid.uuid4().hex
 rec=dict(status='WAITING_FOR_FIXED_TRAINING',evaluation_source_sha256=fh,allocation_sha256=Q.sha(ap),unit=unit,started_utc=datetime.now(timezone.utc).isoformat());write(out/'started.json',rec)
 try:
  while not result.exists():
   assert datetime.now(timezone.utc)<deadline,'fixed completion deadline'
   time.sleep(20)
  r=json.loads(result.read_text());assert r['status']=='COMPLETE_0_FITS_4_OOF_NO_METRICS' and r['allocation_sha256']==Q.sha(ap),(r['status'],r.get('failure'))
  assert Q.sha(freeze)==fh
  for rel,h in f['source_sha256'].items():assert Q.sha(ROOT/rel)==h,rel
  argv=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=200%','-p',f'MemoryMax={4*2**30}','-p','MemorySwapMax=0','--','taskset','-c','28,29','nice','-n','10','timeout','--signal=TERM','--kill-after=5s','600s','.venv/bin/python',str((HERE/'evaluate.py').relative_to(ROOT)),'--evaluation-source-sha256',fh,'--output-dir',str((HERE/'evaluation-01').relative_to(ROOT))]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'))
  for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','ARROW_NUM_THREADS','WNN_THREADS'):env[k]='1'
  rec['argv']=argv
  with (out/'evaluate.log').open('x') as log:rc=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  rec['exit_code']=rc;assert rc==0,('readout failed',rc)
  report=HERE/'evaluation-01/report.json';q=json.loads(report.read_text());assert q['status']=='COMPLETE_FIXED_WARM_MIDPOINT_SCREEN'
  rec.update(status='COMPLETE_FIXED_FORECAST_READOUT',report_sha256=Q.sha(report))
 except BaseException as exc:rec.update(status='FAILED_NO_RETRY',failure=repr(exc));raise
 finally:
  fields,pids=Q.H.scope(unit)
  if pids:Q.H.command(['systemctl','--user','stop',unit+'.scope']);fields,pids=Q.H.scope(unit)
  rec.update(scope_state=fields,remaining_scope_pids=sorted(pids),finished_utc=datetime.now(timezone.utc).isoformat());write(out/'result.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':main()
