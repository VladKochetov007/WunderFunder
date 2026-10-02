"""Root-only four-OOF/no-fit allocation after fixed CPU construction."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import importlib.util,json,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main():
 sp=importlib.util.spec_from_file_location('q',HERE/'run_queue.py');Q=importlib.util.module_from_spec(sp);sp.loader.exec_module(Q)
 rel=str(HERE.relative_to(ROOT));target=HERE/'training-allocation.json';bp=HERE/'constructed-bindings.json';assert not target.exists() and not bp.exists()
 plan=json.loads((HERE/'commands.json').read_text());assert plan['counts']==dict(fits=0,OOF=4) and plan['alpha']==.5
 prepared=json.loads((HERE/'prepare-01/result.json').read_text());assert prepared['status']=='COMPLETE_FIXED_MIDPOINT_CPU_PREPARATION' and not prepared['remaining_scope_pids']
 check=HERE/'prepare-01/checkpoints.json';assert Q.sha(check)==prepared['checkpoints_sha256'];checks=json.loads(check.read_text());assert checks['status']=='PASS_FIXED_WARM_MIDPOINT_CHECKPOINTS'
 prior=ROOT/'research/iterations/2026-10-02-ordinary-warm-rng-control'
 for suffix,status in [('execution-01/result.json','COMPLETE_2_FITS_2_OOF_NO_METRICS'),('finish-01/result.json','COMPLETE_FIXED_FORECAST_READOUT')]:
  r=json.loads((prior/suffix).read_text());assert r['status']==status and not r.get('remaining_scope_pids') and all(not j.get('remaining_scope_pids') for j in r.get('jobs',[]))
 for name in ('launch.json','finish-launch.json'):
  r=json.loads((prior/name).read_text());p=Path('/proc')/str(r['pid'])/'stat'
  if p.exists():
   z=p.read_text().split();assert z[21]!=r['start_ticks'] or z[2]=='Z'
 sources=dict(plan['source_sha256']);sources['research/iterations/2026-10-01-context-replay/run.py']=Q.sha(Q.HELPER)
 for name in ('commands.json','allocate.py','finish.py','launch_finish.py','freeze_evaluation.py','evaluate.py','prepare_once.py','preparation-allocation.json','prepare-01/result.json','prepare-01/checkpoints.json','root-contract.json'):
  sources[str((HERE/name).relative_to(ROOT))]=Q.sha(HERE/name)
 for p,h in sources.items():assert Q.sha(ROOT/p)==h,p
 records=[]
 for row in checks['checkpoints']:
  for binding in (row['checkpoint'],dict(path=str(Path(row['checkpoint']['path']).parent/'construction.json'),sha256=row['construction_receipt_sha256'])):
   p=ROOT/binding['path'];s=p.stat();assert Q.sha(p)==binding['sha256']
   records.append(dict(path=str(p),resolved=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,sha256=binding['sha256']))
 jobs=[]
 for j in plan['jobs']:
  assert j['stage']=='OOF'
  job={k:j[k] for k in ('name','stage','argv','require_absent','required_outputs')};job['timeout_seconds']=300;jobs.append(job)
  for p in j['require_absent']:assert not (ROOT/p).exists() and not (ROOT/p).is_symlink()
 g=Q.H.gpu(0);assert g['uuid']=='GPU-aa969728-9b37-ac58-7855-3f03494fb9e8' and g['free_MiB']>=8192
 assert shutil.disk_usage(ROOT).free>10*2**30
 old=json.loads((ROOT/'research/iterations/2026-10-02-ordinary-warm/training-allocation.json').read_text());now=datetime.now(timezone.utc)
 Q.write(bp,dict(status='BOUND_FOUR_CONSTRUCTED_MIDPOINTS',records=records))
 a=dict(execution_authorized=True,created_utc=now.isoformat(),deadline_utc=(now+timedelta(minutes=20)).isoformat(),execution_directory=rel+'/execution-01',success_status='COMPLETE_0_FITS_4_OOF_NO_METRICS',
  sources=sources,environment=old['environment'],input_bindings=[old['input_bindings'][0],dict(path=str(bp.relative_to(ROOT)),sha256=Q.sha(bp))],jobs=jobs,
  resources=dict(gpu_id=0,gpu_uuid=g['uuid'],allocator_fraction=.375,owned_gpu_max_MiB=7168,initial_free_MiB=8192,running_free_MiB=1024,cpu_ids=list(range(16,22)),cpu_quota=600,host_memory_bytes=20*2**30),quality_queries=0,retries=0)
 Q.write(target,a);Q.contract(target,Q.sha(target));print(json.dumps(dict(status='ALLOCATED_NOT_LAUNCHED',allocation_sha256=Q.sha(target),deadline=a['deadline_utc'])))
if __name__=='__main__':main()
