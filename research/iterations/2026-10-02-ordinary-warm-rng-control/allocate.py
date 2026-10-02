"""Root-owned 20-minute mechanistic diagnostic allocation; no launching."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import importlib.util,json,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main():
 sp=importlib.util.spec_from_file_location('queue',HERE/'run_queue.py');Q=importlib.util.module_from_spec(sp);sp.loader.exec_module(Q)
 rel=str(HERE.relative_to(ROOT));out=HERE/'training-allocation.json';bp=HERE/'plain-parent-bindings.json'
 assert not out.exists() and not bp.exists()
 plan=json.loads((HERE/'commands.json').read_text());assert plan['counts']==dict(fits=2,OOF=2)
 generated=json.loads((HERE/'generated-01.json').read_text());assert generated['status']=='PASS_GENERATED_ORDINARY_WARM_RNG'
 check=json.loads((HERE/'generated-execution.json').read_text());assert check['status']=='COMPLETE' and not check['remaining_scope_pids']
 prior=ROOT/'research/iterations/2026-10-02-ordinary-warm-seed4'
 for suffix,status in [('execution-01/result.json','COMPLETE_4_FITS_2_OOF_NO_METRICS'),('finish-01/result.json','COMPLETE_FIXED_FORECAST_READOUT')]:
  r=json.loads((prior/suffix).read_text());assert r['status']==status and not r.get('remaining_scope_pids') and all(not j.get('remaining_scope_pids') for j in r.get('jobs',[]))
 for name in ('launch.json','finish-launch.json'):
  r=json.loads((prior/name).read_text());p=Path('/proc')/str(r['pid'])/'stat'
  if p.exists():
   z=p.read_text().split();assert z[21]!=r['start_ticks'] or z[2]=='Z'
 old=json.loads((ROOT/'research/iterations/2026-10-02-ordinary-warm/training-allocation.json').read_text())
 sources=dict(plan['source_sha256']);sources['research/iterations/2026-10-01-context-replay/run.py']=Q.sha(Q.HELPER)
 for name in ('commands.json','allocate.py','finish.py','launch_finish.py','freeze_evaluation.py','evaluate.py','generated-01.json','generated-execution.json'):
  sources[str((HERE/name).relative_to(ROOT))]=Q.sha(HERE/name)
 for relpath,h in sources.items():assert Q.sha(ROOT/relpath)==h,relpath
 jobs=[];records=[]
 def bind(p,h):
  p=Path(p);s=p.stat();assert Q.sha(p)==h
  records.append(dict(path=str(p),resolved=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,sha256=h))
 for parent in plan['plain_parents'].values():bind(ROOT/parent['checkpoint'],parent['sha256'])
 for j in plan['jobs']:
  if j['stage']=='OOF':
   jobs.append(dict(name=j['name'],stage='OOF',argv=j['argv'],timeout_seconds=300,require_absent=j['require_absent'],required_outputs=j['required_outputs']));continue
  assert j['stage']=='H2' and len(j['resolved_args'])==103
  bind(ROOT/j['warm_initial_receipt']['path'],j['warm_initial_receipt']['sha256'])
  run='artifacts/runs/'+j['name'];check=rel+'/execution-01/'+j['name']+'.check.json'
  jobs.append(dict(name=j['name'],stage='H2',argv=j['argv'],timeout_seconds=300,require_absent=[run,check],
   required_outputs=[run+'/model_ep0.pt',run+'/model_ep1.pt',run+'/model.pt',run+'/history.json',run+'/initial-state.json',check],
   post_argv=['.venv/bin/python',rel+'/check_checkpoint.py',j['name'],'--commands',rel+'/commands.json','--output',check]))
 for j in jobs:
  for p in j['require_absent']:assert not (ROOT/p).exists() and not (ROOT/p).is_symlink()
 g=Q.H.gpu(0);assert g['uuid']=='GPU-aa969728-9b37-ac58-7855-3f03494fb9e8' and g['free_MiB']>=8192
 assert shutil.disk_usage(ROOT).free>10*2**30
 now=datetime.now(timezone.utc);Q.write(bp,dict(status='BOUND_ORIGINAL_PARENTS_AND_W_H2_RNG_RECEIPTS',records=records))
 a=dict(execution_authorized=True,created_utc=now.isoformat(),deadline_utc=(now+timedelta(minutes=20)).isoformat(),execution_directory=rel+'/execution-01',
  success_status='COMPLETE_2_FITS_2_OOF_NO_METRICS',sources=sources,environment=plan['environment'],
  input_bindings=[old['input_bindings'][0],dict(path=str(bp.relative_to(ROOT)),sha256=Q.sha(bp))],jobs=jobs,
  resources=dict(gpu_id=0,gpu_uuid=g['uuid'],allocator_fraction=.375,owned_gpu_max_MiB=7168,initial_free_MiB=8192,running_free_MiB=1024,cpu_ids=list(range(16,22)),cpu_quota=600,host_memory_bytes=20*2**30),
  quality_queries=0,retries=0,interpretation='Mechanistic2H2fits only, no productionpromotion or recipe search. Primary WminusR; explanatoryRminusold.')
 Q.write(out,a);Q.contract(out,Q.sha(out));print(json.dumps(dict(status='ALLOCATED_NOT_LAUNCHED',allocation_sha256=Q.sha(out),deadline=a['deadline_utc'])))
if __name__=='__main__':main()
