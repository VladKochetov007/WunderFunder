"""Root-owned fixed four-fit allocation; readiness required, no scientific retries."""
import json,importlib.util,shutil
from datetime import datetime,timezone,timedelta
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];REL=str(HERE.relative_to(ROOT))
spec=importlib.util.spec_from_file_location('queue',HERE/'run_queue.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
def main():
 readiness=ROOT/'research/iterations/2026-10-02-mingru-readiness'
 assert json.loads((readiness/'numerical/cpu-01.json').read_text())['status']=='PASS_GENERATED_CPU'
 assert json.loads((readiness/'numerical/gpu-01.json').read_text())['status']=='PASS_GENERATED_NUMERICAL_INVARIANTS_WITH_DRIFT_REPORTED'
 decision=json.loads((readiness/'decision.json').read_text());assert decision['forecast_allocation']=='PROCEED_FIXED_TWO_FOLD_SCREEN'
 plan=json.loads((HERE/'commands.json').read_text())
 g=Q.H.gpu(0);assert g['uuid']=='GPU-aa969728-9b37-ac58-7855-3f03494fb9e8' and g['free_MiB']>=8192
 assert shutil.disk_usage(ROOT).free>10*2**30
 prior=ROOT/'research/iterations/2026-10-01-ensemble-backbone/training-allocation.json'
 old=json.loads(prior.read_text());src={k:v for k,v in old['sources'].items() if k.startswith(('src/','scripts/')) or k.endswith('/train_aux.py') or k.endswith('context-replay/run.py')}
 for p in HERE.glob('*'):
  if p.name not in ('evaluate.py','evaluation-source.json') and p.is_file() and p.suffix in ('.py','.json','.md'):src[str(p.relative_to(ROOT))]=Q.sha(p)
 src[str(readiness.relative_to(ROOT))+'/decision.json']=Q.sha(readiness/'decision.json')
 jobs=[]
 for j in plan['jobs']:
  stage=j['stage'];assert stage in ('parent','H2','OOF')
  if stage=='OOF':
   jobs.append(dict(name=j['name'],stage=stage,argv=j['argv'],timeout_seconds=900,require_absent=j['require_absent'],required_outputs=j['required_outputs']))
   continue
  check=REL+'/execution-01/'+j['name']+'.check.json'
  run='artifacts/runs/'+j['name'];ep=3 if stage=='parent' else 1
  jobs.append(dict(name=j['name'],stage=stage,argv=j['argv'],timeout_seconds=6000 if stage=='parent' else 1800,require_absent=[run,check],required_outputs=[run+f'/model_ep{i}.pt' for i in range(ep+1)]+[run+'/model.pt',run+'/history.json',check],post_argv=['.venv/bin/python',REL+'/check_checkpoint.py',j['name'],'--commands',REL+'/commands.json','--output',check]))
 assert [j['stage'] for j in jobs]==['parent','H2','OOF','parent','H2','OOF']
 now=datetime.now(timezone.utc)
 env=dict(old['environment']);env.update(WNN_THREADS='4',WNN_LOADER_WORKERS='2',WNN_LOADER_RAM_GB='4')
 a=dict(execution_authorized=True,created_utc=now.isoformat(),deadline_utc=(now+timedelta(hours=5)).isoformat(),execution_directory=REL+'/execution-01',success_status='COMPLETE_4_FITS_2_OOF_NO_METRICS',resources=dict(gpu_id=0,gpu_uuid=g['uuid'],allocator_fraction=.375,owned_gpu_max_MiB=7168,initial_free_MiB=8192,running_free_MiB=1024,cpu_ids=[22,24,32,34,36,39],cpu_quota=600,host_memory_bytes=20*2**30),environment=env,sources=src,input_bindings=[old['input_bindings'][0]],jobs=jobs,gpu_preflight=g,quality_queries=0,retries=0)
 for j in jobs:
  for rel in j['require_absent']:assert not (ROOT/rel).exists() and not (ROOT/rel).is_symlink(),rel
 Q.write(HERE/'training-allocation.json',a)
 Q.contract(HERE/'training-allocation.json',Q.sha(HERE/'training-allocation.json'))
 print(json.dumps(dict(status='ALLOCATED_FIXED_MINGRU_SCREEN',sha256=Q.sha(HERE/'training-allocation.json'),deadline=a['deadline_utc'],gpu=g)))
if __name__=='__main__':main()
