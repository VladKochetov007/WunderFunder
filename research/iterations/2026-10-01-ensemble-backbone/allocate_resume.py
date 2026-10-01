"""Resource-only continuation of unstarted jobs; never overwrite or rerun a fit."""
import json,hashlib,ast
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];B=str(HERE.relative_to(ROOT))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
ap=HERE/'training-allocation.json';rp=HERE/'execution-01/result.json'
assert sha(ap)=='d371acbe984865d64938cf42791724fe1894f2633e35087e0823216a8631ac93'
a=json.loads(ap.read_text());r=json.loads(rp.read_text())
assert r['status']=='FAILED_STOP_NO_RETRY' and r['allocation_sha256']==sha(ap)
assert r['fits']==4 and r['OOF_jobs']==1 and r['metric_calls']==0 and r['retries']==0
assert len(r['jobs'])==5
assert r['failure'].startswith('AssertionError(') and r['failure'].endswith(')')
g=ast.literal_eval(r['failure'][15:-1]);assert g['uuid']==a['resources']['gpu_uuid'] and g['free_MiB']<a['resources']['initial_free_MiB']
assert [(j['name'],j['stage']) for j in r['jobs']]==[(j['name'],j['stage']) for j in a['jobs'][:5]]
records=[]
for j,allocated in zip(r['jobs'],a['jobs']):
 assert j['exit_code']==0 and j['failure'] is None and j['remaining_scope_pids']==[]
 wr=j['worker_result'];p=ROOT/wr['path'];assert sha(p)==wr['sha256'];worker=json.loads(p.read_text())
 assert worker['status']=='COMPLETE' and worker['allocation_sha256']==sha(ap) and worker['argv']==allocated['argv']
 assert set(worker['outputs'])==set(allocated['required_outputs'])
 for rel,rec in worker['outputs'].items():
  p=ROOT/rel;s=p.stat();assert s.st_size==rec['bytes'] and sha(p)==rec['sha256'],rel
  records.append(dict(path=str(p),resolved=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,prior_sha256=rec['sha256']))
for oldjob in a['jobs'][5:]:
 assert not (HERE/'execution-01'/f"{oldjob['name']}.{oldjob['stage']}.worker-started.json").exists()
pb=HERE/'prefix-input-bindings.json';write(pb,dict(records=records,original_result_sha256=sha(rp),arrays_decoded=False))
a['jobs']=a['jobs'][5:]
for j in a['jobs']:
 for key in ('post_argv','required_outputs','require_absent'):
  if key in j:j[key]=[x.replace(B+'/execution-01/',B+'/execution-02/') for x in j[key]]
a['resources']=dict(a['resources'],gpu_id=0,gpu_uuid='GPU-aa969728-9b37-ac58-7855-3f03494fb9e8')
a.update(execution_directory=B+'/execution-02',success_status='COMPLETED_REMAINING_4_FITS_3_OOF_NO_METRICS',resource_resumption=dict(reason='Foreign process reduced GPU6 free memory below pre-job start threshold; no next worker was started.',original_allocation_sha256=sha(ap),original_result_sha256=sha(rp),completed_jobs_reused=5,scientific_jobs_repeated=0,created_utc=datetime.now(timezone.utc).isoformat()))
a['input_bindings'].append(dict(path=B+'/prefix-input-bindings.json',sha256=sha(pb)))
a['sources'][B+'/allocate_resume.py']=sha(__file__)
dest=HERE/'training-allocation-02.json';write(dest,a)
print(json.dumps(dict(allocation_path=str(dest.relative_to(ROOT)),sha256=sha(dest),remaining_jobs=len(a['jobs']),scientific_retries=0)))
