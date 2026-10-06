"""Controller allocation, evaluated after metadata preparation on beleriand."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timedelta,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];B=str(HERE.relative_to(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
inputs=HERE/'input-bindings.json';r=json.loads(inputs.read_text());sources=dict(r['source_sha256'])
for rel in [B+'/allocate_cache.py',B+'/run_queue.py',B+'/contract.md','research/iterations/2026-10-01-context-replay/run.py']:
 sources[rel]=sha(ROOT/rel)
now=datetime.now(timezone.utc)
jobs=[]
for fold in (0,1):
 for slot in ('128s3','128s4','c500'):
  name=f'on_ctl{slot}_f{fold}';p=f'{B}/cache/pred_{name}_train'
  jobs.append(dict(name=name,stage='anchor',argv=['.venv/bin/python',B+'/cache_partners.py',name,'--fold',str(fold),'--inputs',B+'/input-bindings.json','--inputs-sha256',sha(inputs)],timeout_seconds=1200,require_absent=[p+'.npy',p+'.npy.partial',p+'.json'],required_outputs=[p+'.npy',p+'.json']))
a=dict(execution_authorized=True,sources=sources,input_bindings=[dict(path=B+'/input-bindings.json',sha256=sha(inputs))],jobs=jobs,execution_directory=B+'/cache-execution-01',success_status='COMPLETED_6_ANCHOR_STREAMS_NO_METRICS',deadline_utc=(now+timedelta(hours=4)).isoformat(),campaign_start_utc=now.isoformat(),resources=dict(gpu_id=6,gpu_uuid='GPU-53eaee12-1803-f8da-7399-ccdbfda91a18',allocator_fraction=.375,owned_gpu_max_MiB=7168,initial_free_MiB=8192,running_free_MiB=1024,cpu_ids=[22,23,24,25,28,32],cpu_quota=600,host_memory_bytes=20*2**30),environment=dict(WNN_THREADS='4',WNN_LOADER_WORKERS='2',WNN_LOADER_RAM_GB='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMBA_NUM_THREADS='4',ARROW_NUM_THREADS='4',WNN_GPU_DUTY='1.0'))
dest=HERE/'cache-allocation.json';write(dest,a);print(json.dumps(dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),jobs=len(jobs),deadline_utc=a['deadline_utc'])))
