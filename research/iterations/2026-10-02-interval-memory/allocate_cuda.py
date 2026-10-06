"""Root's single generated resource check, no market data or forecast fit."""
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REL=str(HERE.relative_to(ROOT))
spec=importlib.util.spec_from_file_location('queue', HERE/'run_queue.py')
Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
assert Q.sha(HERE/'run_queue.py')=='a6d51d912635c071bcf001d98e1e7e251718992a40c84db2a4586345019bd5c2'
assert Q.sha(HERE/'check_cuda.py')=='16e2dea1d9070de44328715f94e97b065680d10d05f9b4aa44fa0d6f650a5e90'
g=Q.H.gpu(0)
assert g['uuid']=='GPU-aa969728-9b37-ac58-7855-3f03494fb9e8' and g['free_MiB']>=8192,g
files=[HERE/'run_queue.py',HERE/'check_cuda.py',HERE/'interval_memory.py',Path(__file__),Q.HELPER]
files+=list((ROOT/'src/wnn').glob('*.py'))
now=datetime.now(timezone.utc)
environment={k:'2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','ARROW_NUM_THREADS','NUMEXPR_NUM_THREADS','WNN_THREADS')}
a=dict(execution_authorized=True,created_utc=now.isoformat(),deadline_utc=(now+timedelta(minutes=2)).isoformat(),
 execution_directory=REL+'/cuda-execution-01',success_status='PASS_GENERATED_CUDA_SMOKE',
 resources=dict(gpu_id=0,gpu_uuid=g['uuid'],allocator_fraction=.375,owned_gpu_max_MiB=7168,initial_free_MiB=8192,running_free_MiB=1024,cpu_ids=[37,38],cpu_quota=200,host_memory_bytes=4*2**30),
 environment=environment,sources={str(p.relative_to(ROOT)):Q.sha(p) for p in files},input_bindings=[],
 jobs=[dict(name='interval-generated',stage='CUDA_generated',argv=['.venv/bin/python',REL+'/check_cuda.py','--output-dir',REL+'/cuda-01','--seconds','45','--gpu-fraction','.375','--threads','2'],timeout_seconds=60,require_absent=[REL+'/cuda-01'],required_outputs=[REL+'/cuda-01/result.json'])],
 gpu_preflight=g,quality_queries=0,retries=0)
target=HERE/'cuda-allocation.json';Q.write(target,a);Q.contract(target,Q.sha(target))
print(json.dumps(dict(status='ALLOCATED_GENERATED_ONLY',sha256=Q.sha(target))))
