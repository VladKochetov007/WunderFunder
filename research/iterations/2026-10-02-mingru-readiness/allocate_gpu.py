"""One generated GPU probe, contingent on completed CPU correctness checks."""
import json,importlib.util
from pathlib import Path
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];REL=str(HERE.relative_to(ROOT))
spec=importlib.util.spec_from_file_location('queue',HERE/'run_queue.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
assert json.loads((HERE/'numerical/cpu-01.json').read_text())['status']=='PASS_GENERATED_CPU'
assert json.loads((HERE/'numerical-cpu-execution/result.json').read_text())['status']=='COMPLETE'
g=Q.H.gpu(0);assert g['uuid']=='GPU-aa969728-9b37-ac58-7855-3f03494fb9e8' and g['free_MiB']>=8192
paths=[*HERE.glob('*.py'),*sorted((HERE/'numerical').glob('*.py')),HERE/'contract.md',ROOT/'src/wnn/nn.py',Q.HELPER]
sources={str(p.relative_to(ROOT)):Q.sha(p) for p in paths}
now=datetime.now(timezone.utc)
a=dict(execution_authorized=True,created_utc=now.isoformat(),deadline_utc=(now+timedelta(minutes=10)).isoformat(),execution_directory=REL+'/gpu-execution-01',success_status='COMPLETE_GENERATED_GPU_PROBE',resources=dict(gpu_id=0,gpu_uuid=g['uuid'],allocator_fraction=.30,owned_gpu_max_MiB=6144,initial_free_MiB=8192,running_free_MiB=1024,cpu_ids=[36],cpu_quota=100,host_memory_bytes=4*2**30),environment={k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','ARROW_NUM_THREADS','WNN_THREADS')},sources=sources,input_bindings=[],jobs=[dict(name='mingru_generated',stage='probe',argv=['.venv/bin/python',REL+'/numerical/check_gpu.py','--output',REL+'/numerical/gpu-01.json'],timeout_seconds=180,require_absent=[REL+'/numerical/gpu-01.json'],required_outputs=[REL+'/numerical/gpu-01.json'])],gpu_preflight=g)
Q.write(HERE/'gpu-allocation.json',a);print(json.dumps(dict(status='ALLOCATED_GENERATED_GPU',sha256=Q.sha(HERE/'gpu-allocation.json'),gpu=g)))
