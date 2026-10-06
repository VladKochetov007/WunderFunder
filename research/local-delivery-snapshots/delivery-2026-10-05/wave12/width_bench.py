import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
import sys
import time
import importlib.util
from pathlib import Path
from types import SimpleNamespace as Point
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave12'
P=R/'research/iterations/2026-10-02-big-ensemble/run_vlad_pair_bw'
sys.path.insert(0,str(D/os.environ.get('NATIVE_VARIANT','native')/'build/v3'))
import wnn_cpp
spec=importlib.util.spec_from_file_location('reference',P/'solution.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
import wnn_cpp
x=np.load(R/'research/iterations/2026-10-05-wave2/tstream_fold3_v2.probe.npz')['x']
timings={n:[] for n in ('reference','h128','h256','h384','h512')}
for order in (list(timings),list(timings)[::-1]):
    for name in order:
        if name=='reference':model=module.PredictionModel()
        else:
            w={k:np.ascontiguousarray(v,dtype=np.float32) for k,v in np.load(D/f'benchmark_{name}.npz').items()}
            model=wnn_cpp.Ensemble([w],[1.],fast_activations=True,half_weights=False,dynamic_int8=True,quant_backend='auto')
        t=time.perf_counter()
        for s in range(3):
            if name!='reference':model.reset()
            for i,row in enumerate(x):
                if name=='reference':value=model.predict(Point(seq_ix=s,step_in_seq=i,need_prediction=i>=99,state=row))
                else:value=model.step(row)
        timings[name].append(time.perf_counter()-t)
ref=np.mean(timings['reference'])
report={'timings_seconds':timings,'rows':len(x)*3,'ratio':{k:float(np.mean(v)/ref) for k,v in timings.items()},'public_minutes_relative_projection':{k:float(np.mean(v)/ref*(37+51/60)) for k,v in timings.items()},'caveat':'Random-weight width screening on repeated real inputs; relative projection to historical scorer, not scoring-host certification or model quality test.'}
(D/('width-runtime-'+os.environ.get('NATIVE_VARIANT','native')+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
