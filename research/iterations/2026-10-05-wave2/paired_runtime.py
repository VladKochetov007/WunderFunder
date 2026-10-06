"""Compare new packages to the scored incumbent under the same timing harness."""
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace as Point

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pyarrow.parquet as pq

def main():
    R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave2'
    paths={'reference':R/'research/iterations/2026-10-02-big-ensemble/run_vlad_pair_bw',
           'full':D/'blend_twostream','lite':D/'blend_twostream_lite'}
    modules={}
    for name,path in paths.items():
        spec=importlib.util.spec_from_file_location('package_'+name,path/'solution.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules[name]=module
    pf=pq.ParquetFile(R/'datasets/valid.parquet');cols=pf.schema_arrow.names[4:116]
    seqs=[np.column_stack([c.to_numpy() for c in pf.read_row_group(i,columns=cols).columns]).astype('float32') for i in (0,5,12)]
    timings={n:[] for n in paths}
    for order in (('reference','full','lite'),('lite','full','reference')):
        for name in order:
            m=modules[name].PredictionModel();start=time.perf_counter()
            for s,x in enumerate(seqs):
                for i,row in enumerate(x):m.predict(Point(seq_ix=s,step_in_seq=i,need_prediction=i>=99,state=row))
            timings[name].append(time.perf_counter()-start)
    ref=float(np.mean(timings['reference']))
    report={'rows_per_run':60000,'timings_seconds':timings,'historical_incumbent_public_minutes':37+51/60,
            'ratio_to_incumbent':{n:float(np.mean(t))/ref for n,t in timings.items()},
            'public_minutes_if_relative_speed_transfers':{n:float(np.mean(t))/ref*(37+51/60) for n,t in timings.items()},
            'caveat':'Same-harness relative timing, calibrated to recorded 37m51s incumbent run. Not a measured scorer runtime for new packages.'}
    (D/'paired-runtime.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
