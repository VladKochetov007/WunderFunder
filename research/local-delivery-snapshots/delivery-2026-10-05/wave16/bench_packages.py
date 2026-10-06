import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import time
import json
import subprocess
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave16'
if len(sys.argv)>1:
 import pyarrow.parquet as pq
 from types import SimpleNamespace as Point
 folder=Path(sys.argv[1]);sys.path.insert(0,str(folder));from solution import PredictionModel
 pf=pq.ParquetFile(R/'datasets/valid.parquet');cols=pf.schema_arrow.names[4:116]
 seqs=[np.column_stack([c.to_numpy() for c in pf.read_row_group(i,columns=cols).columns]).astype('float32') for i in (0,5,12)]
 model=PredictionModel();t=time.perf_counter()
 for s,x in enumerate(seqs):
  for i,row in enumerate(x):model.predict(Point(seq_ix=s,step_in_seq=i,need_prediction=i>=99,state=row))
 print(json.dumps({'seconds':time.perf_counter()-t,'rows':sum(map(len,seqs))}),flush=True)
else:
 paths={'incumbent':R/'research/iterations/2026-10-02-big-ensemble/run_vlad_pair_bw','old_twostream':R/'research/iterations/2026-10-05-wave2/blend_twostream',**{n:D/n for n in ('sparse_twostream','sparse_twostream_fastbase','runtime_only_sparse_pair')}}
 results={n:[] for n in paths}
 for order in (list(paths),list(paths)[::-1]):
  for name in order:
   raw=subprocess.check_output([sys.executable,__file__,str(paths[name])],text=True);r=json.loads(raw);results[name].append(r['seconds']);print(name,r,flush=True)
 ref=np.mean(results['incumbent'])
 report={'timings_seconds':results,'rows_per_run':60000,'projected_public_minutes':{n:float(np.mean(t)/ref*(37+51/60)) for n,t in results.items()},'scope':'Independent subprocess per package to load its own binary. Alternate order, same input rows. Historical incumbent runtime calibration; not measured scoring host. Duplicate-seed pair for runtime only, not quality or submission.'}
 (D/'package-runtime.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
