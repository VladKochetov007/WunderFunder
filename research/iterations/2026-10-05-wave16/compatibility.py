import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import json
import subprocess
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave16'
if len(sys.argv)>1:
 from types import SimpleNamespace as Point
 import pyarrow.parquet as pq
 folder=Path(sys.argv[1]);level=sys.argv[2];output=sys.argv[3]
 sys.path.insert(0,str(folder/level));import wnn_cpp
 sys.path.insert(0,str(folder));from solution import PredictionModel
 model=PredictionModel();pf=pq.ParquetFile(R/'datasets/valid.parquet');cols=pf.schema_arrow.names[4:116];out=[]
 for group in (0,5,12,55,123):
  tab=pf.read_row_group(group,columns=cols);x=np.column_stack([c.to_numpy() for c in tab.columns]).astype('float32')[:2000]
  for i,row in enumerate(x):
   p=model.predict(Point(seq_ix=group,step_in_seq=i,need_prediction=i>=99,state=row))
   if p is not None:out.append(p.copy())
 np.save(output,np.asarray(out))
else:
 report=[]
 for level in ('v2','avx2','v3'):
  for name,folder in [('legacy',R/'research/iterations/2026-10-05-wave2/blend_twostream'),('sparse',D/'sparse_twostream')]:
   subprocess.run([sys.executable,__file__,str(folder),level,str(D/f'compat-{name}-{level}.npy')],check=True)
  old=np.load(D/f'compat-legacy-{level}.npy');new=np.load(D/f'compat-sparse-{level}.npy')
  assert np.array_equal(old,new),(level,float(abs(old-new).max()))
  report.append({'level':level,'rows':10000,'whole_package_bit_exact_to_original':True})
 (D/'compatibility.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
