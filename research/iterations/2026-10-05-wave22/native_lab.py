"""Independent CPU compatibility and relative runtime check for wider GRU."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
import json
import tempfile
import zipfile
import shutil
import time
from pathlib import Path
from types import SimpleNamespace as Point

def main():
 import numpy as np
 archive,weights,probe,output=map(Path,sys.argv[1:5]);root=Path(tempfile.mkdtemp())
 with zipfile.ZipFile(archive) as z:z.extractall(root/'reference')
 sys.path.insert(0,str(root/'reference'));import solution;import wnn_cpp
 wide=root/'wide';wide.mkdir();shutil.copy2(weights,wide/'weights.npz');(wide/'members.json').write_text(json.dumps([{'weights':'weights.npz','coef':1.,'name':'wide384'}]))
 z=np.load(probe);x=z['x'];ref=z['reference'];w={k:np.ascontiguousarray(v,np.float32) for k,v in np.load(weights).items()}
 fp=wnn_cpp.Ensemble([w],[1.],dynamic_int8=False).run(x[:2000]);err=float(abs(fp-ref[:2000]).max());assert err<.001,err
 q=wnn_cpp.Ensemble([w],[1.],dynamic_int8=True);got=q.run(x);corr=[float(np.corrcoef(ref[:,j],got[:,j])[0,1]) for j in (0,1)];assert min(corr)>.999,corr;assert np.array_equal(got,q.run(x))
 times={'reference':[],'wide':[]}
 for order in (('reference','wide'),('wide','reference')):
  for name in order:
   solution.HERE=root/name;model=solution.PredictionModel();start=time.perf_counter()
   for seq in range(3):
    for i,row in enumerate(x):model.predict(Point(seq_ix=seq,step_in_seq=i,need_prediction=i>=99,state=row))
   times[name].append(time.perf_counter()-start)
 ratio=float(np.mean(times['wide'])/np.mean(times['reference']))
 report={'backend':wnn_cpp.quant_backend(),'cpu_level':solution.LEVEL,'fp32_max_abs':err,'int8_correlation':corr,'reset_deterministic':True,'rows_per_run':3*len(x),'timings_seconds':times,'wide_vs_scored_pair_ratio':ratio,'wide_single_public_minutes_relative_projection':ratio*24.95,'wide_plus_h256_public_minutes_rough_projection':(ratio+.5)*24.95,'scope':'Untrained widened initialization, one real full sequence repeated three times for runtime only. Independent DataSphere CPU host, calibrated to user-reported 24m57s h256 pair. Not scoring-host certification; mixed estimate assumes additive member cost and requires actual package timing.'}
 output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__ == '__main__':
 main()
