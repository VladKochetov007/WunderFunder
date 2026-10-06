import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
import json
import time
from pathlib import Path
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave16'
level=sys.argv[1] if len(sys.argv)>1 else 'v3'
sys.path.insert(0,str(D/'native-sparse/build'/level))
import wnn_cpp
w={k:np.ascontiguousarray(v,dtype=np.float32) for k,v in np.load(R/'research/iterations/2026-10-05-wave2/tstream_final.npz').items()}
p=np.load(R/'research/iterations/2026-10-05-wave2/tstream_final.probe.npz');x=p['x'];x=x[:2000]
sparse={**w,'meta.gru_blocks':np.array([128,128],np.float32)}
report={'level':level,'rows':len(x),'cases':[]}
reference=None
for backend in ('scalar','avx2','vnni'):
 a=wnn_cpp.Ensemble([w],[1.],dynamic_int8=True,quant_backend=backend)
 b=wnn_cpp.Ensemble([sparse],[1.],dynamic_int8=True,quant_backend=backend)
 yy=a.run(x);zz=b.run(x)
 assert np.array_equal(yy,zz),(backend,float(abs(yy-zz).max()))
 assert np.array_equal(zz,b.run(x))
 if reference is None:reference=zz
 else:assert np.array_equal(reference,zz)
 b.reset();ss=np.stack([b.step(row) for row in x]);assert np.array_equal(ss,zz)
 report['cases'].append({'backend':backend,'bit_exact':True,'reset_and_step_parity':True})
for widths in ([64,192],[128,64],[128,128,128],[0,256],[127.5,128.5],[float('nan'),128.]):
 bad={**w,'meta.gru_blocks':np.array(widths,np.float32)}
 try:wnn_cpp.Ensemble([bad],[1.],dynamic_int8=True)
 except ValueError:pass
 else:raise AssertionError(('accepted invalid metadata',widths))
bad={**sparse,'gru.weight_ih_l0':w['gru.weight_ih_l0'].copy()};bad['gru.weight_ih_l0'][0,128]=1e-20
try:wnn_cpp.Ensemble([bad],[1.],dynamic_int8=True)
except ValueError:pass
else:raise AssertionError('nonzero offblock accepted')
# Metadata with floating-point paths is safe and preserves their old implementation exactly.
for kw in ({},{'half_weights':True},{'int8_weights':True}):
 a=wnn_cpp.Ensemble([w],[1.],**kw);b=wnn_cpp.Ensemble([sparse],[1.],**kw)
 assert np.array_equal(a.run(x[:300]),b.run(x[:300]))
report['invalid_metadata_rejected']=True
(D/f'sparse-audit-{level}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
