from pathlib import Path
import sys
import json
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave21';level=sys.argv[1]
sys.path.insert(0,str(D/'unpacked'/level));import wnn_cpp
results=[]
for seed in (5,6):
 w={k:np.ascontiguousarray(v,np.float32) for k,v in np.load(D/'unpacked'/f'weights_s{seed}.npz').items()};p=np.load(D/f'student_final_s{seed}.probe.npz');x=p['x'];ref=p['reference']
 fp=wnn_cpp.Ensemble([w],[1.],dynamic_int8=False).run(x);err=float(abs(fp-ref).max());assert err<.001,err
 base=None
 for backend in ('scalar','avx2','vnni'):
  model=wnn_cpp.Ensemble([w],[1.],dynamic_int8=True,quant_backend=backend);got=model.run(x)
  corr=[float(np.corrcoef(ref[:,j],got[:,j])[0,1]) for j in (0,1)];assert min(corr)>.999,corr
  assert np.array_equal(got,model.run(x))
  if base is None:base=got
  else:assert np.array_equal(base,got)
  results.append({'seed':seed,'backend':backend,'correlation':corr,'fp32_max_abs':err,'deterministic':True,'backends_bit_exact':True})
(D/f'parity-{level}.json').write_text(json.dumps(results,indent=2));print(level,'PASS',flush=True)
