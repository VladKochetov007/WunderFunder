"""Preserve the original native_order mixed-coefficient counterexample without assertions hiding arrays."""
from pathlib import Path
import argparse,ctypes,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();sys.path.insert(0,str(a.native.resolve()));import wnn_cpp
package=HERE/'runtime-probe/fixed_v1';spec=json.loads((package/'members.json').read_text());weights=[dict(np.load(package/m['weights'])) for m in spec]
raw=np.load(HERE/'runtime-probe/raw.npy',mmap_mode='r');rng=np.random.default_rng(732191);x=np.concatenate([np.asarray(raw[2,:417]),rng.uniform(-.8,.8,(307,112)).astype(np.float32),np.zeros((71,112),np.float32)])
ids=[3,0,2,1];coeff=[.125,.275,.35,.25];ws=[weights[j] for j in ids];engine=wnn_cpp.Ensemble(ws,coeff,half_weights=True);answer=engine.run(x);engine.reset();stream=np.stack([engine.step(row) for row in x]);solo=[wnn_cpp.Predictor(w,half_weights=True).run(x) for w in ws];oracle=np.zeros_like(answer)
libm=ctypes.CDLL('libm.so.6');fma=libm.fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
for prediction,c in zip(solo,coeff):
 for row in range(len(x)):
  for target in range(2):oracle[row,target]=fma(np.float32(c),prediction[row,target],oracle[row,target])
np.savez(a.output.with_suffix('.npz'),answer=answer,stream=stream,oracle=oracle,x=x)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
result={'status':'COUNTEREXAMPLE; original frozen native_order is not generally bit-exact','indices':ids,'coefs':coeff,'rows':len(x),'max_run_vs_original_fma_oracle':float(np.max(abs(answer-oracle))),'max_stream_vs_original_fma_oracle':float(np.max(abs(stream-oracle))),'max_run_vs_stream':float(np.max(abs(answer-stream))),'native_sha256':sha(wnn_cpp.__file__),'script_sha256':sha(__file__),'arrays_sha256':sha(a.output.with_suffix('.npz')),'scope':'raw-only; no fits or WP'}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
