"""Dynamic INT8 independent-member oracle and reset/prefix/stream checks."""
import argparse,ctypes,hashlib,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
sys.path.insert(0,str(a.native.resolve()));import wnn_cpp
spec=json.loads((a.package/'members.json').read_text());weights=[dict(np.load(a.package/m['weights'])) for m in spec]
fma=ctypes.CDLL('libm.so.6').fmaf
fma.argtypes=[ctypes.c_float,ctypes.c_float,ctypes.c_float];fma.restype=ctypes.c_float
fused=a.native.name in ('v3','avx2')
raw=np.load(a.raw,mmap_mode='r');x=np.ascontiguousarray(raw[0,:1001]);changed=x.copy();changed[431:]=np.float32(.123)
checks=[]
for chosen in ([0],[0,1],[1,2,3],[0,1,2,3],[3,0,2,1]):
 w=[weights[i] for i in chosen];coeff=[1/len(w)]*len(w)
 ens=wnn_cpp.Ensemble(w,coeff,half_weights=True,dynamic_int8=True);got=ens.run(x)
 independent=np.zeros_like(got)
 for wi,c in zip(w,coeff):
  pred=wnn_cpp.Predictor(wi,half_weights=True,dynamic_int8=True).run(x)
  if fused:
   for row in range(len(x)):
    for target in range(2):independent[row,target]=fma(float(np.float32(c)),float(pred[row,target]),float(independent[row,target]))
  else:independent+=np.float32(c)*pred
 np.testing.assert_array_equal(got,independent)
 np.testing.assert_array_equal(ens.run(x[:431]),got[:431])
 np.testing.assert_array_equal(ens.run(changed)[:431],got[:431])
 ens.reset();chunked=np.array([ens.step(row) for part in (x[:17],x[17:302],x[302:]) for row in part])
 np.testing.assert_array_equal(chunked,got)
 ens.run(np.ascontiguousarray(raw[1,:1001]));np.testing.assert_array_equal(ens.run(x),got)
 checks.append({'members':chosen,'rows':len(x),'independent_member_sum_bitexact':True,'prefix_future_reset_stream_bitexact':True})
r={'status':'PASS','checks':checks,'native_sha256':hashlib.sha256(Path(wnn_cpp.__file__).read_bytes()).hexdigest(),'raw_sha256':hashlib.sha256(a.raw.read_bytes()).hexdigest(),'targets_read':False,'WP_queries':0}
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
