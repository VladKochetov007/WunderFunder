"""Scoped CPU FTZ/DAZ experiment: real raw inputs only, no fitting or scoring."""
import argparse,ctypes,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
sys.path.insert(0,str(a.package/'v3'));import wnn_cpp
m=json.loads((a.package/'members.json').read_text());weights=[dict(np.load(a.package/z['weights'])) for z in m];coefs=[z['coef'] for z in m]
x=np.load(a.raw,mmap_mode='r')[:2];lib=ctypes.CDLL(str(a.control.resolve()));lib.get_mxcsr.restype=ctypes.c_uint;lib.set_mxcsr.argtypes=[ctypes.c_uint]
old=lib.get_mxcsr();refs={};rows=[]
try:
 for repeat in range(4):
  for flush in ([False,True] if repeat%2==0 else [True,False]):
   lib.set_mxcsr((old|0x8040) if flush else (old&~0x8040))
   e=wnn_cpp.Ensemble(weights,coefs,half_weights=True);e.run(x[0,:500]);out=[]
   c0=time.process_time();t0=time.perf_counter()
   for seq in x:out.append(e.run(seq))
   cpu=time.process_time()-c0;wall=time.perf_counter()-t0;y=np.stack(out)
   if flush in refs:np.testing.assert_array_equal(y,refs[flush])
   else:refs[flush]=y
   z=dict(flush=flush,repeat=repeat,cpu_us=cpu/(x.shape[0]*x.shape[1])*1e6,wall_us=wall/(x.shape[0]*x.shape[1])*1e6)
   rows.append(z);print(json.dumps(z),flush=True)
finally:lib.set_mxcsr(old)
r=dict(initial_mxcsr=old,measurements=rows,max_abs_prediction_difference=float(np.max(np.abs(refs[False]-refs[True]))),
       bitwise_equal=bool(np.array_equal(refs[False],refs[True])),rows=x.shape[0]*x.shape[1],targets_read=False,WP_queries=0,
       native_sha256=hashlib.sha256(Path(wnn_cpp.__file__).read_bytes()).hexdigest(),scope='finite observed raw sample; no universal error bound')
a.output.write_text(json.dumps(r,indent=2)+'\n')
