"""Independent generated-weight/row tests. No checkpoints, data, targets or scores."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,ctypes,hashlib,json,sys
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--native',type=Path,required=True);ap.add_argument('--isa',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
assert not a.output.exists()
sys.path.insert(0,str(a.native.resolve()));import wnn_cpp
rng=np.random.default_rng(319247)
def matrix(shape,scale=.05):return np.asarray(rng.normal(0,scale,shape),np.float32)
def weights(h):
    w={'mean':np.zeros(211,np.float32),'std':np.ones(211,np.float32),'inp.weight':matrix((h,211),.01),'inp.bias':matrix(h,.1),'head.0.weight':matrix((128,2*h)),'head.0.bias':matrix(128,.1),'head.2.weight':matrix((2,128)),'head.2.bias':matrix(2,.1)}
    for l in range(2):
        for part in ['ih','hh']:
            w[f'gru.weight_{part}_l{l}']=matrix((3*h,h))
            w[f'gru.bias_{part}_l{l}']=matrix(3*h,.1)
    return w
ws=[weights(h) for h in [256,128,128]]
x=matrix((641,112),.3);x[0]=0;x[1]=-0.;x[99:141]=x[98];x[411:501,0]=x[410,0];x[211:311,11]=x[210,11]
backends=['scalar','avx2','vnni'];ys={};cases=[]
libm=ctypes.CDLL('libm.so.6');fma=libm.fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
for coefficients in ([1/3]*3,[.17,.31,.52]):
    first=None
    for backend in backends:
        e=wnn_cpp.Ensemble(ws,coefficients,dynamic_int8=True,quant_backend=backend)
        y=e.run(x);assert np.isfinite(y).all()
        if first is None:first=y.copy()
        else:np.testing.assert_array_equal(first,y)
        e.reset();stream=np.stack([e.step(r) for r in x]);np.testing.assert_array_equal(y,stream)
        e.run(x[:20]);np.testing.assert_array_equal(e.run(x),y)
        e.run(x[:21]);np.testing.assert_array_equal(e.run(x),y)
        np.testing.assert_array_equal(e.run(x[:211]),y[:211])
        changed=x.copy();changed[211:]=.75;np.testing.assert_array_equal(e.run(changed)[:211],y[:211])
        if backend=='scalar':
            expected=np.zeros_like(y)
            for w,c in zip(ws,coefficients):
                child=wnn_cpp.Predictor(w,dynamic_int8=True,quant_backend=backend).run(x);c=np.float32(c)
                if a.isa=='v2':expected=np.add(expected,np.multiply(c,child,dtype=np.float32),dtype=np.float32)
                else:
                    for row in range(len(x)):
                        for target in range(2):expected[row,target]=fma(c,child[row,target],expected[row,target])
            np.testing.assert_array_equal(y,expected)
        cases.append({'coefs':coefficients,'backend':backend,'rows':len(x),'backend_parity':'bitexact','original_member_order_oracle':backend=='scalar','reset_stream_prefix':'bitexact'})
    ys[str(coefficients)]=first
np.savez(a.output.with_suffix('.npz'),**ys)
result={'status':'PASS','isa':a.isa,'cases':cases,'weights_and_inputs':'locally generated RNG seed319247; two-layer [256,128,128] GRUs','native_sha256':hashlib.sha256(Path(wnn_cpp.__file__).read_bytes()).hexdigest(),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'prediction_sha256':hashlib.sha256(a.output.with_suffix('.npz').read_bytes()).hexdigest(),'empirical_rows':0,'checkpoint_reads':0,'WP_queries':0}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
