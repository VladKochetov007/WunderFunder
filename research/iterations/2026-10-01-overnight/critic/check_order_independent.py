"""Independent differential and FMA oracle checks for order-only native optimization."""
from pathlib import Path
import argparse,ctypes,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--native',type=Path,required=True);ap.add_argument('--isa',choices=['v2','avx2','v3'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists() or a.output.with_suffix('.npz').exists():raise FileExistsError(a.output)
    sys.path.insert(0,str(a.native.resolve()));import wnn_cpp
    package=HERE/'runtime-probe/fixed_v1';spec=json.loads((package/'members.json').read_text());weights=[dict(np.load(package/m['weights'])) for m in spec]
    raw=np.load(HERE/'runtime-probe/raw.npy',mmap_mode='r')
    rng=np.random.default_rng(732191)
    x=np.concatenate([np.asarray(raw[2,:417]),rng.uniform(-.8,.8,(307,112)).astype(np.float32),np.zeros((71,112),np.float32)])
    # Exercise exact resets before even and odd stream lengths, not only full even20k sequences.
    tail=rng.uniform(-.7,.7,x.shape).astype(np.float32);prefix=313
    libm=ctypes.CDLL('libm.so.6');fma=libm.fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
    cases=[([1,2,3],[1/3]*3),([3,1,2],[.1,.3,.6]),([3,0,2,1],[.125,.275,.35,.25]),([0,0,2],[.17,.21,.62]),([0,1],[.07,.93]),([2],[1.])]
    results=[];arrays={}
    for i,(ids,coeff) in enumerate(cases):
        ws=[weights[j] for j in ids];engine=wnn_cpp.Ensemble(ws,coeff,half_weights=True)
        answer=engine.run(x);arrays[f'case{i}']=answer
        solo=[wnn_cpp.Predictor(w,half_weights=True).run(x) for w in ws];oracle=np.zeros_like(answer)
        for prediction,c in zip(solo,coeff):
            c=np.float32(c)
            if a.isa=='v2':oracle=np.add(oracle,np.multiply(c,prediction,dtype=np.float32),dtype=np.float32)
            else:
                for row in range(len(x)):
                    for target in range(2):oracle[row,target]=fma(c,prediction[row,target],oracle[row,target])
        assert np.array_equal(answer,oracle),(i,'member-sum oracle',np.max(abs(answer-oracle)))
        engine.reset();stream=np.stack([engine.step(row) for row in x]);assert np.array_equal(answer,stream)
        # run() resets after an arbitrary odd/even previous stream, restoring schedule orientation.
        engine.run(x[:18]);assert np.array_equal(engine.run(x),answer)
        engine.run(x[:19]);assert np.array_equal(engine.run(x),answer)
        assert np.array_equal(engine.run(x[:prefix]),answer[:prefix])
        future=x.copy();future[prefix:]=tail[prefix:];assert np.array_equal(engine.run(future)[:prefix],answer[:prefix])
        results.append({'indices':ids,'coefs':coeff,'rows':len(x),'scalar_original_order_sum':'bit_exact','stream_reset_odd_even_prefix_future':'bit_exact'})
    np.savez(a.output.with_suffix('.npz'),**arrays)
    result={'status':'PASS','isa':a.isa,'cases':results,'native_sha256':sha(Path(wnn_cpp.__file__)),'script_sha256':sha(__file__),'prediction_arrays_sha256':sha(a.output.with_suffix('.npz')),'scope':'independent native member/scalar oracle and state invariants; no targets, fits or WP'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
