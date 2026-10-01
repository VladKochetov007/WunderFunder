"""Generated-only native scalar oracle, whole model and stream/reset checks."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,sys
import numpy as np
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--level',choices=['v3','avx2','v2'],required=True);a=ap.parse_args()
sys.path.insert(0,str(P/'native/build'/a.level));import wnn_cpp
data=P/'generated';weights=dict(np.load(data/'model.npz',allow_pickle=False))
raw=np.load(data/'raw.npy');expected=np.load(data/'scalar_features.npy')
observed=wnn_cpp.scalar_features(weights,raw);assert np.array_equal(observed,expected)
x=np.load(data/'stream_raw.npy');y=np.load(data/'stream_expected.npy')
fp32=wnn_cpp.Predictor(weights,fast_activations=False,half_weights=False).run(x)
fp32_error=float(np.max(np.abs(fp32-y)));assert fp32_error<=2e-5,fp32_error
engine=wnn_cpp.Predictor(weights,fast_activations=True,half_weights=True)
out=engine.run(x);half_error=float(np.max(np.abs(out-y)));assert half_error<=2e-3,half_error
engine.reset();steps=np.stack([engine.step(row) for row in x]);assert np.array_equal(steps,out)
for stop in [1,2,99,100,317,640]:assert np.array_equal(engine.run(x[:stop]),out[:stop])
future=x.copy();future[100:]*=-13;assert np.array_equal(engine.run(future)[:100],out[:100])
assert np.array_equal(engine.run(x),out)
zero={k:v.copy() for k,v in weights.items()};zero['scalar.projection'].fill(0)
plain={k:v for k,v in zero.items() if not k.startswith('scalar.')}
zero_out=wnn_cpp.Predictor(zero,half_weights=True).run(x)
plain_out=wnn_cpp.Predictor(plain,half_weights=True).run(x);assert np.array_equal(zero_out,plain_out)
ensemble=wnn_cpp.Ensemble([weights,plain,plain],[1/3]*3,half_weights=True)
combined=ensemble.run(x)
fma=ctypes.CDLL(None).fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
oracle=np.zeros_like(combined)
for pred in [out,plain_out,plain_out]:
 for t in range(len(x)):
  for k in range(2):oracle[t,k]=fma(float(pred[t,k]),float(np.float32(1/3)),float(oracle[t,k])) if a.level!='v2' else np.float32(np.float32(pred[t,k]*np.float32(1/3))+oracle[t,k])
assert np.array_equal(combined,oracle)
result={'status':'PASS','level':a.level,'generated_only':True,'competition_inputs_read':False,'native_sha256':hashlib.sha256(Path(wnn_cpp.__file__).read_bytes()).hexdigest(),'scalar_oracle_exact':True,'float32_vs_CPU_torch_max_abs':fp32_error,'serving_vs_CPU_torch_max_abs':half_error,'rows':len(x),'step_reset_prefix_future_exact':True,'zero_scalar_vs_plain_exact':True,'ensemble_original_order_oracle_exact':True,'weights_sha256':hashlib.sha256((data/'model.npz').read_bytes()).hexdigest(),'scalar_source_sha256':hashlib.sha256((P/'scalar_model.py').read_bytes()).hexdigest()}
(P/f'native-check-{a.level}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
