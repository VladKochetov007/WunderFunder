"""Generated weights/inputs only; four-member native and wrapper smoke, no timing claim."""
import argparse, ctypes, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--native',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
assert not a.output.exists();sys.path.insert(0,str(a.native.resolve()));import wnn_cpp
rng=np.random.default_rng(917)
weights=[]
for H in [256,128,128,128]:
 w={'mean':np.zeros(211,np.float32),'std':np.ones(211,np.float32)}
 for k,shape in {'inp.weight':(H,211),'inp.bias':(H,), 'head.0.weight':(H,2*H),'head.0.bias':(H,), 'head.2.weight':(2,H),'head.2.bias':(2,)}.items():w[k]=rng.normal(0,.02,shape).astype(np.float32)
 for l in range(2):
  for k,shape in {f'gru.weight_ih_l{l}':(3*H,H),f'gru.weight_hh_l{l}':(3*H,H),f'gru.bias_ih_l{l}':(3*H,),f'gru.bias_hh_l{l}':(3*H,)}.items():w[k]=rng.normal(0,.02,shape).astype(np.float32)
 weights.append(w)
x=rng.normal(0,.4,(129,112)).astype(np.float32);x[:3]=0;x[6:9]=x[5]
engine=wnn_cpp.Ensemble(weights,[.25]*4,dynamic_int8=True,quant_backend='auto');assert len(engine)==4
y=engine.run(x);assert np.isfinite(y).all()
parts=[wnn_cpp.Predictor(w,dynamic_int8=True,quant_backend='auto').run(x)for w in weights]
fma=ctypes.CDLL('libm.so.6').fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
expected=np.zeros_like(y)
for part in parts:
 for i in range(len(x)):
  for k in range(2):expected[i,k]=fma(.25,float(part[i,k]),float(expected[i,k]))
assert y.tobytes()==expected.tobytes()
engine.reset();assert np.stack([engine.step(v)for v in x]).tobytes()==y.tobytes()
assert engine.run(x[:37]).tobytes()==y[:37].tobytes()
with tempfile.TemporaryDirectory(prefix='generated-q4-')as td:
 d=Path(td);members=[]
 for i,w in enumerate(weights):
  np.savez(d/f'weights_{i}.npz',**w);members.append({'weights':f'weights_{i}.npz','coef':.25})
 (d/'members.json').write_text(json.dumps(members));spec=importlib.util.spec_from_file_location('generated_q4_solution',P/'solution.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.HERE=d
 model=mod.PredictionModel();assert model.member_count==4
 for seq_id,alternate in [(4,False),(7,True),(4,False)]:
  for i,row in enumerate(x):
   need=i>=99 and(not alternate or i%3!=0);v=model.predict(SimpleNamespace(seq_ix=seq_id,step_in_seq=i,need_prediction=need,state=row))
   if need:assert v.tobytes()==y[i].tobytes()
   else:assert v is None
r={'status':'PASS_GENERATED_FOUR_MEMBERS','rows':len(x),'member_count':4,'coefficients':[.25]*4,'ordered_fmaf_oracle':True,'stream_prefix_reset_mask':True,'native_sha256':hashlib.sha256(Path(wnn_cpp.__file__).read_bytes()).hexdigest(),'competition_data_or_weights_read':False,'WP_queries':0,'benchmark':False}
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
