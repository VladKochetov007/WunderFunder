"""Fresh ZIP synthetic stream checks; no competition rows, targets, or timing."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse,hashlib,importlib,io,json,os,sys,tempfile,zipfile
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path,required=True);ap.add_argument('--level',choices=['v2','avx2','v3'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
a.zip=a.zip.resolve();a.output=a.output.resolve();assert not a.output.exists()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class NoTorch:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch runtime forbidden')
sys.meta_path.insert(0,NoTorch())
with tempfile.TemporaryDirectory(prefix='q3-synthetic-') as tmp:
 d=Path(tmp)/'package';cwd=Path(tmp)/'empty';d.mkdir();cwd.mkdir();os.chdir(cwd)
 with zipfile.ZipFile(a.zip) as z:
  assert z.testzip() is None and 'solution.py' in z.namelist();assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(d)
 flags={'v2':'','avx2':'avx avx2 fma','v3':'avx avx2 bmi1 bmi2 f16c fma abm movbe xsave'}[a.level]
 original_open=open
 def forced_open(path,*args,**kwargs):return io.StringIO('flags : '+flags+'\n') if str(path)=='/proc/cpuinfo' else original_open(path,*args,**kwargs)
 sys.path.insert(0,str(d))
 with patch('builtins.open',forced_open):solution=importlib.import_module('solution')
 assert solution.LEVEL==a.level
 w=[]
 for member in json.loads((d/'members.json').read_text()):
  assert member['coef']==1/3
  with np.load(d/member['weights'],allow_pickle=False) as npz:w.append({k:np.ascontiguousarray(npz[k]) for k in npz.files})
 assert [x['inp.weight'].shape[0] for x in w]==[256,128,128]
 rng=np.random.default_rng(81336);x=rng.normal(0,1,(1024,112)).astype(np.float32)
 x[:2]=0;x[2]=-0.;x[10:22]=x[9];x[50,48:52]=[-5.19,-5.189999,5.189999,5.19];x[300:320]=1
 backends=['scalar']+(['avx2'] if solution.wnn_cpp.AVX2_QUANT_SUPPORTED else [])+(['vnni'] if solution.wnn_cpp.VNNI_SUPPORTED else [])
 ys={};errors={}
 for backend in backends:
  bad=[dict(v) for v in w];bad[0]['gru.weight_ih_l0']=bad[0]['gru.weight_ih_l0'].copy();bad[0]['gru.weight_ih_l0'][0]=0;bad[0]['gru.weight_ih_l0'][0,0]=np.nextafter(np.float32(0),np.float32(1))
  rejected=False
  try:solution.wnn_cpp.Ensemble(bad,[1/3]*3,dynamic_int8=True,quant_backend=backend)
  except ValueError:rejected=True
  assert rejected,'subnormal weight scale must reject consistently'
  e=solution.wnn_cpp.Ensemble(w,[1/3]*3,fast_activations=True,dynamic_int8=True,quant_backend=backend)
  y=e.run(x);assert y.shape==(len(x),2) and np.isfinite(y).all();ys[backend]=y
  np.testing.assert_array_equal(e.run(x),y)
  e.reset();np.testing.assert_array_equal(np.stack([e.step(v) for v in x]),y)
  np.testing.assert_array_equal(e.run(x[:301]),y[:301]);altered=x.copy();altered[301:]=.125;np.testing.assert_array_equal(e.run(altered)[:301],y[:301])
  np.testing.assert_array_equal(y,ys['scalar'])
  errors[backend]={'max_abs_vs_scalar':float(np.max(np.abs(y-ys['scalar']))),'bitexact':True}
 model=solution.PredictionModel();assert model.member_count==3
 def run(rows,seq,alternate=False):
  out=np.zeros((len(rows),2),np.float32)
  for i,row in enumerate(rows):
   need=i>=99 and (not alternate or i%3!=0);v=model.predict(SimpleNamespace(seq_ix=seq,step_in_seq=i,state=row,need_prediction=need))
   if need:assert isinstance(v,np.ndarray) and v.dtype==np.float32 and v.shape==(2,) and np.isfinite(v).all();out[i]=v
   else:assert v is None
  return out
 ref=ys['scalar'].copy();ref[:99]=0
 np.testing.assert_array_equal(run(x,1),ref);run(x[:113],2);np.testing.assert_array_equal(run(x,1),ref)
 alt=ref.copy();alt[np.arange(len(x))%3==0]=0;np.testing.assert_array_equal(run(x,3,True),alt)
 np.testing.assert_array_equal(run(x[:301],4),ref[:301])
 value=model.predict(SimpleNamespace(seq_ix=5,step_in_seq=0,state=x[0].astype(np.float64),need_prediction=True));assert value.shape==(2,) and np.isfinite(value).all()
 assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
 np.save(a.output.with_suffix('.pred.npy'),ys['scalar']);np.save(a.output.with_suffix('.input.npy'),x)
 result={'status':'PASS','level':a.level,'zip_sha256':sha(a.zip),'native_sha256':sha(solution.wnn_cpp.__file__),'backends':errors,'rows':len(x),'prediction_sha256':hashlib.sha256(ys['scalar'].tobytes()).hexdigest(),'reset_prefix_stream':True,'masked_rows_still_update':True,'warmup_none':True,'two_finite_float32':True,'fresh_empty_cwd':True,'torch_absent':True,'competition_rows_read':False,'targets_read':False,'WP_queries':0}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
