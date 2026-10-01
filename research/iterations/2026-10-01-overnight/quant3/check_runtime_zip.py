"""After root freeze only: existing raw fixture correctness, never targets or WP."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse,hashlib,importlib,io,json,os,sys,tempfile,zipfile
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path,required=True);ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--fp16-reference',type=Path,required=True);ap.add_argument('--level',choices=['v2','avx2','v3'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
for key in ['zip','raw','reference','fp16_reference','output']:setattr(a,key,getattr(a,key).resolve())
assert not a.output.exists()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def error_report(candidate,reference):
 reports=[]
 for label,index in [('pooled',slice(None))]+[(f'stream{i}',slice(i,i+1)) for i in range(len(candidate))]:
  c=candidate[index,99:].reshape(-1,2).astype(np.float64);r=reference[index,99:].reshape(-1,2).astype(np.float64);assert np.isfinite(c).all() and np.isfinite(r).all()
  for k in range(2):
   d=c[:,k]-r[:,k];rmse=float(np.sqrt(np.mean(d*d)));std=float(np.std(r[:,k]));v={'scope':label,'target':k,'rmse':rmse,'reference_population_std':std,'relative_rmse':rmse/max(std,1e-8),'p99_abs':float(np.quantile(np.abs(d),.99)),'max_abs':float(np.max(np.abs(d)))}
   assert v['relative_rmse']<=.03 and v['p99_abs']<=.05 and v['max_abs']<=.5,v;reports.append(v)
 return reports
assert sha(a.raw)=='1e90c52f1332950b70d573618adad29324fb6705675e74cb75c7ca6efda993b4'
class NoTorch:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch runtime forbidden')
sys.meta_path.insert(0,NoTorch())
with tempfile.TemporaryDirectory(prefix='q3-runtime-check-') as tmp:
 d=Path(tmp)/'package';d.mkdir();cwd=Path(tmp)/'empty';cwd.mkdir();os.chdir(cwd)
 with zipfile.ZipFile(a.zip) as z:assert z.testzip() is None;z.extractall(d)
 flags={'v2':'','avx2':'avx avx2 fma','v3':'avx avx2 bmi1 bmi2 f16c fma abm movbe xsave'}[a.level];original_open=open
 def forced_open(path,*args,**kwargs):return io.StringIO('flags : '+flags+'\n') if str(path)=='/proc/cpuinfo' else original_open(path,*args,**kwargs)
 sys.path.insert(0,str(d))
 with patch('builtins.open',forced_open):solution=importlib.import_module('solution')
 assert solution.LEVEL==a.level;model=solution.PredictionModel();assert model.member_count==3 and model.quant_backend==2
 x=np.load(a.raw,mmap_mode='r');ref=np.load(a.reference,mmap_mode='r');assert x.shape==(5,20000,112) and ref.shape==(5,20000,2)
 n=20000 if a.level=='v3' else 512;out=[]
 for seq,row in enumerate(x):
  y=np.zeros((n,2),np.float32)
  for j,features in enumerate(row[:n]):
   value=model.predict(SimpleNamespace(seq_ix=seq,step_in_seq=j,need_prediction=j>=99,state=features))
   if j<99:assert value is None
   else:assert value.dtype==np.float32 and value.shape==(2,) and np.isfinite(value).all();y[j]=value
  out.append(y)
 out=np.stack(out);expected=np.asarray(ref[:,:n]).copy();expected[:,:99]=0
 if a.level=='v3':np.testing.assert_array_equal(out,expected)
 errors=error_report(out,expected)
 w=[]
 for member in json.loads((d/'members.json').read_text()):
  with np.load(d/member['weights'],allow_pickle=False) as z:w.append({k:np.ascontiguousarray(z[k]) for k in z.files})
 backend_hashes={};baseline=None
 for backend in ['scalar','avx2','vnni']:
  e=solution.wnn_cpp.Ensemble(w,[1/3]*3,fast_activations=True,dynamic_int8=True,quant_backend=backend);z=np.stack([e.run(np.ascontiguousarray(v[:512])) for v in x])
  if baseline is None:baseline=z
  else:np.testing.assert_array_equal(z,baseline)
  backend_hashes[backend]=hashlib.sha256(z.tobytes()).hexdigest()
 e.reset();stream=np.stack([e.step(row) for row in x[0,:512]]);np.testing.assert_array_equal(stream,baseline[0]);np.testing.assert_array_equal(e.run(np.ascontiguousarray(x[0,:301])),baseline[0,:301])
 alternate=[]
 for j,row in enumerate(x[0,:512]):
  value=model.predict(SimpleNamespace(seq_ix=99,step_in_seq=j,state=row,need_prediction=j>=99 and j%3!=0))
  if j<99 or j%3==0:assert value is None
  else:np.testing.assert_array_equal(value,baseline[0,j])
 fp16_errors=error_report(np.asarray(ref),np.load(a.fp16_reference,mmap_mode='r')) if a.level=='v3' else None
 assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
 r={'status':'PASS','level':a.level,'rows':5*n,'zip_sha256':sha(a.zip),'raw_sha256':sha(a.raw),'reference_sha256':sha(a.reference),'native_sha256':sha(solution.wnn_cpp.__file__),'reference_bitexact':out.tobytes()==expected.tobytes(),'reference_errors':errors,'backend_compatibility_rows':2560,'within_engine_backend_hashes':backend_hashes,'fp16_same_composition_errors':fp16_errors,'reset_prefix_stream_masks':True,'torch_absent':True,'quality_runtime_qualification':'v3/VNNI only if later gates pass; portable ISA WP/runtime unqualified','targets_read':False,'WP_queries':0}
 a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':'PASS','level':a.level,'rows':5*n,'reference_bitexact':r['reference_bitexact']}))
