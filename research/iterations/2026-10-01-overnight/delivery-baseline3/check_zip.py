"""Fresh cwd, extracted final ZIP, forced ISA selection; no torch available to solution."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib,io,json,os,sys,tempfile,time,zipfile
from unittest.mock import patch
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--level',choices=['v2','avx2','v3'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
a.zip=a.zip.resolve();a.raw=a.raw.resolve();a.reference=a.reference.resolve();a.output=a.output.resolve()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoTorch:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch'or fullname.startswith('torch.'):raise AssertionError('Torch runtime forbidden')
sys.meta_path.insert(0,NoTorch());checks={}
with tempfile.TemporaryDirectory(prefix='baseline3-finalzip-')as tmp:
 d=Path(tmp)/'package';cwd=Path(tmp)/'empty-cwd';d.mkdir();cwd.mkdir();os.chdir(cwd)
 with zipfile.ZipFile(a.zip)as z:
  assert z.testzip()is None;names=z.namelist();assert 'solution.py'in names and all(not Path(n).is_absolute()and'..'not in Path(n).parts for n in names);assert not any(n.endswith('.pt')for n in names);z.extractall(d)
 sys.path.insert(0,str(d));original_open=open
 flags={'v2':'','avx2':'avx avx2 fma','v3':'avx avx2 bmi1 bmi2 f16c fma abm movbe xsave'}[a.level]
 def forced_open(path,*args,**kwargs):return io.StringIO('flags : '+flags+'\n')if str(path)=='/proc/cpuinfo'else original_open(path,*args,**kwargs)
 with patch('builtins.open',forced_open):solution=importlib.import_module('solution')
 assert solution.LEVEL==a.level and Path(solution.wnn_cpp.__file__).parent==d/a.level
 model=solution.PredictionModel();assert model.member_count==3;members=json.loads((d/'members.json').read_text());assert [m['coef']for m in members]==[1/3]*3
 raw=np.load(a.raw,mmap_mode='r');ref=np.load(a.reference.with_name(a.reference.name+'-real.npy'));out=[]
 def run(x,seq_ix,alternate=False):
  y=np.zeros((len(x),2),np.float32);point=SimpleNamespace(seq_ix=seq_ix,step_in_seq=0,state=None,need_prediction=False)
  for j,row in enumerate(x):
   point.step_in_seq=j;point.state=row;point.need_prediction=j>=99 and(not alternate or j%3!=0);v=model.predict(point)
   if not point.need_prediction:assert v is None
   else:assert isinstance(v,np.ndarray)and v.shape==(2,)and v.dtype==np.float32 and np.isfinite(v).all();y[j]=v
  return y
 for k,seq in enumerate(raw):
  y=run(seq[:ref.shape[1]],100+k);expected=ref[k].copy();expected[:99]=0;np.testing.assert_array_equal(y,expected);out.append(y)
 checks['archived_native_real_rows_bitexact']=int(ref.shape[0]*ref.shape[1])
 rng=np.random.default_rng(823645);s=rng.normal(0,1,(2048,112)).astype(np.float32);s[:2]=0;s[2]=-0.;s[10:20]=s[9];s[50,48:52]=[-5.19,-5.189999,5.189999,5.19];s[500:510]=1
 sy=run(s,200);sr=np.load(a.reference.with_name(a.reference.name+'-synthetic.npy'));sr[:99]=0;np.testing.assert_array_equal(sy,sr);checks['archived_native_synthetic_rows_bitexact']=len(s)
 x=np.ascontiguousarray(raw[0,:1001]);first=run(x,300);run(np.ascontiguousarray(raw[1,:333]),301);np.testing.assert_array_equal(run(x,300),first);checks['reset_repeat']=True
 pre=run(x[:431],302);np.testing.assert_array_equal(pre,first[:431]);changed=x.copy();changed[431:]=.123;np.testing.assert_array_equal(run(changed,303)[:431],first[:431]);checks['prefix_future_independence']=True
 alt=run(x,304,True);expected=first.copy();expected[np.arange(len(x))%3==0]=0;np.testing.assert_array_equal(alt,expected);checks['need_prediction_false_still_updates_state']=True
 point=SimpleNamespace(seq_ix=305,step_in_seq=0,state=x[0].astype(np.float64),need_prediction=True);v=model.predict(point);assert v.shape==(2,)and np.isfinite(v).all();checks['float64_input_coercion']=True
 assert not any(n=='torch'or n.startswith('torch.')for n in sys.modules);checks['torch_runtime_absent']=True
 r={'status':'PASS','zip_sha256':sha(a.zip),'zip_bytes':a.zip.stat().st_size,'level':a.level,'native_sha256':sha(solution.wnn_cpp.__file__),'solution_sha256':sha(d/'solution.py'),'member_count':model.member_count,'checks':checks,'real_callback_prediction_sha256':hashlib.sha256(np.stack(out).tobytes()).hexdigest(),'fresh_empty_cwd':True,'forced_isa_for_test_only':True,'targets_read':False,'WP_queries':0,'python':sys.version}
 a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
