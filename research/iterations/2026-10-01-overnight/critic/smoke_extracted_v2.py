from pathlib import Path
from types import SimpleNamespace
import argparse,sys,json,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();sys.path.insert(0,str(a.package.resolve()));import solution
m=solution.PredictionModel();rng=np.random.default_rng(718193);x=rng.normal(0,.1,(143,112)).astype(np.float32)
def run(seq,raw):
 out=[]
 for i,row in enumerate(raw):
  answer=m.predict(SimpleNamespace(seq_ix=seq,step_in_seq=i,need_prediction=i>=99,state=row))
  if i<99:assert answer is None
  else:out.append(answer)
 return np.asarray(out)
y=run(1,x);z=run(2,x);assert np.array_equal(y,z) and y.shape==(44,2) and np.isfinite(y).all()
result={'status':'PASS','members':m.member_count,'python':sys.version,'isa':solution.LEVEL,'warmup':'None','reset':'bit_exact','prediction_sha256':hashlib.sha256(y.tobytes()).hexdigest(),'rows_per_stream':len(x),'torch_imported':'torch' in sys.modules,'synthetic_only':True}
assert not result['torch_imported'];a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
