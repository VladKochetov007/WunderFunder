from pathlib import Path
import sys,json
from types import SimpleNamespace as Point
import numpy as np
import pyarrow.parquet as pq
folder=Path(sys.argv[1]).resolve();sys.path.insert(0,str(folder));from solution import PredictionModel
import wnn_cpp
R=Path.home()/'WunderFunder-creative'
cols=[f'i{i}_{g}{j}' for i in range(2) for g,n in (('p',22),('v',22),('dp',4),('dv',4)) for j in range(n)]+[f'a{i}' for i in range(8)]
seqs=[]
for split,g in [('valid',0),('train',0)]:
 t=pq.ParquetFile(R/f'datasets/{split}.parquet').read_row_group(g,columns=cols,use_threads=False);seqs.append(np.ascontiguousarray(np.column_stack([c.to_numpy() for c in t.columns]),np.float32))
members=json.loads((folder/'members.json').read_text());assert len(members)==4 and all(m['coef']==.25 for m in members)
m=PredictionModel();assert m.member_count==4;got=[]
for s,x in enumerate(seqs):
 vals=[]
 for i,row in enumerate(x):
  p=m.predict(Point(seq_ix=s,step_in_seq=i,need_prediction=i>=99,state=row))
  if i<99:assert p is None
  else:assert np.asarray(p).shape==(2,) and np.isfinite(p).all();vals.append(np.asarray(p).copy())
 got.append(np.asarray(vals))
for s,x in enumerate(seqs):
 expected=np.zeros_like(got[s])
 for member in members:
  with np.load(folder/member['weights'],allow_pickle=False) as z:w={k:np.ascontiguousarray(z[k],np.float32) for k in z.files}
  expected+=member['coef']*wnn_cpp.Ensemble([w],[1.],fast_activations=True,dynamic_int8=True,quant_backend='auto').run(x)[99:]
 assert np.max(np.abs(got[s]-expected))<2e-6
 # New sequence id on the same live object must reset all four states.
 for i,row in enumerate(x[:2000]):
  p=m.predict(Point(seq_ix=10+s,step_in_seq=i,need_prediction=i>=99,state=row))
  if i>=99:assert np.array_equal(p,got[s][i-99])
report={'rows':sum(map(len,seqs)),'train_and_valid_named_features':True,'finite_shape':True,'warmup_none':True,'four_members_present':True,'member_sum_parity':True,'reset_bit_exact':True,'runtime_benchmark':'omitted per user instruction'}
Path(sys.argv[2]).write_text(json.dumps(report,indent=2));print(json.dumps(report))
