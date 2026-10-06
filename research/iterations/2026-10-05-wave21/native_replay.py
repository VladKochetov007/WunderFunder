import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pyarrow.parquet as pq
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave21';_models=None;_pf=None

def worker(group):
 global _models,_pf
 if _models is None:
  sys.path.insert(0,str(R/'research/iterations/2026-10-05-wave16/native-sparse/build/v3'));import wnn_cpp
  _models={}
  for seed in (3,4):
   w={k:np.ascontiguousarray(v,np.float32) for k,v in np.load(D/f'student_fold_s{seed}.npz').items()}
   _models[seed]=wnn_cpp.Ensemble([w],[1.],dynamic_int8=True)
  _pf=pq.ParquetFile(R/'datasets/valid.parquet')
 tab=_pf.read_row_group(int(group),columns=_pf.schema_arrow.names[4:116],use_threads=False)
 x=np.ascontiguousarray(np.column_stack([c.to_numpy() for c in tab.columns]),dtype=np.float32)
 return group,{seed:model.run(x) for seed,model in _models.items()}

def main():
 sys.path.insert(0,str(R/'research/iterations/2026-10-05-wave16/native-sparse/build/v3'));import wnn_cpp
 parity=[]
 for seed in (3,4):
  w={k:np.ascontiguousarray(v,np.float32) for k,v in np.load(D/f'student_fold_s{seed}.npz').items()};p=np.load(D/f'student_fold_s{seed}.probe.npz');x=p['x'];ref=p['reference']
  for quant in (False,True):
   model=wnn_cpp.Ensemble([w],[1.],dynamic_int8=quant);got=model.run(x)
   error=float(abs(ref-got).max());corr=[float(np.corrcoef(ref[:,j],got[:,j])[0,1]) for j in (0,1)]
   assert min(corr)>(.999 if quant else .9999999),(seed,quant,error,corr)
   if not quant:assert error<.001,error
   assert np.array_equal(got,model.run(x))
   parity.append({'seed':seed,'int8':quant,'max_abs':error,'correlation':corr})
 cf=np.load(R/'cache/chain_folds.npz');seen=set(c for s,c,f in zip(cf['split'],cf['chain'],cf['fold']) if s=='train' or (s=='valid' and f!=0));cm={g:(c,f) for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if s=='valid'}
 groups=np.load(R/'cache/valid_targets.npz')['row_group'];pos={int(g):i for i,g in enumerate(groups)};selected=[g for g in groups if cm[g][1]==0 and cm[g][0] not in seen]
 paths={s:R/f'cache/pred_w21native_distilled_s{s}_valid.npy' for s in (3,4)};assert all(not p.exists() for p in paths.values())
 out={s:np.lib.format.open_memmap(str(p)+'.partial',mode='w+',dtype=np.float32,shape=(len(groups),20000,2)) for s,p in paths.items()}
 for a in out.values():a[:]=np.nan
 with ProcessPoolExecutor(max_workers=8) as pool:
  for j,(g,values) in enumerate(pool.map(worker,selected,chunksize=1)):
   for s,p in values.items():assert np.isfinite(p).all();out[s][pos[g]]=p
   if j%50==0:print('native students',j,flush=True)
 for a in out.values():a.flush()
 del out
 for p in paths.values():Path(str(p)+'.partial').rename(p)
 (D/'native-student-parity.json').write_text(json.dumps(parity,indent=2));print('DONE',parity,flush=True)
if __name__=='__main__':main()
