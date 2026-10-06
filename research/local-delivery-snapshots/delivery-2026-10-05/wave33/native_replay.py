import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,sys,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pyarrow.parquet as pq
FEATURE_COLUMNS=[f'i{i}_{g}{j}' for i in range(2) for g,n in (('p',22),('v',22),('dp',4),('dv',4)) for j in range(n)]+[f'a{i}' for i in range(8)]
assert len(FEATURE_COLUMNS)==112 and not {'t0','t1','is_scored','need_prediction'} & set(FEATURE_COLUMNS)
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave33'
TAGS=[f'{m}_s{s}' for m in ('uniform','soft','parent','old') for s in (3,4)];_models=None;_files={}
def worker(task):
 global _models,_files
 split,group=task
 if _models is None:
  sys.path.insert(0,str(R/'research/iterations/2026-10-05-wave26/native-class/build/v3'));import wnn_cpp;assert wnn_cpp.CLASS_HEAD_SUPPORTED
  _models={tag:wnn_cpp.Ensemble([{k:np.ascontiguousarray(v,np.float32) for k,v in np.load(D/f'native_{tag}.npz').items()}],[1.],dynamic_int8=True) for tag in TAGS}
 if split not in _files:_files[split]=pq.ParquetFile(R/f'datasets/{split}.parquet')
 pf=_files[split];t=pf.read_row_group(int(group),columns=FEATURE_COLUMNS,use_threads=False);x=np.ascontiguousarray(np.column_stack([c.to_numpy() for c in t.columns]),np.float32)
 return group,{tag:m.run(x) for tag,m in _models.items()}
def main():
 sys.path.insert(0,str(R/'research/iterations/2026-10-05-wave26/native-class/build/v3'));import wnn_cpp;assert wnn_cpp.CLASS_HEAD_SUPPORTED
 input_prefix=KIND+"-" if "KIND" in globals() else ""
 input_checks=[]
 for split in ('valid','train'):
  p=np.load(D/f'{input_prefix}input-{split}.npz')
  pf=pq.ParquetFile(R/f'datasets/{split}.parquet');tab=pf.read_row_group(int(p['row_group']),columns=FEATURE_COLUMNS,use_threads=False)
  raw=np.ascontiguousarray(np.column_stack([c.to_numpy() for c in tab.columns]),np.float32)
  assert np.array_equal(raw[:len(p['x'])],p['x']),('feature schema mismatch',split)
  input_checks.append({'split':split,'row_group':int(p['row_group']),'feature_names_exact':True,'rows':len(p['x']),'bytes_equal_sequencefile':True})
 (D/f'native-input-parity{"-"+KIND if "KIND" in globals() else ""}.json').write_text(json.dumps(input_checks,indent=2))
 parity=[]
 for tag in TAGS:
  w={k:np.ascontiguousarray(v,np.float32) for k,v in np.load(D/f'native_{tag}.npz').items()};z=np.load(D/f'native_{tag}.probe.npz');x=z['x'];ref=z['reference']
  for quant in (False,True):
   m=wnn_cpp.Ensemble([w],[1.],dynamic_int8=quant);got=m.run(x);err=float(abs(got-ref).max());corr=[float(np.corrcoef(got[:,j],ref[:,j])[0,1]) for j in (0,1)]
   assert min(corr)>(.999 if quant else .9999999)
   if not quant:assert err<.001
   assert np.array_equal(got,m.run(x));parity.append({'tag':tag,'int8':quant,'max_abs':err,'correlation':corr})
 for tag in TAGS:
  w={k:np.ascontiguousarray(v,np.float32) for k,v in np.load(D/f'native_{tag}.npz').items()};z=np.load(D/f'native_{tag}.train.probe.npz')
  for quant in (False,True):
   got=wnn_cpp.Ensemble([w],[1.],dynamic_int8=quant).run(z['x']);ref=z['reference'];err=float(abs(got-ref).max());corr=[float(np.corrcoef(got[:,j],ref[:,j])[0,1]) for j in (0,1)]
   assert min(corr)>(.999 if quant else .9999999),(tag,quant,err,corr)
   if not quant:assert err<.001,(tag,err)
   parity.append({'tag':tag,'source':'train','int8':quant,'max_abs':err,'correlation':corr})
 cf=np.load(R/'cache/chain_folds.npz');seen={c for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if (s=='valid' and f!=0) or (s=='train' and int(hashlib.sha256(f'half:{g}'.encode()).hexdigest(),16)%2==0)}
 hold=set(np.load(R/'research/iterations/2026-10-05-wave2/complement_hold.npy').tolist())
 for split in (('train',) if '--train-only' in sys.argv else ('valid','train')):
  groups=np.load(R/f'cache/{split}_targets.npz')['row_group'];pos={int(g):i for i,g in enumerate(groups)};cmap={g:(c,f) for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if s==split}
  selected=[g for g in groups if (cmap[g][1]==0 and cmap[g][0] not in seen)] if split=='valid' else [g for g in groups if g in hold]
  paths={tag:R/f'cache/pred_w33native_{tag}_{split}.npy' for tag in TAGS};assert all(not p.exists() for p in paths.values())
  out={tag:np.lib.format.open_memmap(str(p)+'.partial',mode='w+',dtype=np.float32,shape=(len(groups),20000,2)) for tag,p in paths.items()}
  for a in out.values():a[:]=np.nan
  with ProcessPoolExecutor(max_workers=8) as pool:
   for j,(g,values) in enumerate(pool.map(worker,[(split,g) for g in selected],chunksize=1)):
    for tag,p in values.items():assert np.isfinite(p).all();out[tag][pos[g]]=p
    if j%50==0:print(split,j,'/',len(selected),flush=True)
  for a in out.values():a.flush()
  del out
  for p in paths.values():Path(str(p)+'.partial').rename(p)
 (D/'native-student-parity.json').write_text(json.dumps(parity,indent=2))
if __name__=='__main__':main()
