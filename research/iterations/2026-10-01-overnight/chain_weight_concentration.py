"""Describe saved evaluation weight masses, without any new model score."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess
import numpy as np
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=P/'chain-weight-concentration-plan.json';assert sha(plan)=='84815d7fe0d7adbff1a7e85189dd7fb90e3ad37623fb50ac330ae86d23034efb'
src=P/'evaluations/Q3-quant-vs-baseline3-sufficient.npz';assert sha(src)=='27b5c00e3fd1c3b33fc8b476d25fc7ce934158f6b25ed979056cf6f0ad80d50f'
with np.load(src,allow_pickle=False) as z:
 w=z['reference'][...,0].copy();assert np.array_equal(w,z['candidate'][...,0])
assert w.shape==(860,2) and np.isfinite(w).all() and (w>0).all()
code=r'''
from pathlib import Path
import numpy as np,hashlib,json
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=Path('cache/valid_targets.npz');h=sha(f);assert h=='dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955'
with np.load(f,allow_pickle=False) as z:groups=z['row_group']
f=Path('cache/chain_folds.npz');ch=sha(f);assert ch=='315b84f385f5368615c76087288b623f99c33bd680e9ebf7c3465bf5ab8afd35'
with np.load(f,allow_pickle=False) as z:a={k:z[k] for k in ['split','fold','chain','pure','row_group']}
v=a['split']=='valid';info={int(g):(int(f),int(c),bool(p)) for g,f,c,p in zip(a['row_group'][v],a['fold'][v],a['chain'][v],a['pure'][v])};assert len(info)==1873
rows=[]
for fold in (0,1):
 for g in groups:
  f,c,p=info[int(g)]
  if f==fold and p:rows.append([fold,c,int(g)])
assert len(rows)==860
print(json.dumps({'rows':rows,'row_group_archive_sha256':h,'chain_metadata_sha256':ch,'row_group_array_sha256':hashlib.sha256(groups.tobytes()).hexdigest(),'decoded_arrays':['valid_targets.row_group','chain_folds.split','chain_folds.fold','chain_folds.chain','chain_folds.pure','chain_folds.row_group']}))
'''
cmd='cd /home/grinchenkoeg/WunderFunder-creative && taskset -c 37 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -c '+shlex.quote(code)
raw=subprocess.run(['ssh','beleriand',cmd],check=True,capture_output=True,text=True,timeout=30).stdout;meta=json.loads(raw);rows=np.array(meta['rows'],np.int64);assert rows.shape==(860,3)
assert np.array_equal(np.bincount(rows[:,0]),[425,435])
def concentration(mass):
 mass=np.asarray(mass,np.float64);assert mass.ndim==1 and np.isfinite(mass).all() and (mass>=0).all() and mass.sum()>0
 share=mass/mass.sum();h=float(share@share)
 return {'total_mass':float(mass.sum()),'positive_mass_chains':int((mass>0).sum()),'largest_chain_share':float(share.max()),'top10_chain_share':float(np.sort(share)[-10:].sum()),'sum_squared_mass_shares':h,'weight_concentration_equivalent_chains':1/h}
assert abs(concentration(np.ones(17))['weight_concentration_equivalent_chains']-17)<1e-12
assert concentration([0,0,9])['weight_concentration_equivalent_chains']==1
# Splitting one sequence while retaining its chain cannot change chain totals.
np.testing.assert_array_equal(np.bincount([0,0,1],weights=[1,2,4]),[3,4])
assert concentration([3,4])==concentration(np.bincount([0,0,1],weights=[1,2,4]))
results={}
for label,at in [('pooled',np.ones(860,bool)),('fold0',rows[:,0]==0),('fold1',rows[:,0]==1)]:
 keys=rows[at,:2];unique,inv=np.unique(keys,axis=0,return_inverse=True);mass=np.stack([np.bincount(inv,weights=w[at,t],minlength=len(unique)) for t in range(2)],axis=1)
 results[label]={'sequences':int(at.sum()),'chains':len(unique),'targets':[concentration(mass[:,t]) for t in range(2)]}
assert results['pooled']['chains']==250
out={'status':'PASS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),'plan_sha256':sha(plan),'saved_statistics_sha256':sha(src),'metadata':meta,'generated_checks':'uniform masses; single positive mass; splitting rows within unchanged chain','results':results,'new_model_scores':0,'new_WP_readouts':0,'bootstrap_runs':0,'warning':'Descriptive mass concentration, not correlation effective sample size or a standard error; no claim chains are fully independent.'}
f=P/'chain-weight-concentration.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(results,indent=2))
