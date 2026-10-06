from pathlib import Path
import json
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave21'
specs={'teacher_raw':{'w6fullteacher':1.},'teacher_twostream':{'w19teacher':1.},'student_raw':{f'w12compress256_distilled_s{s}':.5 for s in (3,4)},'student_twostream':{f'w19distilled_s{s}':.5 for s in (3,4)},'direct':{f'w12compress256_direct_s{s}':.5 for s in (3,4)}}
def wp(s):
 w,sy,sp,syy,spp,syp=np.moveaxis(s,-1,0)
 return (syp-sy*sp/w)/np.sqrt(np.maximum(syy-sy**2/w,1e-20)*np.maximum(spp-sp**2/w,1e-20))
cf=np.load(R/'cache/chain_folds.npz');seen=set(c for s,c,f in zip(cf['split'],cf['chain'],cf['fold']) if s=='train' or (s=='valid' and f!=0));cm={g:(c,f) for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if s=='valid'}
z=np.load(R/'cache/valid_targets.npz');groups=z['row_group'];y=z['y'];mask=z['mask'];market=z['market_class'];ids=[i for i,g in enumerate(groups) if cm[g][1]==0 and cm[g][0] not in seen]
arr={n:np.load(R/f'cache/pred_{n}_valid.npy',mmap_mode='r') for sp in specs.values() for n in sp};names=list(specs)
mm=np.zeros((len(names),4,3,2,6));counts=np.zeros(4,dtype=int)
for i in ids:
 yy=np.clip(y[i],-2,2).astype('float64');q=mask[i].astype('float64');q[:99]=0;ww=abs(yy)*q[:,None];cls=int(market[i]);counts[cls]+=1
 ps={n:sum(c*np.asarray(arr[k][i],np.float64) for k,c in sp.items()).clip(-2,2) for n,sp in specs.items()}
 for stage,sel in enumerate((slice(99,2000),slice(2000,10000),slice(10000,None))):
  wy=ww[sel];ty=yy[sel]
  for j,n in enumerate(names):
   p=ps[n][sel];mm[j,cls,stage]+=np.stack([wy.sum(0),(wy*ty).sum(0),(wy*p).sum(0),(wy*ty*ty).sum(0),(wy*p*p).sum(0),(wy*ty*p).sum(0)],-1)
r={'counts_by_market':counts.tolist(),'scores_by_market':{n:wp(mm[j].sum(1)).tolist() for j,n in enumerate(names)},'scores_by_time':{n:wp(mm[j].sum(0)).tolist() for j,n in enumerate(names)},'metric_mass_by_market':mm[0].sum(1)[...,0].tolist(),'scope':'Descriptive reused valid only. Market class is ground-truth metadata used for diagnostics, never an inference gate. Stage boundaries fixed at 2000/10000.'}
(D/'strata.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
