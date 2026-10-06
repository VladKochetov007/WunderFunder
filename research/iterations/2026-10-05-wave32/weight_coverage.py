from pathlib import Path
import json,numpy as np
from wnn.data import SequenceFile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave32'
cf=np.load(R/'cache/chain_folds.npz');gc={int(g):int(c) for s,g,c in zip(cf['split'],cf['row_group'],cf['chain']) if s=='train'}
chosen=np.random.default_rng(20261006).permutation(sorted(set(gc.values())))[64:128];pick={c:min(g for g,cc in gc.items() if cc==c) for c in chosen}
keys=np.load(D/'codebook.npz');prop=np.load(R/'cache/propcache_prop_f0_train.npy',mmap_mode='r');classes=np.load(R/'research/artifacts/market-classes-v1.npz')['train'];sf=SequenceFile(R/'datasets/train.parquet')
m=np.zeros((4,5));seqcounts=np.zeros(4,int)
for c in chosen:
 g=pick[c];s=sf.read(g);x=s.features;y=np.clip(s.targets,-2,2);q=np.asarray(prop[g],dtype='float64');w=np.abs(y).mean(-1)*(q*q+.03);w[:99]=0;assert np.isfinite(w).all();known=[]
 for a,b,name in ((0,11,'i0'),(52,63,'i1')):
  hi=np.ascontiguousarray(x[:,a]).view(np.uint32).astype(np.uint64);lo=np.ascontiguousarray(x[:,b]).view(np.uint32).astype(np.uint64);key=((hi<<np.uint64(32))|lo).view(np.int64);known.append(np.isin(key,keys[name]))
 k=int(classes[g]);seqcounts[k]+=1;m[k]+=[w.sum(),w[known[0]].sum(),w[known[1]].sum(),w[known[0]&known[1]].sum(),w[known[0]|known[1]].sum()]
r={'scope':'Training-only diagnostic on64 disjoint probe train chains, not validation quality; fitweights abs(clippedy)*(prop^2+.03),warmup99zero; vocabulary and architecture unchanged.','columns':['total_weight','i0_known','i1_known','both_known','at_least_one_known'],'pooled_coverage':(m.sum(0)[1:]/m.sum(0)[0]).tolist(),'classes':{str(k):{'sequences':int(seqcounts[k]),'weight_share':float(m[k,0]/m[:,0].sum()),'coverage':(m[k,1:]/m[k,0]).tolist()} for k in range(4)}}
(D/'training-weight-coverage.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
