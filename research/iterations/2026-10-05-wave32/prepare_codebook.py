"""Training-feature-only screen of exact categorical price code lookup."""
from pathlib import Path
import json
import numpy as np
from wnn.data import SequenceFile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave32'
cf=np.load(R/'cache/chain_folds.npz');gchain={int(g):int(c) for s,g,c in zip(cf['split'],cf['row_group'],cf['chain']) if s=='train'}
chains=sorted(set(gchain.values()));rng=np.random.default_rng(20261006);chosen=rng.permutation(chains)[:128];assert len(chosen)==128
pick={c:min(g for g,cc in gchain.items() if cc==c) for c in chosen}
sf=SequenceFile(R/'datasets/train.parquet');cols=list(range(22))+list(range(52,74))
parts=[]
for k,chunk in enumerate((chosen[:64],chosen[64:])):
 a=np.concatenate([sf.read(pick[c]).features[:,cols] for c in chunk]).astype('float32')
 assert np.isfinite(a).all();parts.append(a);print('loaded',k,a.shape,flush=True)
a,b=parts
result={'scope':'Fixed top4096 exact paired-code keys perinstrument fitted on64 training chains only, checked on64 disjoint training chains. No targets/validation data. Keys use raw float32 bits of coordinates(0,11) and(52,63), signedint64 sort. They are partial price-state identifiers, not a complete physical-price decoder.','pairs':[]}
out={}
for j,k,name in ((0,11,'i0'),(22,33,'i1')):
 def key(x):
  hi=np.ascontiguousarray(x[:,j]).view(np.uint32).astype(np.uint64)
  lo=np.ascontiguousarray(x[:,k]).view(np.uint32).astype(np.uint64)
  return ((hi<<np.uint64(32))|lo).view(np.int64)
 ka,kb=key(a),key(b);keys,counts=np.unique(ka,return_counts=True)
 order=np.argsort(-counts,kind='stable')[:4096];keep=np.sort(keys[order]);out[name]=keep
 result['pairs'].append({'name':name,'unique_fit':len(keys),'retained':len(keep),'fit_coverage':float(np.isin(ka,keep).mean()),'probe_coverage':float(np.isin(kb,keep).mean())})
assert all(v['probe_coverage']>.95 for v in result['pairs']),result
np.savez_compressed(D/'codebook.npz',**out)
(D/'codebook-pair-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
