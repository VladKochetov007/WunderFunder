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
a,b=parts;result={'scope':'No targets or validation rows read. 64 fitting train chains and64 disjoint probe train chains, first sequence perchain, seed20261006. Full 20000 rows persequence. Feasibility only, not model quality.','scalar':[],'joint':[]}
for j,col in enumerate(cols):
 keys,counts=np.unique(a[:,j],return_counts=True)
 result['scalar'].append({'column':col,'unique_fit':len(keys),'probe_unseen_fraction':float((~np.isin(b[:,j],keys)).mean()),'fit_singleton_row_fraction':float(counts[counts==1].sum()/len(a))})
for lo,name in ((0,'i0_price22'),(22,'i1_price22')):
 aa=np.ascontiguousarray(a[:,lo:lo+22]).view(np.dtype((np.void,88))).ravel();bb=np.ascontiguousarray(b[:,lo:lo+22]).view(np.dtype((np.void,88))).ravel()
 keys,counts=np.unique(aa,return_counts=True)
 result['joint'].append({'name':name,'unique_fit':len(keys),'probe_unseen_fraction':float((~np.isin(bb,keys)).mean()),'fit_singleton_row_fraction':float(counts[counts==1].sum()/len(a))})
(D/'codebook-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
