from pathlib import Path
import json
import shutil
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave16';src=R/'research/iterations/2026-10-05-wave2/blend_twostream'
for tag,linear,pair in [('sparse_twostream',False,False),('sparse_twostream_fastbase',True,False),('runtime_only_sparse_pair',True,True)]:
 dst=D/tag;assert not dst.exists();dst.mkdir()
 shutil.copy2(src/'solution.py',dst/'solution.py')
 for level in ('v2','avx2','v3'):shutil.copytree(D/'native-sparse/build'/level,dst/level)
 members=json.loads((src/'members.json').read_text())
 for m in members:
  name=m['weights']
  if 'twostream' in name:
   w={k:v for k,v in np.load(src/name).items()};w['meta.gru_blocks']=np.array([128,128],np.float32);np.savez_compressed(dst/name,**w)
   if pair:m['coef']=.125
  elif linear and m['name'].startswith('base:'):
   shutil.copy2(R/'research/iterations/2026-10-05-wave13/baseline-linear.npz',dst/name);m.pop('fp32',None);m['name']='base:linear_equivalent_int8'
  else:shutil.copy2(src/name,dst/name)
 if pair:
  m=next(m.copy() for m in members if 'twostream' in m['weights']);m['name']='RUNTIME_ONLY_DUPLICATE_SEED3';members.insert(-1,m)
 (dst/'members.json').write_text(json.dumps(members,indent=2))
 if linear:
  p=dst/'solution.py';s=p.read_text();s=s.replace('Native GRU pair (dynamic int8) + organizer baseline member in fp32 (exact float path; its inputs carry a +16 shift).','Native GRU ensemble with exact sparse GRU blocks and linear organizer baseline in dynamic int8.');p.write_text(s)
print('Built folders only; runtime duplicate is not a submission.',flush=True)
