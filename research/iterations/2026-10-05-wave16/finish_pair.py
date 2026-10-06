from pathlib import Path
import json
import shutil
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave16'
for tag,parent in [('sparse_twostream_pair','sparse_twostream'),('sparse_twostream_pair_fastbase','sparse_twostream_fastbase')]:
 dst=D/tag;assert not dst.exists();shutil.copytree(D/parent,dst,ignore=shutil.ignore_patterns('__pycache__'))
 members=json.loads((dst/'members.json').read_text());tw=next(m for m in members if 'twostream' in m['weights']);tw['coef']=.125
 w={k:v for k,v in np.load(D/'tstream_final_s4.npz').items()};w['meta.gru_blocks']=np.array([128,128],np.float32)
 np.savez_compressed(dst/'weights_twostream_s4.npz',**w)
 members.insert(-1,{'name':'twostream:w2fts_s4ft','weights':'weights_twostream_s4.npz','coef':.125})
 (dst/'members.json').write_text(json.dumps(members,indent=2))
print('Actual final seed3/seed4 pair folders built.',flush=True)
