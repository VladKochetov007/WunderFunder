from pathlib import Path
import json,numpy as np,torch
from train_rnn import make_model,build_inputs
from wnn.data import SequenceFile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave33';torch.set_num_threads(2)
x=SequenceFile(R/'datasets/valid.parquet').read(0).features[:2000];inp=build_inputs(torch.tensor(x)[None]);report=[]
for mode,prefix in (('uniform','w33_uniform'),('soft','w33_soft'),('parent','w24comp_distilled'),('old','w18compcompress_distilled')):
 for seed in (3,4):
  name=f'{prefix}_s{seed}';ck=torch.load(R/f'artifacts/runs/{name}/model_ep1.pt',weights_only=False,map_location='cpu');m=make_model(ck['args'],ck['n_in'],23).eval();m.load_state_dict(ck['model'])
  w={k:v.numpy().copy() for k,v in ck['model'].items() if k in ('mean','std') or k.startswith(('inp.','gru.','head.','class_res.'))}
  if mode in ('uniform','soft'):
   w['class_gate.weight']=w['head.2.weight'][17:21].copy();w['class_gate.bias']=w['head.2.bias'][17:21].copy();w['meta.class_head_mode']=np.array([0 if mode=='uniform' else 1],dtype=np.float32)
  w['head.2.weight']=w['head.2.weight'][:2].copy();w['head.2.bias']=w['head.2.bias'][:2].copy();tag=f'{mode}_s{seed}'
  path=D/f'native_{tag}.npz';assert not path.exists();np.savez_compressed(path,**w)
  with torch.no_grad():p,_=m(inp)
  np.savez_compressed(D/f'native_{tag}.probe.npz',x=x,reference=p[0,:,:2].numpy());report.append({'model':name+'@ep1','tag':tag})
(D/'native-export.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
