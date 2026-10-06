from pathlib import Path
import sys,json,numpy as np,torch
from train_rnn import make_model,build_inputs
from wnn.data import SequenceFile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave36';kind=sys.argv[1];assert kind in ('full','comp');torch.set_num_threads(2)
x=SequenceFile(R/'datasets/valid.parquet').read(0).features[:2000];inp=build_inputs(torch.tensor(x)[None]);report=[]
for seed in (3,4):
 tag=f'{kind}_s{seed}';name=f'w36_{tag}';ck=torch.load(R/f'artifacts/runs/{name}/model_ep1.pt',weights_only=False,map_location='cpu');m=make_model(ck['args'],211,23).eval();m.load_state_dict(ck['model']);assert ck['args']['family_class_mode']=='off'
 w={k:v.numpy().copy() for k,v in ck['model'].items() if k in ('mean','std') or k.startswith(('inp.','gru.','head.'))}
 w['head.2.weight']=w['head.2.weight'][:2].copy();w['head.2.bias']=w['head.2.bias'][:2].copy();path=D/f'native_{tag}.npz';assert not path.exists();np.savez_compressed(path,**w)
 with torch.no_grad():p,_=m(inp)
 np.savez_compressed(D/f'native_{tag}.probe.npz',x=x,reference=p[0,:,:2].numpy());report.append({'tag':tag,'unused_class_weights_omitted':True})
(D/f'native-export-{kind}.json').write_text(json.dumps(report,indent=2))
