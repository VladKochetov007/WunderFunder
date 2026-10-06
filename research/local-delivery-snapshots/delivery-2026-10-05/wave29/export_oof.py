from pathlib import Path
import json
import sys
import numpy as np
import torch
from train_rnn import make_model,build_inputs
from wnn.data import SequenceFile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave29'
torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
report=[]
sf=SequenceFile(R/'datasets/valid.parquet');seq=sf.read(0);x=seq.features[:2000];inp=build_inputs(torch.tensor(x)[None])
for seed in ('scaled_s3','scaled_s4'):
 name=f'w29_{seed}';ck=torch.load(R/f'artifacts/runs/{name}/model_ep1.pt',map_location='cpu',weights_only=False)
 model=make_model(ck['args'],ck['n_in'],ck['model']['head.2.weight'].shape[0]);model.load_state_dict(ck['model']);model.eval()
 w={k:v.numpy().copy() for k,v in ck['model'].items() if k in ('mean','std') or k.startswith(('inp.','gru.','head.'))}
 w['head.2.weight']=w['head.2.weight'][:2].copy();w['head.2.bias']=w['head.2.bias'][:2].copy()
 path=D/f'native_{seed}.npz';assert not path.exists();np.savez_compressed(path,**w)
 with torch.no_grad():y,_=model(inp)
 np.savez_compressed(D/f'native_{seed}.probe.npz',x=x,reference=y[0,:,:2].numpy())
 report.append({'seed':seed,'checkpoint':name+'@ep1','hidden':w['inp.weight'].shape[0],'head_hidden':w['head.0.weight'].shape[0],'families_discarded_for_inference':True})
(D/'native-export.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
