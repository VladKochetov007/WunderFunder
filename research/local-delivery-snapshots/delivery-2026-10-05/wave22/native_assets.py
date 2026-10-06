from pathlib import Path
import numpy as np
import torch
from train_rnn import make_model,build_inputs
from wnn.data import SequenceFile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave22'
torch.set_num_threads(2)
c=torch.load(R/'artifacts/runs/w22wide_init_s3/model_ep0.pt',map_location='cpu',weights_only=False)
m=make_model(c['args'],c['n_in'],c['model']['head.2.weight'].shape[0]);m.load_state_dict(c['model']);m.eval()
w={k:v.numpy().copy() for k,v in c['model'].items() if k in ('mean','std') or k.startswith(('inp.','gru.','head.'))};w['head.2.weight']=w['head.2.weight'][:2].copy();w['head.2.bias']=w['head.2.bias'][:2].copy()
assert not (D/'wide-init.npz').exists();np.savez_compressed(D/'wide-init.npz',**w)
x=SequenceFile(R/'datasets/valid.parquet').read(0).features
with torch.no_grad():p,_=m(build_inputs(torch.tensor(x)[None]))
np.savez_compressed(D/'wide-probe.npz',x=x,reference=p[0,:,:2].numpy())
print('Native inputs prepared',flush=True)
