from pathlib import Path
import sys,json,numpy as np,torch
from train_rnn import make_model,build_inputs
from wnn.data import SequenceFile,FEATURE_COLUMNS
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave36';torch.set_num_threads(2)
kind=sys.argv[1];assert kind in ('full','comp')
for split in ('train','valid'):
 x=SequenceFile(R/f'datasets/{split}.parquet').read(0,targets=False,mask=False).features[:2000]
 np.savez_compressed(D/f'{kind}-input-{split}.npz',x=x,row_group=0)
x=np.load(D/f'{kind}-input-train.npz')['x'];inp=build_inputs(torch.tensor(x)[None])
for seed in (3,4):
 tag=f'{kind}_s{seed}';name=f'w36_{tag}';ck=torch.load(R/f'artifacts/runs/{name}/model_ep1.pt',weights_only=False,map_location='cpu');m=make_model(ck['args'],211,23).eval();m.load_state_dict(ck['model'])
 with torch.no_grad():p,_=m(inp)
 np.savez_compressed(D/f'native_{tag}.train.probe.npz',x=x,reference=p[0,:,:2].numpy())
print('Canonical train/valid features and train Torch probes saved',flush=True)
