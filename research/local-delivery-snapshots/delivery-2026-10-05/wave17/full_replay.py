"""Replay compact retrieval students on every row, with sequence-local causal GRU state."""
import os
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_rnn import build_inputs,make_model
from wnn.data import SequenceFile
R=Path(os.environ['WUNDER_RESEARCH_ROOT']);D=R/'research/iterations/2026-10-05-wave17'

def main():
 torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=True
 ar=np.load(D/'student-artifacts.npz');mean=torch.tensor(ar['mean'],device='cuda');std=torch.tensor(ar['std'],device='cuda')
 nets={};pca={};scale={}
 for seed in (3,4):
  pca[seed]=torch.tensor(ar[f'pca_s{seed}'],device='cuda');scale[seed]=torch.tensor(ar[f'pcascale_s{seed}'],device='cuda')
  for mode in ('direct','pure'):
   model=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,2)).cuda().eval()
   model.load_state_dict({k:torch.tensor(ar[f'{mode}_s{seed}_{k}'],device='cuda') for k in model.state_dict()});nets[f'{mode}_s{seed}']=(seed,model)
 ck=torch.load(R/'artifacts/runs/cmp_fullft_s3_ft/model_ep1.pt',map_location='cpu',weights_only=False)
 model=make_model(ck['args'],ck['n_in'],ck['model']['head.2.weight'].shape[0]).cuda().eval();model.load_state_dict(ck['model'])
 cf=np.load(R/'cache/chain_folds.npz');seen=set(c for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if (s=='valid' and f!=0) or (s=='train' and int(hashlib.sha256(f'half:{g}'.encode()).hexdigest(),16)%2==0))
 hold=set(np.load(R/'research/iterations/2026-10-05-wave2/complement_hold.npy'));counts={}
 for split in ('valid','train'):
  groups=np.load(R/f'cache/{split}_targets.npz')['row_group'];cm={g:(c,f) for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if s==split}
  ids=[i for i,g in enumerate(groups) if (g in hold if split=='train' else cm[g][1]==0 and cm[g][0] not in seen)];counts[split]=len(ids)
  ref=np.load(R/f'cache/pred_w2_reference_{split}.npy',mmap_mode='r');sf=SequenceFile(R/f'datasets/{split}.parquet')
  paths={n:R/f'cache/pred_w17_{n}_{split}.npy' for n in nets};assert all(not p.exists() for p in paths.values())
  out={n:np.lib.format.open_memmap(str(p)+'.partial',mode='w+',dtype=np.float32,shape=(len(groups),20000,2)) for n,p in paths.items()}
  for a in out.values():a[:]=np.nan
  for j,i in enumerate(ids):
   seq=sf.read(int(groups[i]));raw=torch.tensor(seq.features,device='cuda')[None];hidden=None
   with torch.no_grad():
    inp=build_inputs(raw)
    for start in range(0,20000,2000):
     with torch.autocast('cuda',dtype=torch.float16):
      enc=model.encode((inp[:,start:start+2000]-model.mean)/model.std);state,hidden=model.gru(enc,hidden)
     xx=((state[0].float()-mean)/std).clamp(-8,8)
     emb={s:xx@pca[s]/scale[s] for s in (3,4)}
     for n,(seed,net) in nets.items():out[n][i,start:start+len(xx)]=ref[i,start:start+len(xx)]+.05*net(emb[seed]).cpu().numpy()
   if j%100==0:print(split,j,flush=True)
  for a in out.values():a.flush()
  del out
  for p in paths.values():Path(str(p)+'.partial').rename(p)
 (D/'replay-manifest.json').write_text(json.dumps({'counts':counts,'correction_scale':.05,'causality':'Only current/past GRU states, fixed training-only normalizer/PCA and MLP; hidden resets every sequence. No labels/masks enter inference.','backbone':'cmp_fullft_s3_ft@ep1','scope':'Full sequence replay; reused holdouts. Primary pure pair vs reference and matched direct pair.'},indent=2))
 print('DONE',counts,flush=True)
if __name__=='__main__':main()
