"""ENSHEAD-01 model and immutable training-cache helpers (no metric evaluation)."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from wnn.nn import RNNModel


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
 return h.hexdigest()


class ScaleModel(RNNModel):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  for p in self.parameters():p.requires_grad_(False)
  self.ensemble_log_scale=torch.nn.Parameter(torch.zeros(2))
 def load_state_dict(self,state_dict,strict=True,assign=False):
  value=dict(state_dict)
  value.setdefault('ensemble_log_scale',torch.zeros_like(self.ensemble_log_scale))
  return super().load_state_dict(value,strict=strict,assign=assign)
 def forward(self,x,h=None):
  out,state=super().forward(x,h)
  target=out[...,:2].float()*self.ensemble_log_scale.exp()
  return torch.cat([target,out[...,2:].float()],-1),state


def export_state(model):
 state={key:value.detach().clone() for key,value in model.state_dict().items()}
 if isinstance(model,ScaleModel):
  scale=state.pop('ensemble_log_scale').exp()
  state['head.2.weight'][:2]*=scale[:,None]
  state['head.2.bias'][:2]=state['head.2.bias'][:2]*scale
 return state


def scale_values(model):
 if not isinstance(model,ScaleModel):return None
 return {'log_scale':model.ensemble_log_scale.detach().cpu().tolist(),'offset_fixed_zero':[0.0,0.0]}


class PartnerCache:
 def __init__(self,manifest,root,fold):
  self.manifest=Path(manifest).resolve();self.root=Path(root).resolve();self.fold=fold
  self.meta=json.loads(self.manifest.read_text())
  if self.meta['schema']!='enshead01-partners-v1' or self.meta['fold']!=fold:raise ValueError('partner manifest schema/fold mismatch')
  if len(self.meta['partners'])!=3 or len({v['run'] for v in self.meta['partners']})!=3:raise ValueError('requires exactly three distinct frozen partner slots')
  valid=np.load(self.root/'cache/valid_targets.npz');groups=valid['row_group']
  if hashlib.sha256(groups.tobytes()).hexdigest()!=self.meta['source_row_group_order_sha256']:raise ValueError('partner row order mismatch')
  if sha(self.root/'cache/chain_folds.npz')!=self.meta['chain_folds_sha256']:raise ValueError('chain-map hash mismatch')
  cf=np.load(self.root/'cache/chain_folds.npz');v=cf['split']=='valid';fm={int(g):int(f) for g,f in zip(cf['row_group'][v],cf['fold'][v])}
  self.index={int(g):i for i,g in enumerate(groups)};self.allowed={int(g) for g in groups if fold<0 or fm[int(g)]!=fold}
  self.arrays=[]
  for entry in self.meta['partners']:
   if entry['fold']!=fold or entry['epoch']!=1 or entry['role']!='eligible_training_rows':raise ValueError('partner provenance mismatch')
   if entry['source_row_group_order_sha256']!=self.meta['source_row_group_order_sha256'] or entry['chain_folds_sha256']!=self.meta['chain_folds_sha256']:raise ValueError('partner individual source-order/map mismatch')
   positions=np.asarray(entry['rows'],dtype=np.int64)
   if len(np.unique(positions))!=len(positions) or np.any(positions<0) or np.any(positions>=len(groups)) or not np.array_equal(groups[positions],np.asarray(entry['row_groups'])):raise ValueError('partner row position mapping mismatch')
   path=Path(entry['prediction_path']);checkpoint=Path(entry['checkpoint'])
   if not path.resolve().is_relative_to(self.root/'research/iterations/2026-10-01-overnight/ensemble/cache'):raise ValueError('partner cache outside campaign')
   if sha(checkpoint)!=entry['checkpoint_sha256'] or sha(path)!=entry['prediction_sha256']:raise ValueError('partner cache/checkpoint hash mismatch')
   if set(entry['row_groups'])!=self.allowed:raise ValueError('partner training row set mismatch')
   arr=np.load(path,mmap_mode='r')
   if arr.shape!=(len(groups),20000,2) or arr.dtype!=np.float32:raise ValueError('partner array schema')
   self.arrays.append(arr)
 def sum_for_group(self,group):
  if group not in self.allowed:raise ValueError('held-fold group requested during ensemble training')
  pos=self.index[group];parts=[np.asarray(a[pos],dtype=np.float32) for a in self.arrays]
  if not all(np.isfinite(part).all() for part in parts):raise ValueError('nonfinite/unfilled training partner rows')
  return sum(parts)
