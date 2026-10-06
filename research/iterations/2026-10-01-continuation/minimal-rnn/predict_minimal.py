"""Prepared minGRU fixed-epoch OOF adapter; no forecast execution allocated yet."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
from train_minimal import load_trainer
train_rnn=load_trainer()
from wnn.resources import configure_torch


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()


def main():
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--fold',type=int,required=True,choices=[0,1,2,3]);ap.add_argument('--epoch',type=int,default=1);ap.add_argument('--output-dir',type=Path,default=HERE.parent/'cache');ap.add_argument('--batch',type=int,default=16)
 a=ap.parse_args();t0=time.monotonic();configure_torch()
 outdir=a.output_dir.resolve();campaign=HERE.parent.resolve()
 if not outdir.is_relative_to(campaign):raise ValueError('prediction output must stay inside campaign')
 outdir.mkdir(parents=True,exist_ok=True)
 final=outdir/f'pred_{a.run}_valid.npy'
 for existing in (final,final.with_suffix('.npy.partial'),final.with_suffix('.inputs.json')):
  if existing.exists() or existing.is_symlink():raise FileExistsError(existing)
 path=ROOT/'artifacts/runs'/a.run/f'model_ep{a.epoch}.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
 if int(ck['args']['valid_fold'])!=a.fold:raise ValueError('checkpoint fold differs from requested OOF fold')
 if int(ck.get('epoch',-1))!=a.epoch:raise ValueError('checkpoint epoch differs')
 if ck['args'].get('lookahead'):raise ValueError('noncausal parent cannot enter deployed comparison')
 if ck['args'].get('arch')!='mingru':raise ValueError('minGRU checkpoint required')
 if ck['args'].get('architecture_source_sha256')!=sha(HERE/'minimal_rnn.py'):raise ValueError('architecture source differs from saved checkpoint')
 device=torch.device('cuda');model=train_rnn.make_model(ck['args'],ck['n_in'],ck['model']['head.2.weight'].shape[0]).to(device)
 model.load_state_dict(ck['model'],strict=True)
 for name,key,cast in [('RAW_ONLY','raw_only',bool),('ONLINE_NORM','online_norm',bool),('QUANT','quant_feats',bool),('QUANT_I1','quant_i1',bool),('STATE_RESET','state_reset',lambda z:int(z or 0))]:setattr(train_rnn,name,cast(ck['args'].get(key)))
 groups=np.load(ROOT/'cache/valid_targets.npz')['row_group'];cf=np.load(ROOT/'cache/chain_folds.npz');v=cf['split']=='valid';folds={int(g):int(f) for g,f in zip(cf['row_group'][v],cf['fold'][v])}
 rows=np.array([i for i,g in enumerate(groups) if folds[int(g)]==a.fold],dtype=np.int64)
 loader=DataLoader(train_rnn.Sequences('valid',groups[rows]),batch_size=a.batch,num_workers=2,prefetch_factor=1)
 tmp=final.with_suffix('.npy.partial');out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(groups),20000,2));out[:]=np.nan
 i=0
 for x,_,_ in loader:
  p=train_rnn.predict(model,x,4000,device)
  if not np.isfinite(p).all():raise ValueError('nonfinite predictions')
  out[rows[i:i+len(p)]]=p;i+=len(p)
 if i!=len(rows):raise ValueError('incomplete prediction set')
 out.flush();del out;tmp.replace(final)
 receipt={'run':a.run,'checkpoint':str(path),'checkpoint_sha256':sha(path),'epoch':a.epoch,'fold':a.fold,'rows':rows.tolist(),'row_groups':groups[rows].tolist(),'source_row_group_order_sha256':hashlib.sha256(groups.tobytes()).hexdigest(),'chain_folds_sha256':sha(ROOT/'cache/chain_folds.npz'),'nonselected_rows':'NaN sentinel','prediction_path':str(final),'prediction_sha256':sha(final),'seconds':time.monotonic()-t0,'metrics_computed':0,'script_sha256':sha(Path(__file__))}
 final.with_suffix('.inputs.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ('rows','row_groups')},indent=2))
if __name__=='__main__':main()
