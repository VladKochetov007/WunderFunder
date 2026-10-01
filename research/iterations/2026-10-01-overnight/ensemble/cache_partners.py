"""Generate fixed own-fold H2 forecasts on legal head-training rows, no WP/target reads."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import train_rnn as base
from wnn.data import SequenceFile
from support import sha


class Inputs(Dataset):
 def __init__(self,groups):self.groups=list(groups);self.reader=None
 def __len__(self):return len(self.groups)
 def __getitem__(self,index):
  if self.reader is None:self.reader=SequenceFile(ROOT/'datasets/valid.parquet')
  return torch.from_numpy(self.reader.read(int(self.groups[index]),targets=False,mask=False).features)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--fold',type=int,required=True,choices=[-1,0,1]);a=ap.parse_args();started=time.monotonic();base.configure_torch()
 outdir=HERE/'cache';outdir.mkdir(parents=True,exist_ok=True);outpath=outdir/f'pred_{a.run}_training.npy';tmp=outpath.with_suffix('.npy.partial');receipt=outpath.with_suffix('.json')
 if any(p.exists() for p in (outpath,tmp,receipt)):raise FileExistsError(outpath)
 checkpoint=ROOT/'artifacts/runs'/a.run/'model_ep1.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if ck.get('epoch')!=1 or ck['args']['valid_fold']!=a.fold:raise ValueError('partner checkpoint must match epoch1 and own fold')
 if ck['args'].get('lookahead') or ck['args'].get('transition_features'):raise ValueError('requires unmodified causal plain control partner')
 if ck['args']['train_source']!='valid' or ck['args']['epochs']!=2:raise ValueError('partner must be H2 control')
 groups=np.load(ROOT/'cache/valid_targets.npz')['row_group'];cf=np.load(ROOT/'cache/chain_folds.npz');v=cf['split']=='valid';fold={int(g):int(f) for g,f in zip(cf['row_group'][v],cf['fold'][v])}
 positions=np.array([i for i,g in enumerate(groups) if a.fold<0 or fold[int(g)]!=a.fold],np.int64)
 model=base.make_model(ck['args'],ck['n_in'],ck['model']['head.2.weight'].shape[0]).cuda().eval();model.load_state_dict(ck['model'],strict=True)
 out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(groups),20000,2));out[:]=np.nan
 loader=DataLoader(Inputs(groups[positions]),batch_size=16,num_workers=2,prefetch_factor=1);i=0
 for x in loader:
  p=base.predict(model,x,4000,'cuda')
  if not np.isfinite(p).all():raise ValueError('nonfinite partner prediction')
  out[positions[i:i+len(p)]]=p;i+=len(p)
 if i!=len(positions):raise ValueError('incomplete partner cache')
 out.flush();del out;tmp.replace(outpath)
 meta={'schema':'enshead01-cache-v1','run':a.run,'role':'eligible_training_rows','fold':a.fold,'epoch':1,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'prediction_path':str(outpath),'prediction_sha256':sha(outpath),'rows':positions.tolist(),'row_groups':groups[positions].tolist(),'source_row_group_order_sha256':hashlib.sha256(groups.tobytes()).hexdigest(),'chain_folds_sha256':sha(ROOT/'cache/chain_folds.npz'),'source_code_sha256':sha(__file__),'target_columns_read':False,'mask_read':False,'WP_queries':0,'seconds':time.monotonic()-started}
 receipt.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k not in ('rows','row_groups')},indent=2))
if __name__=='__main__':main()
