"""One fixed H2 anchor stream over original train row map; root allocates execution.

Exact inherited engine: base.predict(model,x,4000,'cuda'), AMP FP16, batch16,
2 loader workers/prefetch1, original complete-sequence causal inputs. No labels/masks.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
from prepare_inputs import verify_inputs,sha,stat,checkstat,rel

def main():
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--fold',type=int,required=True,choices=[0,1]);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--inputs-sha256',required=True);a=ap.parse_args();started=time.monotonic()
 bound=verify_inputs(a.inputs,a.inputs_sha256)
 assert a.run in tuple(f'on_ctl{s}_f{a.fold}' for s in ('128s3','128s4','c500'))
 rec=bound['partners'][a.run];assert rec['fold']==a.fold
 outdir=HERE/'cache';outdir.mkdir(exist_ok=True);assert outdir.resolve().is_relative_to(HERE)
 final=outdir/f'pred_{a.run}_train.npy';tmp=final.with_suffix('.npy.partial');receipt=final.with_suffix('.json')
 assert not any(p.exists() or p.is_symlink() for p in (final,tmp,receipt)),('output collision',str(final))
 # Load metadata only before the GPU/model stage. Never request target/mask NPZ keys.
 with np.load(ROOT/'cache/train_targets.npz',allow_pickle=False) as z:groups=z['row_group']
 assert len(groups)==10606 and len(np.unique(groups))==10606 and 528 not in groups
 assert groups.tolist()==bound['row_maps']['train']['row_groups'] and hashlib.sha256(groups.tobytes()).hexdigest()==bound['row_maps']['train']['row_group_sha256']
 sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
 import torch
 from torch.utils.data import Dataset,DataLoader
 import train_rnn as base
 from wnn.data import SequenceFile
 assert os.environ.get('CUDA_VISIBLE_DEVICES') and torch.cuda.is_available();base.configure_torch()
 class Inputs(Dataset):
  def __init__(self,g):self.groups=list(g);self.reader=None
  def __len__(self):return len(self.groups)
  def __getitem__(self,index):
   if self.reader is None:self.reader=SequenceFile(ROOT/'datasets/train.parquet')
   seq=self.reader.read(int(self.groups[index]),targets=False,mask=False)
   assert seq.targets is None and seq.scored is None and seq.features.shape==(20000,112) and seq.features.dtype==np.float32
   return torch.from_numpy(seq.features)
 checkpoint=ROOT/rec['checkpoint']['path'];assert sha(checkpoint)==rec['checkpoint']['sha256']
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert ck.get('epoch')==1 and ck['args']['valid_fold']==a.fold and ck['n_in']==211
 assert ck['args']['train_source']=='valid' and ck['args']['epochs']==2
 assert not any(ck['args'].get(k) for k in ('lookahead','transition_features','online_norm','raw_only','quant_feats','quant_i1','state_reset','heuristics','prop_input','class_heads','atom_head'))
 assert ck['args'].get('arch','gru')=='gru' and ck['args'].get('enc','linear')=='linear'
 # Plain control partners use unchanged default base globals, as in the prior producer.
 assert not any((base.ONLINE_NORM,base.RAW_ONLY,base.QUANT,base.QUANT_I1,base.STATE_RESET,base.HEURISTICS,base.LOOKAHEAD)) and base.READOUT is None
 model=base.make_model(ck['args'],ck['n_in'],ck['model']['head.2.weight'].shape[0]).cuda().eval();model.load_state_dict(ck['model'],strict=True)
 out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(groups),20000,2));out[:]=np.nan
 loader=DataLoader(Inputs(groups),batch_size=16,num_workers=2,prefetch_factor=1);i=0
 for x in loader:
  p=base.predict(model,x,4000,'cuda')
  assert p.dtype==np.float32 and p.shape==(len(x),20000,2) and np.isfinite(p).all()
  out[i:i+len(p)]=p;i+=len(p)
  if i%256==0 or i==len(groups):print(json.dumps({'run':a.run,'sequences_complete':i,'total':len(groups),'seconds':time.monotonic()-started}),flush=True)
 assert i==len(groups);out.flush();del out
 verify_inputs(a.inputs,a.inputs_sha256);assert sha(checkpoint)==rec['checkpoint']['sha256']
 tmp.replace(final)
 meta={'schema':'ensemble-backbone-train-cache-v1','status':'COMPLETE','run':a.run,'role':'all_train_rows','fold':a.fold,'epoch':1,'checkpoint':str(checkpoint),'checkpoint_sha256':rec['checkpoint']['sha256'],'prediction_path':str(final),'prediction_sha256':sha(final),'prediction_stat':stat(final),'rows':list(range(len(groups))),'row_groups':groups.tolist(),'source_row_group_order_sha256':hashlib.sha256(groups.tobytes()).hexdigest(),'chain_folds_sha256':bound['chain_folds']['sha256'],'source_code_sha256':sha(__file__),'input_receipt':str(a.inputs),'inputs_sha256':a.inputs_sha256,'all_predictions_finite':True,'target_columns_read':False,'mask_read':False,'WP_queries':0,'fits':0,'engine':{'predict':'scripts/train_rnn.py:predict','chunk':4000,'batch':16,'workers':2,'prefetch':1,'dtype_storage':'float32','autocast':'float16','cudnn_benchmark':torch.backends.cudnn.benchmark,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32},'seconds':time.monotonic()-started,'cuda_max_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_max_reserved_bytes':torch.cuda.max_memory_reserved()}
 with receipt.open('x') as f:json.dump(meta,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in meta.items() if k not in ('rows','row_groups')},indent=2),flush=True)
if __name__=='__main__':main()
