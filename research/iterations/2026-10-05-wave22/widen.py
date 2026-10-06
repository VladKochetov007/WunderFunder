"""Function-preserving GRU widening 256 -> 384, with zero-sum outgoing jitter."""
from pathlib import Path
import json
import copy
import numpy as np
import torch
from train_rnn import make_model,build_inputs
from wnn.data import SequenceFile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave22'

def main():
 torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 index=np.r_[np.arange(256),np.arange(128)];count=np.bincount(index,minlength=256)
 def expand_columns(w,indices,counts,rng):
  new=w[:,indices]/counts[indices][None]
  noise_scale=.02*float(w.std())
  for old in np.flatnonzero(counts>1):
   positions=np.flatnonzero(indices==old);assert len(positions)==2
   jitter=rng.normal(0,noise_scale,size=w.shape[0])
   new[:,positions[0]]+=jitter;new[:,positions[1]]-=jitter
  return new.astype('float32')
 sf=SequenceFile(R/'datasets/valid.parquet');raw=[torch.tensor(sf.read(g).features[:2000])[None] for g in (0,5)]
 report=[]
 for seed in (3,4):
  parent=R/f'artifacts/runs/w19distilled_s{seed}/model_ep1.pt';ck=torch.load(parent,map_location='cpu',weights_only=False)
  old=make_model(ck['args'],ck['n_in'],ck['model']['head.2.weight'].shape[0]);old.load_state_dict(ck['model']);old.eval()
  args=copy.deepcopy(ck['args']);args['hidden']=384;args['name']=f'w22wide_init_s{seed}';args['widening_parent']=str(parent)
  wide=make_model(args,ck['n_in'],ck['model']['head.2.weight'].shape[0]);wide.eval()
  src={k:v.numpy() for k,v in ck['model'].items()};dst={};rng=np.random.default_rng(20261005+seed)
  dst['mean']=src['mean'].copy();dst['std']=src['std'].copy();dst['inp.weight']=src['inp.weight'][index].copy();dst['inp.bias']=src['inp.bias'][index].copy()
  gates=np.concatenate([index+g*256 for g in range(3)])
  for layer in range(args['layers']):
   for kind in ('ih','hh'):
    key=f'gru.weight_{kind}_l{layer}';dst[key]=expand_columns(src[key][gates],index,count,rng)
    key=f'gru.bias_{kind}_l{layer}';dst[key]=src[key][gates].copy()
  catindex=np.r_[index,index+256];catcount=np.tile(count,2)
  dst['head.0.weight']=expand_columns(src['head.0.weight'][index],catindex,catcount,rng);dst['head.0.bias']=src['head.0.bias'][index].copy()
  for key in ('head.2','family_head'):
   dst[key+'.weight']=expand_columns(src[key+'.weight'],index,count,rng);dst[key+'.bias']=src[key+'.bias'].copy()
  wide.load_state_dict({k:torch.tensor(v) for k,v in dst.items()})
  errors=[]
  with torch.no_grad():
   for xx in raw:
    inp=build_inputs(xx);a,_=old(inp);b,_=wide(inp)
    error=float((a-b).abs().max());family_error=float((old.family_output-wide.family_output).abs().max());assert max(error,family_error)<5e-5,(seed,error,family_error)
    errors.append({'all_output_max_abs':error,'family_output_max_abs':family_error})
  out=R/f'artifacts/runs/w22wide_init_s{seed}';assert not out.exists();out.mkdir()
  torch.save({'model':wide.state_dict(),'args':args,'n_in':ck['n_in'],'epoch':0},out/'model_ep0.pt')
  report.append({'seed':seed,'parent':str(parent),'hidden':384,'clone_indices':index.tolist(),'outgoing_zero_sum_noise_relative_std':.02,'parity':errors})
 (D/'widening-parity.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()
