"""Saved-weight four-decoder counterfactual. No training, targets or metric reads.

Run once per fold under the root resource controller. --self-test is generated CPU only.
Old arithmetic, including autocast matmuls, is identical to AtomRNNModel.forward.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from torch.utils.data import Dataset,DataLoader
import train_rnn as base
from wnn.atom_head import AtomRNNModel
from wnn.data import SequenceFile
from wnn.outcomes import N_SP_AUX
VARIANTS=('old','clip_only','untilt_only','both')
EXPECTED_SHA={0:'6781e2a7f86f9195ee7abf6c3dbce735b1cfafcfbfc21d26219f0c5e065b3e7d',1:'4333c514858c68cb3bf10a2f88f6f4739d7c80f8c907adca3085a49f43516710'}
SOURCE_NAMES=('scripts/train_rnn.py','src/wnn/atom_head.py','src/wnn/nn.py','src/wnn/features.py','src/wnn/data.py','src/wnn/outcomes.py','src/wnn/resources.py')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def identity(p):
 p=Path(p);s=p.stat();return dict(resolved=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)

def decode(pi,g):
 ga=g.abs()
 return (pi @ (g*ga))/(pi @ ga).clamp_min(1e-6)

class VariantsModel(AtomRNNModel):
 def forward(self,x,h=None):
  z=self.encode((x-self.mean)/self.std)
  o,h=self.gru(z,h)
  feats=torch.cat([o,z],dim=-1)
  out=self.head(feats).float()
  logits=self.atom_head(feats).float()
  pi=torch.softmax(logits,dim=-1)
  g=self.atoms
  # These first operations intentionally preserve original source order/dtypes.
  dec=decode(pi,g)
  a,b=self.class_slice
  p3=torch.softmax(out[...,a:b],dim=-1)[...,3:4]
  lam=torch.sigmoid(self.mix_logit)
  reg=out[...,:2]
  old=reg+lam*p3*(dec-reg)
  clipped=g.clamp(-2,2)
  tilt=clipped.abs().mean(-1)
  u=pi/tilt  # positive for all64 frozen atoms; no re-normalization needed in ratio.
  ds=(decode(pi,clipped),decode(u,g),decode(u,clipped))
  return torch.stack([old]+[reg+lam*p3*(d-reg) for d in ds],dim=-2),h

class Rows(Dataset):
 def __init__(self,positions,groups):self.positions=positions;self.groups=groups;self.reader=None
 def __len__(self):return len(self.positions)
 def __getitem__(self,i):
  pos=int(self.positions[i])
  if self.reader is None:self.reader=SequenceFile(ROOT/'datasets/valid.parquet')
  seq=self.reader.read(int(self.groups[pos]),targets=False,mask=True)
  assert seq.targets is None and seq.scored is not None
  assert seq.features.shape==(20000,112) and seq.features.dtype==np.float32
  return torch.from_numpy(seq.features),torch.from_numpy(seq.scored.copy()),pos

@torch.no_grad()
def self_test():
 torch.set_num_threads(1);torch.manual_seed(531)
 atoms=torch.tensor([[-1.,-1.],[1.,-3.],[4.,-2.],[-3.,2.]])
 old=AtomRNNModel(7,8,2,0.,21,(17,21),atoms).eval()
 new=VariantsModel(7,8,2,0.,21,(17,21),atoms).eval();new.load_state_dict(old.state_dict())
 h0=h1=None
 for length in (11,7):
  x=torch.randn(2,length,7);p,h0=old(x,h0);v,h1=new(x,h1)
  assert torch.equal(p[...,:2],v[...,0,:]) and torch.equal(h0,h1)
 # Independent known categorical law: weighted CE learns P*a. Both must undo a
 # and return E[P*abs(clip(g))*clip(g)] / E[P*abs(clip(g))].
 g=atoms.double();c=g.clamp(-2,2);p=torch.tensor([[.1,.2,.3,.4]],dtype=torch.float64)
 a=c.abs().mean(-1);q=p*a;q=q/q.sum(-1,keepdim=True)
 want=torch.stack([sum(p[0,k]*abs(c[k,j])*c[k,j] for k in range(4))/sum(p[0,k]*abs(c[k,j]) for k in range(4)) for j in range(2)])
 got=decode(q/a,c)[0];assert torch.allclose(got,want,atol=1e-14,rtol=0)
 assert not torch.allclose(decode(q,g)[0],want)
 print(json.dumps(dict(status='PASS_GENERATED_CPU_ONLY',checks=['old_forward_and_carried_state_exact','weighted_CE_population_decode_oracle'],source_sha256=sha(__file__),actual_data_read=False,GPU_used=False)))

@torch.no_grad()
def run(a):
 started=time.monotonic();assert a.fold in EXPECTED_SHA and a.checkpoint.is_absolute()
 assert a.checkpoint_sha256==EXPECTED_SHA[a.fold] and sha(a.checkpoint)==a.checkpoint_sha256
 out=a.output_dir.resolve();assert out.is_relative_to(HERE) and out!=HERE
 out.mkdir(parents=True,exist_ok=True)
 dest=out/f'fold{a.fold}.npz';partial=dest.with_suffix('.npz.partial');rp=out/f'fold{a.fold}.json'
 assert not any(p.exists() or p.is_symlink() for p in (dest,partial,rp))
 sources={s:sha(ROOT/s) for s in SOURCE_NAMES};sources[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
 tracked={str(p):identity(p) for p in (a.checkpoint,ROOT/'datasets/valid.parquet',ROOT/'cache/valid_targets.npz',ROOT/'cache/chain_folds.npz')}
 assert os.environ.get('CUDA_VISIBLE_DEVICES') and torch.cuda.is_available();base.configure_torch()
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);args=ck['args']
 assert ck['epoch']==0 and ck['n_in']==211 and args['valid_fold']==a.fold
 assert args['seed']==3 and args['hidden']==128 and args['layers']==2 and args['atom_head']==.5
 assert args['freeze']=='inp,gru' and args['init']==f'ah128_f{a.fold}@ep3'
 assert args.get('arch','gru')=='gru' and args.get('cell','gru')=='gru' and args.get('enc','linear')=='linear'
 assert not any(args.get(k) for k in ('lookahead','quant_feats','quant_i1','online_norm','raw_only','heuristics','state_reset','class_heads','prop_input'))
 assert args['aux_struct']>0 and ck.get('readout') is None
 assert not any((base.ONLINE_NORM,base.RAW_ONLY,base.QUANT,base.QUANT_I1,base.STATE_RESET,base.HEURISTICS,base.LOOKAHEAD)) and base.READOUT is None
 atoms=ck['model']['atoms'];assert atoms.shape==(64,2) and atoms.dtype==torch.float32 and torch.isfinite(atoms).all()
 assert (atoms.clamp(-2,2).abs().mean(-1)>0).all()
 class_at=2+2*len([v for v in str(args.get('aux','')).split(',') if v])+N_SP_AUX+3
 model=VariantsModel(211,128,2,args['dropout'],ck['model']['head.2.weight'].shape[0],(class_at,class_at+4),atoms.clone())
 model.load_state_dict(ck['model'],strict=True);assert torch.equal(model.atoms,atoms)
 atom_sha=hashlib.sha256(atoms.numpy().tobytes()).hexdigest();model=model.cuda().eval()
 with np.load(ROOT/'cache/valid_targets.npz',allow_pickle=False) as z:groups=z['row_group']
 with np.load(ROOT/'cache/chain_folds.npz',allow_pickle=False) as z:
  v=z['split']=='valid';info={int(g):(int(f),int(c),bool(p)) for g,f,c,p in zip(z['row_group'][v],z['fold'][v],z['chain'][v],z['pure'][v])};train_chains=set(z['chain'][z['split']=='train'].tolist())
 assert len(groups)==1873 and len(np.unique(groups))==1873
 positions=[i for i,g in enumerate(groups) if info[int(g)][0]==a.fold and info[int(g)][2]]
 assert len(positions)==(425,435)[a.fold]
 assert not {info[int(groups[i])][1] for i in positions}&train_chains
 loader=DataLoader(Rows(positions,groups),batch_size=16,num_workers=2,prefetch_factor=1)
 preds=[];meta={k:[] for k in ('cache_position','row_group','step','chain','fold')};seen=[]
 for raw,mask,pos in loader:
  inputs=base.build_inputs(raw.cuda(non_blocking=True));state=None;parts=[]
  assert inputs.shape[-1]==211
  for start in range(0,20000,4000):
   with torch.autocast('cuda',dtype=torch.float16):p,state=base.forward_chunk(model,inputs,slice(start,start+4000),state)
   parts.append(p.float())
  pp=torch.cat(parts,1).cpu().numpy();assert pp.shape==(len(raw),20000,4,2) and np.isfinite(pp).all()
  for j,position in enumerate(pos.tolist()):
   steps=np.flatnonzero(mask[j].numpy() & (np.arange(20000)>=99));preds.append(pp[j,steps]);seen.append(position)
   values=(position,int(groups[position]),steps,info[int(groups[position])][1],a.fold)
   for key,value in zip(meta,values):meta[key].append(steps if key=='step' else np.full(len(steps),value,np.int64))
 assert seen==positions
 arrays={k:np.concatenate(v).astype(np.int64,copy=False) for k,v in meta.items()};pred=np.concatenate(preds)
 assert pred.dtype==np.float32 and pred.shape==((761688,852796)[a.fold],4,2)
 with partial.open('xb') as f:np.savez_compressed(f,predictions=pred,variants=np.asarray(VARIANTS),**arrays)
 for p,before in tracked.items():assert identity(p)==before,('input changed',p)
 for p,h in sources.items():assert sha(ROOT/p)==h,('source changed',p)
 assert sha(a.checkpoint)==a.checkpoint_sha256;partial.replace(dest)
 result=dict(status='COMPLETE',fold=a.fold,run=f'ftah128_f{a.fold}h',checkpoint=str(a.checkpoint),checkpoint_sha256=a.checkpoint_sha256,epoch=0,source_sha256=sources,input_identity=tracked,chain_folds_sha256=sha(ROOT/'cache/chain_folds.npz'),row_group_sha256=hashlib.sha256(groups.tobytes()).hexdigest(),variants=list(VARIANTS),sequences=len(positions),scored_rows=len(pred),outputs={dest.name:dict(sha256=sha(dest),bytes=dest.stat().st_size)},atoms_sha256=atom_sha,atoms_range=[float(atoms.min()),float(atoms.max())],model_inputs_target_free=True,targets_read=False,mask_usage='output selection only after complete20k causal inference; step>=99',fits=0,metrics=0,engine=dict(batch=16,chunk=4000,workers=2,AMP='fp16',storage='float32',all_four_decoders_same_autocast_scope=True,untilt='pi/mean(abs(clip(atom,-2,2))); no renormalization',member_output_clipped=False),seconds=time.monotonic()-started,cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_max_reserved_bytes=torch.cuda.max_memory_reserved())
 with rp.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--fold',type=int,choices=[0,1]);ap.add_argument('--checkpoint',type=Path);ap.add_argument('--checkpoint-sha256');ap.add_argument('--output-dir',type=Path);a=ap.parse_args()
 if a.self_test:self_test()
 else:assert a.fold is not None and a.checkpoint and a.checkpoint_sha256 and a.output_dir;run(a)
if __name__=='__main__':main()
