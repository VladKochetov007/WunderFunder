"""Four fixed late-head fits and two compact shared-backbone OOF passes. No grading."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import train_rnn as base
from wnn.data import SequenceFile
from wnn.outcomes import sequence_class,N_SP_AUX
from wnn.nn import weighted_pooled_corr_loss
from model import LateAtom
from vocab import sha,identity
BASE_SHA={0:'1e0cfb8931e7fb351aff9903aac173887fb7fbdd085c28c90514aa6a0bb13616',1:'d90770ba95c8801bacacd7040ae0c239f8fab8ef322c3c6ea8d31ecb10982c8e'}
SOURCES=['scripts/train_rnn.py','src/wnn/data.py','src/wnn/features.py','src/wnn/nn.py','src/wnn/outcomes.py','src/wnn/resources.py','src/wnn/metric.py','src/wnn/gmetric.py','src/wnn/propensity.py','src/wnn/splits.py']+[str((HERE/n).relative_to(ROOT)) for n in ('run.py','model.py','vocab.py')]
VARIANTS=['candidate','control','base']
def source_hashes():return {p:sha(ROOT/p) for p in SOURCES}
def state_sha(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def save_json(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
def guarded_output(path):
 p=path.resolve();assert p.is_relative_to(HERE) and p!=HERE;return p
def baseload(fold,atoms):
 path=ROOT/f'artifacts/runs/cont_sh256_s3h2_f{fold}/model_ep1.pt';assert sha(path)==BASE_SHA[fold]
 ck=torch.load(path,map_location='cpu',weights_only=False);a=ck['args']
 assert ck['epoch']==1 and ck['n_in']==211 and a['valid_fold']==fold and a['seed']==3
 assert a['hidden']==256 and a['layers']==1 and a['dropout']==0
 assert a.get('arch','gru')=='gru' and a.get('cell','gru')=='gru' and a.get('enc','linear')=='linear'
 assert not any(a.get(k) for k in ('atom_head','lookahead','online_norm','quant_feats','quant_i1','raw_only','heuristics','state_reset','class_heads','prop_input'))
 assert not any((base.ONLINE_NORM,base.RAW_ONLY,base.QUANT,base.QUANT_I1,base.STATE_RESET,base.HEURISTICS,base.LOOKAHEAD)) and base.READOUT is None
 nout=ck['model']['head.2.weight'].shape[0];assert nout==23 and a['aux_struct']>0
 b=base.make_model(a,211,nout);b.load_state_dict(ck['model'],strict=True)
 start=2+2*len([v for v in str(a.get('aux','')).split(',') if v])+N_SP_AUX+3;assert start==17
 model=LateAtom(b,atoms,start);assert state_sha(model.baseline.state_dict())==state_sha(ck['model'])
 return model,dict(path=str(path),sha256=BASE_SHA[fold]),a
def vocabulary(path):
 path=path.resolve();assert path.is_relative_to(HERE)
 rec=json.loads(path.with_name('receipt.json').read_text());assert rec['status']=='COMPLETE_TRAIN_ONLY_VOCAB' and rec['excluded_all_valid_sharing_chains'] and rec['atoms']==64
 assert sha(path)==rec['atoms_sha256'];a=np.load(path,allow_pickle=False);assert a.shape==(64,2) and a.dtype==np.float32
 return torch.from_numpy(a),sha(path)
def partitions(fold):
 with np.load(ROOT/'cache/valid_targets.npz',allow_pickle=False) as z:groups=z['row_group']
 with np.load(ROOT/'cache/chain_folds.npz',allow_pickle=False) as z:
  v=z['split']=='valid';info={int(g):(int(f)%4,int(c),bool(p)) for g,f,c,p in zip(z['row_group'][v],z['fold'][v],z['chain'][v],z['pure'][v])}
  train_chains=set(z['chain'][z['split']=='train'].tolist())
 assert len(groups)==1873 and len(set(groups.tolist()))==1873
 train=sorted(int(g) for g in groups if info[int(g)][0]!=fold)
 held={info[int(g)][1] for g in groups if info[int(g)][0]==fold}
 assert not {info[g][1] for g in train}&held
 positions=[i for i,g in enumerate(groups) if info[int(g)][0]==fold and info[int(g)][2]]
 assert len(positions)==(425,435)[fold] and not {info[int(groups[p])][1] for p in positions}&train_chains
 return groups,info,train,positions
class Rows(Dataset):
 def __init__(self,groups,targets):self.groups=list(map(int,groups));self.targets=targets;self.reader=None
 def __len__(self):return len(self.groups)
 def __getitem__(self,i):
  if self.reader is None:self.reader=SequenceFile(ROOT/'datasets/valid.parquet')
  seq=self.reader.read(self.groups[i],targets=self.targets,mask=True)
  assert seq.features.shape==(20000,112) and seq.scored is not None
  if self.targets:return torch.from_numpy(seq.features),torch.from_numpy(seq.targets),torch.from_numpy(seq.scored.copy()),self.groups[i]
  assert seq.targets is None
  return torch.from_numpy(seq.features),torch.from_numpy(seq.scored.copy()),self.groups[i]
def tracked_inputs(vocab,baseline):
 paths=[vocab,vocab.with_name('receipt.json'),Path(baseline['path']),ROOT/'datasets/valid.parquet',ROOT/'cache/valid_targets.npz',ROOT/'cache/chain_folds.npz']
 return {str(p):identity(p) for p in paths}
def unchanged(tracked,sources):
 for p,i in tracked.items():assert identity(p)==i,('input changed',p)
 for p,h in sources.items():assert sha(ROOT/p)==h,('source changed',p)
def ce_loss(logits,y,valid,classes,atoms):
 # Raw nearest-atom labels, original class3 support tolerance; unsupported main loss retained.
 with torch.no_grad():
  distance=(y.float()[...,None,:]-atoms[None,None,:,:]).abs().sum(-1);dmin,labels=distance.min(-1)
  eligible=valid.float()*y.float().clamp(-2,2).abs().mean(-1)*(classes==3)[:,None]
  weight=eligible*(dmin<1e-3)
 ce=F.cross_entropy(logits.reshape(-1,len(atoms)),labels.reshape(-1),reduction='none').reshape_as(weight)
 return (ce*weight).sum()/weight.sum().clamp_min(1e-8),weight.sum(),eligible.sum()
def train(a):
 started=time.monotonic();out=guarded_output(a.output_dir);out.mkdir(parents=True,exist_ok=False)
 sources=source_hashes();atoms,vhash=vocabulary(a.vocab);model,baseline,baseargs=baseload(a.fold,atoms)
 tracked=tracked_inputs(a.vocab,baseline);groups,info,train_groups,_=partitions(a.fold)
 initial=state_sha(model.residual.state_dict());frozen=state_sha(model.baseline.state_dict())
 torch.manual_seed(3);np.random.seed(3);torch.cuda.manual_seed_all(3);base.configure_torch()
 assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')
 model=model.cuda().train();params=[p for p in model.parameters() if p.requires_grad]
 assert set(n for n,p in model.named_parameters() if p.requires_grad)==set('residual.'+n for n,p in model.residual.named_parameters())
 ds=Rows(np.random.default_rng(3).permutation(train_groups),True)
 loader=DataLoader(ds,batch_size=16,shuffle=True,drop_last=True,num_workers=2,prefetch_factor=1,pin_memory=True)
 opt=torch.optim.AdamW(params,lr=.0005,weight_decay=.01,fused=True)
 sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.0005,total_steps=2*len(loader)*10,pct_start=.05,final_div_factor=100)
 scaler=torch.amp.GradScaler('cuda');history=[];orders=[];attempts=updates=0
 for epoch in range(2):
  oh=hashlib.sha256();losses=[];support=eligible=0.;epoch_updates=0
  for raw,y,mask,rowgroups in loader:
   oh.update(rowgroups.numpy().astype('<i8').tobytes());x=base.build_inputs(raw.cuda(non_blocking=True));y=y.cuda(non_blocking=True);mask=mask.cuda(non_blocking=True)
   assert x.shape[-1]==211;valid=mask&(torch.arange(20000,device='cuda')[None]>=99);classes=sequence_class(y);state=None
   for s in range(0,20000,2000):
    sl=slice(s,s+2000);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):pred,logits,state=model(x[:,sl],state)
    main=weighted_pooled_corr_loss(pred,y[:,sl],valid[:,sl]);ce,w,e=ce_loss(logits,y[:,sl],valid[:,sl],classes,model.residual.atoms)
    loss=main+(0.5 if a.arm=='candidate' else 0.)*ce;assert torch.isfinite(loss)
    scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,1.)
    old_scale=scaler.get_scale();scaler.step(opt);scaler.update();updated=scaler.get_scale()>=old_scale;updates+=int(updated);epoch_updates+=int(updated);attempts+=1;sched.step();state=state.detach()
    losses.append([float(main.detach()),float(ce.detach()),float(loss.detach())]);support+=float(w);eligible+=float(e)
   del x,y,mask,state
  orders.append(oh.hexdigest());assert state_sha(model.baseline.state_dict())==frozen
  assert all(torch.isfinite(v).all() for v in model.state_dict().values())
  hist=dict(epoch=epoch,mean_losses=np.mean(losses,0).tolist(),CE_support_mass=support,CE_class3_mass=eligible,CE_mass_coverage=support/max(eligible,1e-8),gain=float(model.residual.gain.detach()),updates=epoch_updates);history.append(hist);print(json.dumps(hist),flush=True)
  ck=dict(format='late_atom_v1',epoch=epoch,arm=a.arm,fold=a.fold,seed=3,model={k:v.detach().cpu() for k,v in model.state_dict().items()},baseline=baseline,baseline_args=baseargs,vocab_path=str(a.vocab),vocab_sha256=vhash,initial_residual_sha256=initial,sampler_order_sha256=orders.copy(),source_sha256=sources,history=history.copy())
  torch.save(ck,out/f'model_ep{epoch}.pt')
 unchanged(tracked,sources)
 rec=dict(status='COMPLETE_FIXED_EP1',arm=a.arm,fold=a.fold,run=out.name,epoch=1,frozen_base_exact=state_sha(model.baseline.state_dict())==frozen,only_residual_trainable=True,initial_residual_sha256=initial,sampler_order_sha256=orders,checkpoint=dict(path=str(out/'model_ep1.pt'),sha256=sha(out/'model_ep1.pt')),baseline=baseline,vocab_path=str(a.vocab),vocab_sha256=vhash,source_sha256=sources,input_identity=tracked,training_groups=train_groups,chain_disjoint=True,attempts=attempts,updates=updates,skips=attempts-updates,history=history,metrics=0,seconds=time.monotonic()-started,cuda_peak_reserved=torch.cuda.max_memory_reserved())
 save_json(out/'receipt.json',rec);print(json.dumps(rec),flush=True)
def newload(path,arm,fold,vhash,baseline,atoms):
 ck=torch.load(path,map_location='cpu',weights_only=False)
 assert ck['format']=='late_atom_v1' and ck['epoch']==1 and ck['arm']==arm and ck['fold']==fold and ck['seed']==3 and ck['vocab_sha256']==vhash and ck['baseline']==baseline
 model,_,_=baseload(fold,atoms);model.load_state_dict(ck['model'],strict=True)
 original=torch.load(baseline['path'],map_location='cpu',weights_only=False)['model'];assert state_sha(model.baseline.state_dict())==state_sha(original)
 assert torch.equal(model.residual.atoms,atoms) and all(torch.isfinite(v).all() for v in model.state_dict().values())
 return model,ck
@torch.no_grad()
def predict(a):
 started=time.monotonic();out=guarded_output(a.output_dir);out.mkdir(parents=True,exist_ok=True);dest=out/f'fold{a.fold}.npz';rp=out/f'fold{a.fold}.json';partial=dest.with_suffix('.npz.partial')
 assert not any(p.exists() or p.is_symlink() for p in (dest,rp,partial))
 sources=source_hashes();atoms,vhash=vocabulary(a.vocab);_,baseline,_=baseload(a.fold,atoms)
 candidate,cc=newload(a.candidate,'candidate',a.fold,vhash,baseline,atoms);control,ct=newload(a.control,'control',a.fold,vhash,baseline,atoms)
 assert cc['initial_residual_sha256']==ct['initial_residual_sha256'] and cc['sampler_order_sha256']==ct['sampler_order_sha256']
 tracked=tracked_inputs(a.vocab,baseline);tracked.update({str(p):identity(p) for p in (a.candidate,a.control)})
 base.configure_torch();candidate=candidate.cuda().eval();control_head=control.residual.cuda().eval();del control
 groups,info,_,positions=partitions(a.fold);posmap={int(groups[p]):p for p in positions}
 loader=DataLoader(Rows([groups[p] for p in positions],False),batch_size=16,num_workers=2,prefetch_factor=1,pin_memory=True)
 preds=[];meta={k:[] for k in ('cache_position','row_group','step','chain','fold')};seen=[]
 for raw,mask,rowgroups in loader:
  inputs=base.build_inputs(raw.cuda(non_blocking=True));state=None;parts=[]
  for start in range(0,20000,4000):
   with torch.autocast('cuda',dtype=torch.float16):
    feats,p,p3,state=candidate.context(inputs[:,start:start+4000],state)
    cp,_=candidate.residual(feats,p,p3);tp,_=control_head(feats,p,p3)
   parts.append(torch.stack((cp,tp,p),-2).float())
  pp=torch.cat(parts,1).cpu().numpy();assert np.isfinite(pp).all()
  for j,g in enumerate(rowgroups.tolist()):
   pos=posmap[g];steps=np.flatnonzero(mask[j].numpy()&(np.arange(20000)>=99));seen.append(pos);preds.append(pp[j,steps])
   for k,v in zip(meta,(pos,g,steps,info[g][1],a.fold)):meta[k].append(steps if k=='step' else np.full(len(steps),v,np.int64))
 assert seen==positions;arrays={k:np.concatenate(v).astype(np.int64,copy=False) for k,v in meta.items()};pred=np.concatenate(preds)
 assert pred.shape==((761688,852796)[a.fold],3,2) and pred.dtype==np.float32
 with partial.open('xb') as f:np.savez_compressed(f,predictions=pred,variants=np.asarray(VARIANTS),**arrays)
 unchanged(tracked,sources);partial.replace(dest)
 cps={arm:dict(path=str(p),sha256=sha(p)) for arm,p in [('candidate',a.candidate),('control',a.control),('base',Path(baseline['path']))]}
 rec=dict(status='COMPLETE_OOF',fold=a.fold,variants=VARIANTS,checkpoints=cps,source_sha256=sources,outputs={dest.name:dict(sha256=sha(dest),bytes=dest.stat().st_size)},sequences=len(positions),scored_rows=len(pred),vocab_path=str(a.vocab),vocab_sha256=vhash,input_identity=tracked,model_inputs_target_free=True,targets_read=False,shared_frozen_forward=True,metrics=0,engine=dict(batch=16,chunk=4000,AMP='fp16',decoder='FP32',member_output_clipped=False),seconds=time.monotonic()-started)
 save_json(rp,rec);print(json.dumps(rec),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','predict']);ap.add_argument('--fold',type=int,choices=[0,1],required=True);ap.add_argument('--arm',choices=['candidate','control']);ap.add_argument('--vocab',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--candidate',type=Path);ap.add_argument('--control',type=Path);a=ap.parse_args()
 for key in ('vocab','output_dir','candidate','control'):
  if getattr(a,key) is not None:setattr(a,key,getattr(a,key).resolve())
 if a.mode=='train':assert a.arm;train(a)
 else:assert a.candidate and a.control;predict(a)
if __name__=='__main__':main()
