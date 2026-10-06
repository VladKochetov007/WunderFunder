"""Causal student of leave-chain-out retrieval; fit labels never come from evaluation chains."""
import sys
import json
from pathlib import Path

def main():
 import numpy as np
 import torch
 from torch import nn
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 dev='cuda' if torch.cuda.is_available() else 'cpu'
 z=np.load(sys.argv[1]);part=z['partition'];chain=z['chain'];fit=np.flatnonzero(part==0)
 x0=z['x'].astype('float32')[:,:128];mean=x0[fit].mean(0);std=x0[fit].std(0).clip(.05)
 x=torch.tensor(np.clip((x0-mean)/std,-8,8),device=dev)
 y=np.clip(z['y'],-2,2);ref=z['reference'];weight=torch.tensor(np.abs(y)*z['weight'][:,None],device=dev)
 target=torch.tensor(y-ref,device=dev);ids=torch.tensor(fit,device=dev);groups=[fit[chain[fit]==c] for c in np.unique(chain[fit])]
 preds={'reference':ref};artifacts={'mean':mean,'std':std}
 for seed in (3,4):
  rng=np.random.default_rng(seed)
  def bank(size):
   q=(size+len(groups)-1)//len(groups);out=np.concatenate([rng.choice(g,min(q,len(g)),replace=False) for g in groups])
   return rng.choice(out,size,replace=False) if len(out)>size else out
  bank(8192);bank_np=bank(65536);bi=torch.tensor(bank_np,device=dev);bc=torch.tensor(chain[bank_np],device=dev)
  torch.manual_seed(seed);_,_,v=torch.pca_lowrank(x[ids[::4]],q=64,center=False,niter=3)
  e=x@v;scale=e[ids].std(0).clamp_min(.1);e=e/scale;eb=e[bi];bn=eb.square().sum(1);out=[]
  with torch.no_grad():
   for start in range(0,len(x),1024):
    qe=e[start:start+1024];d=(qe.square().sum(1)[:,None]+bn[None]-2*qe@eb.T).clamp_min(0)
    same=torch.tensor(chain[start:start+len(qe)],device=dev)[:,None]==bc[None]
    d.masked_fill_(same,float('inf'));dd,ii=d.topk(128,largest=False,sorted=True);near=bi[ii]
    ww=weight[near]/(dd[...,None]+1.);rr=(ww*target[near]).sum(1)/ww.sum(1).clamp_min(1e-8);out.append(rr)
  ret=torch.cat(out);print('retrieval teacher complete',seed,flush=True)
  assert torch.isfinite(ret).all()
  def train(tgt):
   torch.manual_seed(seed);net=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,2)).to(dev)
   nn.init.zeros_(net[2].weight);nn.init.zeros_(net[2].bias);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.03)
   gen=torch.Generator(device=dev);gen.manual_seed(seed)
   for ep in range(10):
    order=ids[torch.randperm(len(ids),device=dev,generator=gen)]
    for batch in order.split(8192):
     pr=net(e[batch]);ww=weight[batch];loss=(ww*(pr-tgt[batch]).square()).sum()/ww.sum().clamp_min(1e-8)
     opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
   with torch.no_grad():pr=torch.cat([net(b) for b in e.split(32768)]).cpu().numpy()
   return pr,{k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
  for mode,tgt in (('direct',target),('mixed',.5*target+.5*ret),('pure',ret)):
   p,sd=train(tgt)
   for a in (.05,.2):preds[f'{mode}_a{a}_s{seed}']=ref+a*p
   for k,value in sd.items():artifacts[f'{mode}_s{seed}_{k}']=value
  for a in (.05,.2):preds[f'retrieval_a{a}_s{seed}']=ref+a*ret.cpu().numpy()
  artifacts[f'pca_s{seed}']=v.cpu().numpy();artifacts[f'pcascale_s{seed}']=scale.cpu().numpy()
  print('students complete',seed,flush=True)
 for n in list(preds):
  if n.endswith('_s3'):preds[n[:-3]]=.5*(preds[n]+preds[n[:-3]+'_s4'])
 names=list(preds);pairs=[]
 for a in (.05,.2):
  for mode in ('direct','mixed','pure','retrieval'):pairs.append((f'{mode}_a{a}','reference'))
  pairs.extend([(f'pure_a{a}',f'direct_a{a}'),(f'mixed_a{a}',f'direct_a{a}'),(f'pure_a{a}',f'retrieval_a{a}')])
 for seed in (3,4):pairs.append((f'pure_a0.05_s{seed}','reference'))
 def wp(s):
  w,sy,sp,syy,spp,syp=np.moveaxis(s,-1,0)
  return ((syp-sy*sp/w)/np.sqrt(np.maximum(syy-sy**2/w,1e-20)*np.maximum(spp-sp**2/w,1e-20))).mean(-1)
 report={'primary':'pure_a0.05 - reference','results':{},'scope':'Two-seed causal MLP students trained from leave-chain-out retrieval teacher. Reused stride32 feature screen, train mask estimated. Fixed ten epochs, no evaluation-based epoch selection. Other scale and mixed teacher exploratory. Full-stream causal and deployment validation required.'}
 y=y.astype('float64');w=np.abs(y)*z['weight'][:,None]
 for part_id in (1,2):
  sel=part==part_id;cc,inv=np.unique(chain[sel],return_inverse=True);mm=np.zeros((len(names),len(cc),2,6));yy=y[sel];ww=w[sel]
  for j,n in enumerate(names):
   p=np.clip(preds[n][sel],-2,2).astype('float64');np.add.at(mm[j],inv,np.stack([ww,ww*yy,ww*p,ww*yy*yy,ww*p*p,ww*yy*p],-1))
  sc=wp(mm.sum(1));rng=np.random.default_rng(20261005);draw=np.array([wp(mm[:,rng.integers(len(cc),size=len(cc))].sum(1)) for _ in range(2000)])
  rr={'scores':dict(zip(names,map(float,sc))),'chains':len(cc),'comparisons':{}}
  for a,b in pairs:
   ia,ib=names.index(a),names.index(b);rr['comparisons'][a+' - '+b]={'delta':float(sc[ia]-sc[ib]),'conditional_ci95':np.quantile(draw[:,ia]-draw[:,ib],[.025,.975]).tolist()}
  report['results'][str(part_id)]=rr;artifacts[f'moments_{part_id}']=mm;artifacts[f'chains_{part_id}']=cc
 artifacts['names']=np.array(names)
 with Path(sys.argv[3]).open('wb') as f:np.savez_compressed(f,**artifacts)
 Path(sys.argv[2]).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__ == '__main__':
 main()
