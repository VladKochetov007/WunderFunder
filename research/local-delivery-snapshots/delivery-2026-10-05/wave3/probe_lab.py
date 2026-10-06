"""GPU compact feature lab. Privileged inputs are restricted to teacher-only fits."""
import sys
import json
import hashlib
from pathlib import Path

def main():
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    z=np.load(sys.argv[1]);out=Path(sys.argv[2]);predpath=Path(sys.argv[3])
    x0=z['x'].astype('float32');f0=z['future'].astype('float32');part=z['partition'];chain=z['chain']
    fit=part==0
    mean=x0[fit].mean(0);std=x0[fit].std(0).clip(.05)
    x=torch.tensor(np.clip((x0-mean)/std,-8,8),device=dev)
    future=torch.tensor(np.clip((f0-mean)/std,-8,8),device=dev)
    ref=torch.tensor(z['reference'],device=dev);yy=torch.tensor(np.clip(z['y'],-2,2),device=dev)
    weight=yy.abs()*torch.tensor(z['weight'],device=dev)[:,None]
    ids=torch.tensor(np.flatnonzero(fit),device=dev)
    folds=np.array([int(hashlib.sha256(f'future-cf:{int(c)}'.encode()).hexdigest(),16)%2 for c in chain])
    target=yy-ref
    preds={'reference':ref.cpu().numpy()};logs=[]
    def train(features, target, indices, seed, epochs=10):
        torch.manual_seed(seed)
        net=nn.Sequential(nn.Linear(features.shape[1],128),nn.GELU(),nn.Linear(128,2)).to(dev)
        nn.init.zeros_(net[2].weight);nn.init.zeros_(net[2].bias)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.03)
        gen=torch.Generator(device=dev);gen.manual_seed(seed)
        for ep in range(epochs):
            idx=indices[torch.randperm(len(indices),generator=gen,device=dev)]
            for batch in idx.split(8192):
                p=net(features[batch]);w=weight[batch]
                loss=(w*(p-target[batch]).square()).sum()/w.sum().clamp_min(1e-8)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        with torch.no_grad():result=torch.cat([net(v) for v in features.split(32768)])
        return result
    privileged=torch.cat([x,future],-1)
    for seed in (3,4):
        causal=train(x,target,ids,seed)
        oracle=train(privileged,target,ids,seed)
        oof=torch.zeros_like(target)
        for fold in (0,1):
            ix=torch.tensor(np.flatnonzero(fit&(folds!=fold)),device=dev)
            p=train(privileged,target,ix,seed)
            sel=torch.tensor(fit&(folds==fold),device=dev);oof[sel]=p[sel]
        # Causal reachability estimate itself is chain cross-fitted.
        reach_oof=torch.zeros_like(target)
        for fold in (0,1):
            ix=torch.tensor(np.flatnonzero(fit&(folds!=fold)),device=dev)
            p=train(x,oof,ix,seed+10)
            sel=torch.tensor(fit&(folds==fold),device=dev);reach_oof[sel]=p[sel]
        gate=(reach_oof*oof/(oof.square()+.05)).clamp(0,1)
        blind=train(x,.5*target+.5*oof,ids,seed)
        gated=train(x,.5*target+.5*gate*oof,ids,seed)
        for name,p in [('causal',causal),('privileged_oracle',oracle),('blind_distill',blind),('gated_distill',gated)]:
            preds[f'{name}_s{seed}']=(ref+.2*p).cpu().numpy()
        logs.append({'seed':seed,'mean_reachability_gate':float(gate[ids].mean()),'fit_weighted_future_delta_mse':float((weight[ids]*(oof[ids]-target[ids]).square()).sum()/weight[ids].sum())})
        print('future family complete',seed,flush=True)
        # Retrieval is an explicit small-bank ceiling screen, not a final ANN system.
        torch.manual_seed(seed)
        _,_,v=torch.pca_lowrank(x[ids][::4],q=32,center=False,niter=3)
        emb=x@v
        escale=emb[ids].std(0).clamp_min(.1);emb=emb/escale
        rng=np.random.default_rng(seed)
        bank_np=rng.choice(np.flatnonzero(fit),min(8192,int(fit.sum())),replace=False)
        bank=torch.tensor(bank_np,device=dev);be=emb[bank];bc=torch.tensor(chain[bank_np],device=dev)
        values=[]
        with torch.no_grad():
            for start in range(0,len(x),2048):
                e=emb[start:start+2048]
                dist=(e.square().sum(1)[:,None]+be.square().sum(1)[None]-2*e@be.T).clamp_min(0)
                same=torch.tensor(chain[start:start+len(e)],device=dev)[:,None]==bc[None]
                dist.masked_fill_(same,float('inf'))
                dd,ind=dist.topk(32,largest=False)
                near=bank[ind]
                w=weight[near]/(dd[...,None]+1.)
                value=(w*target[near]).sum(1)/w.sum(1).clamp_min(1e-8)
                values.append(value)
        ret=torch.cat(values)
        preds[f'retrieval_s{seed}']=(ref+.2*ret).cpu().numpy()
        print('retrieval complete',seed,flush=True)
    for name in ('causal','privileged_oracle','blind_distill','gated_distill','retrieval'):
        preds[name]=.5*(preds[name+'_s3']+preds[name+'_s4'])
    def wp(s):
        w,sy,sp,syy,spp,syp=np.moveaxis(s,-1,0)
        return ((syp-sy*sp/w)/np.sqrt(np.maximum(syy-sy**2/w,1e-20)*np.maximum(spp-sp**2/w,1e-20))).mean(-1)
    report={'device':dev,'rows':len(x),'logs':logs,'results':{},'caveats':['Sampled stride32 features; full-sequence replay required for promotion.','Validation adaptively reused; train scoring mask estimated.','Privileged oracle cannot be deployed.','Retrieval uses a small 8192-row bank; negative result does not exclude a larger retrieval teacher.']}
    names=list(preds);y=np.clip(z['y'],-2,2).astype('float64');w=np.abs(y)*z['weight'][:,None]
    pairs=[('causal','reference'),('privileged_oracle','causal'),('blind_distill','causal'),('gated_distill','causal'),('gated_distill','blind_distill'),('retrieval','reference')]
    for p in (1,2):
        sel=part==p;chains,inv=np.unique(chain[sel],return_inverse=True)
        moments=np.zeros((len(names),len(chains),2,6))
        wy=w[sel];ty=y[sel]
        for j,name in enumerate(names):
            pr=np.clip(preds[name][sel],-2,2).astype('float64')
            row=np.stack([wy,wy*ty,wy*pr,wy*ty*ty,wy*pr*pr,wy*ty*pr],-1)
            np.add.at(moments[j],inv,row)
        scores=wp(moments.sum(1));rng=np.random.default_rng(20261005)
        draws=np.array([wp(moments[:,rng.integers(len(chains),size=len(chains))].sum(1)) for _ in range(2000)])
        result={'rows':int(sel.sum()),'chains':len(chains),'scores':dict(zip(names,map(float,scores))),'comparisons':{}}
        for a,b in pairs:
            ia,ib=names.index(a),names.index(b)
            result['comparisons'][a+' - '+b]={'delta':float(scores[ia]-scores[ib]),'conditional_ci95':np.quantile(draws[:,ia]-draws[:,ib],[.025,.975]).tolist()}
        report['results'][str(p)]=result
    np.savez_compressed(predpath,**preds)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()
