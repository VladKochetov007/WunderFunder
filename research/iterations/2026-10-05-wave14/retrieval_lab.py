"""Predeclared larger-bank retrieval screen; no evaluation chains enter bank or PCA."""
import sys
import json
from pathlib import Path

def main():
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    z=np.load(sys.argv[1]);part=z['partition'];chain=z['chain'];fit=np.flatnonzero(part==0);query=np.flatnonzero(part!=0)
    x0=z['x'].astype('float32');y=np.clip(z['y'],-2,2).astype('float64');ref=z['reference'];w=np.abs(y)*z['weight'][:,None]
    target=torch.tensor(y-ref,dtype=torch.float32,device=device);weight=torch.tensor(w,dtype=torch.float32,device=device)
    chains=np.unique(chain[fit]);groups=[fit[chain[fit]==c] for c in chains]
    assert set(chain[fit]).isdisjoint(chain[query])
    preds={'reference':ref[query]};logs=[]
    for seed in (3,4):
        rng=np.random.default_rng(seed)
        # Equal quotas per chain, sampled without replacement. Longer chains cannot dominate.
        def bank(size):
            quota=(size+len(groups)-1)//len(groups)
            chunks=[rng.choice(g,min(quota,len(g)),replace=False) for g in groups]
            ids=np.concatenate(chunks)
            if len(ids)>size:ids=rng.choice(ids,size,replace=False)
            return ids
        banks={size:bank(size) for size in (8192,65536)}
        for embedding in ('gru','all'):
            raw=x0[:,:128] if embedding=='gru' else x0
            mean=raw[fit].mean(0);std=raw[fit].std(0).clip(.05)
            x=torch.tensor(np.clip((raw-mean)/std,-8,8),device=device)
            torch.manual_seed(seed)
            _,_,v=torch.pca_lowrank(x[torch.tensor(fit[::4],device=device)],q=64,center=False,niter=3)
            e=x@v;e=e/e[torch.tensor(fit,device=device)].std(0).clamp_min(.1)
            eq=e[torch.tensor(query,device=device)]
            for size,bank_np in banks.items():
                bi=torch.tensor(bank_np,device=device);eb=e[bi];ebnorm=eb.square().sum(1)
                residuals={32:[],128:[]}
                with torch.no_grad():
                    for start in range(0,len(query),1024):
                        qe=eq[start:start+1024]
                        dist=(qe.square().sum(1)[:,None]+ebnorm[None]-2*qe@eb.T).clamp_min(0)
                        dd,ind=dist.topk(128,largest=False,sorted=True)
                        near=bi[ind]
                        for k in residuals:
                            ww=weight[near[:,:k]]/(dd[:,:k,None]+1.)
                            value=(ww*target[near[:,:k]]).sum(1)/ww.sum(1).clamp_min(1e-8)
                            residuals[k].append(value.cpu().numpy())
                for k,values in residuals.items():
                    r=np.concatenate(values)
                    for scale in (.05,.2):preds[f'{embedding}_b{size}_k{k}_a{scale}_s{seed}']=ref[query]+scale*r
                logs.append({'seed':seed,'embedding':embedding,'requested_bank':size,'bank_rows':len(bank_np),'bank_chains':len(np.unique(chain[bank_np]))})
                print('retrieval complete',logs[-1],flush=True)
            del e,x,eq
    singles=list(preds)
    for name in singles:
        if name.endswith('_s3'):preds[name[:-3]]=.5*(preds[name]+preds[name[:-3]+'_s4'])
    names=list(preds)
    primary='gru_b65536_k128_a0.05';small='gru_b8192_k128_a0.05'
    pairs=[(primary,'reference'),(primary,small)]+[(primary+'_s'+str(s),'reference') for s in (3,4)]
    pairs += [(n,'reference') for n in names if not n.endswith(('_s3','_s4')) and n not in ('reference',primary)]
    def wp(s):
        ww,sy,sp,syy,spp,syp=np.moveaxis(s,-1,0)
        return ((syp-sy*sp/ww)/np.sqrt(np.maximum(syy-sy**2/ww,1e-20)*np.maximum(spp-sp**2/ww,1e-20))).mean(-1)
    report={'device':device,'bank_protocol':logs,'primary':primary,'results':{},'scope':'Sampled stride32 screen, reused validation, estimated train mask, frozen backbone, original organizer provenance uncertain. PCA and banks only fit partition. Evaluation chains fully excluded from bank. Primary comparison predeclared; other configurations exploratory. Successful teacher requires causal student/full-stream validation and CPU-budget check.'}
    artifacts={}
    for p in (1,2):
        sel=part[query]==p;cc,inv=np.unique(chain[query][sel],return_inverse=True)
        mm=np.zeros((len(names),len(cc),2,6));yy=y[query][sel];ww=w[query][sel]
        for j,n in enumerate(names):
            pr=np.clip(preds[n][sel],-2,2).astype('float64')
            np.add.at(mm[j],inv,np.stack([ww,ww*yy,ww*pr,ww*yy*yy,ww*pr*pr,ww*yy*pr],-1))
        scores=wp(mm.sum(1));rng=np.random.default_rng(20261005)
        draws=np.array([wp(mm[:,rng.integers(len(cc),size=len(cc))].sum(1)) for _ in range(2000)])
        result={'rows':int(sel.sum()),'chains':len(cc),'scores':dict(zip(names,map(float,scores))),'comparisons':{}}
        for a,b in pairs:
            ia,ib=names.index(a),names.index(b)
            result['comparisons'][a+' - '+b]={'delta':float(scores[ia]-scores[ib]),'conditional_ci95':np.quantile(draws[:,ia]-draws[:,ib],[.025,.975]).tolist()}
        report['results'][str(p)]=result;artifacts['moments_'+str(p)]=mm;artifacts['chains_'+str(p)]=cc
    artifacts['names']=np.array(names);artifacts['query_rows']=query;artifacts['primary_predictions']=preds[primary]
    with Path(sys.argv[3]).open('wb') as f:np.savez_compressed(f,**artifacts)
    Path(sys.argv[2]).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()
