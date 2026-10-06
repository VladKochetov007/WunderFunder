"""Four-event competing-move prototype on frozen causal GRU representations."""
import sys
import json
from pathlib import Path

def main():
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    torch.set_num_threads(4)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    z=np.load(sys.argv[1]);h=np.load(sys.argv[2]);report_path=Path(sys.argv[3]);artifact=Path(sys.argv[4])
    feat=torch.tensor(z['x'].astype('float32'),device=dev)
    weights={k:torch.tensor(h[k],device=dev) for k in h.files}
    with torch.no_grad():
        hidden=F.gelu(F.linear(feat,weights['head.0.weight'],weights['head.0.bias']))
        parent=F.linear(hidden,weights['head.2.weight'],weights['head.2.bias'])
        spread=parent[:,2:14].softmax(-1);gate=parent[:,17:21].softmax(-1)[:,3:4]
    part=z['partition'];chain=z['chain'];fit=part==0;ix=torch.tensor(np.flatnonzero(fit),device=dev)
    mean=feat[ix].mean(0);std=feat[ix].std(0).clamp_min(.05);x=((feat-mean)/std).clamp(-8,8)
    ref=torch.tensor(z['reference'],device=dev);y=torch.tensor(np.clip(z['y'],-2,2),device=dev)
    w=y.abs()*torch.tensor(z['weight'],device=dev)[:,None]
    # Mid moves {-2,-1.5,...,2} ticks. Four future moves contribute weights 4,3,2,1.
    sp=np.arange(1,13)[None,:,None]
    displacement=np.arange(-40,41)[:,None,None]/8
    u=np.rint(sp/2+displacement)
    t0=np.where(u<0,-(sp+2),u-sp);t1=np.where(u>sp,-(sp+2),-u)
    table=np.clip(np.concatenate([t0,t1],-1)*.65687,-2,2).astype('float32')
    numerator=torch.tensor((abs(table)*table).reshape(81,24),device=dev)
    denominator=torch.tensor(abs(table).reshape(81,24),device=dev)
    class Race(nn.Module):
        def __init__(self):
            super().__init__();self.net=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,9))
            nn.init.zeros_(self.net[2].weight);nn.init.constant_(self.net[2].bias,-2.)
            self.net[2].bias.data[4]=2.
        def probabilities(self,values):
            hazards=F.softplus(self.net(values))+1e-5
            p=hazards/hazards.sum(-1,keepdim=True)
            spectra=[]
            for k in (4,3,2,1):
                polynomial=torch.zeros(len(values),128,device=values.device)
                polynomial[:,torch.arange(9,device=values.device)*k]=p
                spectra.append(torch.fft.rfft(polynomial))
            pmf=torch.fft.irfft(spectra[0]*spectra[1]*spectra[2]*spectra[3],n=128)[:,:81].clamp_min(0)
            return pmf/pmf.sum(-1,keepdim=True).clamp_min(1e-8)
        def forward(self,values,sp):
            pmf=self.probabilities(values)
            num=((pmf@numerator).reshape(-1,12,2)*sp[:,:,None]).sum(1)
            den=((pmf@denominator).reshape(-1,12,2)*sp[:,:,None]).sum(1)
            return num/den.clamp_min(.02)
    torch.manual_seed(3);test=Race().to(dev)
    pmf=test.probabilities(x[:20]);assert torch.allclose(pmf.sum(1),torch.ones(20,device=dev),atol=1e-6)
    loss=test(x[:20],spread[:20]).square().mean();loss.backward();assert torch.isfinite(test.net[2].weight.grad).all() and test.net[2].weight.grad.abs().sum()>0
    predictions={'reference':ref.cpu().numpy()};states={}
    for seed in (3,4):
        for kind in ('direct','race'):
            torch.manual_seed(seed)
            if kind=='race':model=Race().to(dev)
            else:
                model=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,2)).to(dev)
                nn.init.zeros_(model[2].weight);nn.init.zeros_(model[2].bias)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.03)
            gen=torch.Generator(device=dev);gen.manual_seed(seed)
            for epoch in range(8):
                ids=ix[torch.randperm(len(ix),generator=gen,device=dev)]
                for ids0 in ids.split(4096):
                    delta=model(x[ids0],spread[ids0])-parent[ids0,:2] if kind=='race' else model(x[ids0])
                    p=ref[ids0]+.2*gate[ids0]*delta
                    loss=(w[ids0]*(p-y[ids0]).square()).sum()/w[ids0].sum().clamp_min(1e-8)
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            outs=[]
            with torch.no_grad():
                for start in range(0,len(x),8192):
                    sl=slice(start,start+8192)
                    delta=model(x[sl],spread[sl])-parent[sl,:2] if kind=='race' else model(x[sl])
                    outs.append((ref[sl]+.2*gate[sl]*delta).cpu().numpy())
            predictions[f'{kind}_s{seed}']=np.concatenate(outs)
            for k,v in model.state_dict().items():states[f'{kind}_s{seed}.{k}']=v.cpu().numpy()
            print(kind,seed,'complete',flush=True)
    for kind in ('direct','race'):predictions[kind]=.5*(predictions[kind+'_s3']+predictions[kind+'_s4'])
    def wp(s):
        w,sy,sp,syy,spp,syp=np.moveaxis(s,-1,0)
        return ((syp-sy*sp/w)/np.sqrt(np.maximum(syy-sy**2/w,1e-20)*np.maximum(spp-sp**2/w,1e-20))).mean(-1)
    names=list(predictions);yy=y.cpu().numpy().astype('float64');ww=w.cpu().numpy().astype('float64')
    report={'status':'complete','rows':len(x),'tests':'event PMF normalization and finite nonzero gradient passed','results':{},
        'scope':'Sampled frozen-backbone prototype; requires full replay for promotion. Parent causal class3 probability gates all updates. No target-derived inference gate.',
        'limitations':['Four future moves conditionally iid; amplitudes fixed on a nine-point grid.','Event-clock competing rates normalized to move probabilities; calendar-time hazard not identified by this four-event target model.','Training mask estimated; reused evaluation data.']}
    for part_id in (1,2):
        sel=part==part_id;chains,inv=np.unique(chain[sel],return_inverse=True)
        m=np.zeros((len(names),len(chains),2,6));wy=ww[sel];ty=yy[sel]
        for j,name in enumerate(names):
            p=np.clip(predictions[name][sel],-2,2).astype('float64')
            np.add.at(m[j],inv,np.stack([wy,wy*ty,wy*p,wy*ty*ty,wy*p*p,wy*ty*p],-1))
        score=wp(m.sum(1));rng=np.random.default_rng(20261005)
        draws=np.array([wp(m[:,rng.integers(len(chains),size=len(chains))].sum(1)) for _ in range(2000)])
        result={'chains':len(chains),'scores':dict(zip(names,map(float,score))),'comparisons':{}}
        for a,b in [('race','direct'),('race','reference'),('direct','reference')]:
            ai,bi=names.index(a),names.index(b)
            result['comparisons'][a+' - '+b]={'delta':float(score[ai]-score[bi]),'conditional_ci95':np.quantile(draws[:,ai]-draws[:,bi],[.025,.975]).tolist()}
        report['results'][str(part_id)]=result;states[f'moments_{part_id}']=m
    states['normalization_mean']=mean.cpu().numpy();states['normalization_std']=std.cpu().numpy()
    with artifact.open('wb') as f:np.savez_compressed(f,**states)
    report_path.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()
