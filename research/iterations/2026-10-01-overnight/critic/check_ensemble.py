"""Independent exact actor objective AST, scale parameterization, and export checks."""
from pathlib import Path
import ast,hashlib,json,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];ENS=HERE.parent/'ensemble'
sys.path[:0]=[str(ROOT/'src'),str(ENS)]
from support import ScaleModel,export_state
from wnn.nn import RNNModel,weighted_pooled_corr_loss

def literal(pred,target,mask):
    rs=[]
    for k in range(2):
        y=target[...,k].float().clamp(-2,2).flatten();p=pred[...,k].float().flatten();w=y.abs()*mask.flatten()
        total=w.sum().clamp_min(1e-8);dy=y-(w*y).sum()/total;dp=p-(w*p).sum()/total
        vy=(w*dy*dy).sum()/total;vp=(w*dp*dp).sum()/total;cov=(w*dy*dp).sum()/total
        rs.append(cov/torch.sqrt(vy*vp+1e-12))
    return 1-torch.stack(rs).mean()

def main():
    torch.set_num_threads(2);torch.manual_seed(828);
    file=ENS/'train_ensemble.py';tree=ast.parse(file.read_text())
    blocks=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and ast.unparse(n).startswith('loss = 4.0 * objective')]
    assert len(blocks)==1;code=compile(ast.Module(body=blocks,type_ignores=[]),str(file),'exec')
    B,T=3,27;initial=torch.randn(B,T,23)*4;target=torch.randn(B,T,2)*3;mask=(torch.rand(B,T)>.3).float();partner=torch.randn(B,T,2)
    out=initial.clone().requires_grad_();env={'out':out,'partner_sum':partner,'sl':slice(None),'y':target,'main_valid':mask,'objective':weighted_pooled_corr_loss}
    exec(code,env);env['loss'].backward();got=out.grad.clone()
    o2=initial.clone().requires_grad_();expected=4*literal((o2[...,:2]+partner)/4,target,mask);expected.backward()
    diff=float((got-o2.grad).abs().max());assert diff<1e-7
    assert partner.grad is None and torch.count_nonzero(got[...,2:])==0
    # Identical fixed partners give the same own-slot main gradient as standalone.
    own=initial[...,:2].clone().requires_grad_();direct=weighted_pooled_corr_loss(own,target,mask);direct.backward();a=own.grad.clone()
    other=own.detach().clone().requires_grad_();(4*weighted_pooled_corr_loss((other+3*own.detach())/4,target,mask)).backward();assert torch.allclose(a,other.grad,atol=1e-8,rtol=1e-6)
    # Nonidentity scale export must match and must not alter private aux outputs.
    m=ScaleModel(211,hidden=16,layers=2,n_out=23).eval();before={k:v.clone() for k,v in m.state_dict().items()}
    assert [name for name,p in m.named_parameters() if p.requires_grad]==['ensemble_log_scale']
    m.ensemble_log_scale.data.copy_(torch.tensor([.3,-.2]));x=torch.randn(2,31,211)
    y,st=m(x);sd=export_state(m);assert 'ensemble_log_scale' not in sd
    ordinary=RNNModel(211,hidden=16,layers=2,n_out=23).eval();ordinary.load_state_dict(sd,strict=True)
    yo,so=ordinary(x);export_diff=float((y-yo).abs().max());assert export_diff<2e-7 and torch.equal(st,so)
    for key,value in before.items():
        if key not in ['ensemble_log_scale','head.2.weight','head.2.bias']:assert torch.equal(sd[key],value)
    opt=torch.optim.AdamW([m.ensemble_log_scale],lr=5e-4)
    loss=4*weighted_pooled_corr_loss((y[...,:2]+torch.randn_like(y[...,:2]))/4,torch.randn_like(y[...,:2]),torch.ones(2,31))
    loss.backward();assert m.ensemble_log_scale.grad is not None and torch.count_nonzero(m.ensemble_log_scale.grad)>0;opt.step()
    for name,p in m.named_parameters():
        if name!='ensemble_log_scale':assert p.grad is None and torch.equal(p,before[name])
    result={'status':'PASS','objective_gradient_max_abs':diff,'nonidentity_scale_export_max_abs':export_diff,'identity_partner_gradient':'matches standalone with fixed4 multiplier','only_trainable_parameters':['ensemble_log_scale'],'backbone_and_original_head_unchanged':True,'scope':'synthetic CPU checks only; no data, fits or WP queries','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [file,ENS/'support.py',ENS/'cache_partners.py']}}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
