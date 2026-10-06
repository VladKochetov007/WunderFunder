"""Same function, different parameterization: +16 conversion versus direct linear maps."""
from pathlib import Path
import json
import torch
from wnn.nn import RNNModel,weighted_pooled_corr_loss
from wnn.data import SequenceFile
from linear_baseline import LinearBaseline
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave13'
def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    ck=torch.load(R/'artifacts/runs/w13base_direct/model_ep0.pt',map_location='cpu',weights_only=False)
    old={k:v.clone() for k,v in ck['model'].items()}
    old['inp.bias']+=16
    old['gru.bias_ih_l0']-=16*old['gru.weight_ih_l0'].sum(1)
    old['head.0.bias']+=16
    old['head.2.bias']-=16*old['head.2.weight'].sum(1)
    sf=SequenceFile(R/'datasets/valid.parquet');seq=[sf.read(g) for g in (0,5)]
    x=torch.stack([torch.tensor(s.features[:1000]) for s in seq]).cuda()
    x=torch.cat([x,torch.zeros(2,1000,99,device='cuda')],-1)
    y=torch.stack([torch.tensor(s.targets[:1000]) for s in seq]).cuda()
    mask=torch.stack([torch.tensor(s.scored[:1000]) for s in seq]).cuda();mask[:,:99]=False
    records={};initial={}
    for mode,cls,weights in [('shifted',RNNModel,old),('linear',LinearBaseline,ck['model'])]:
        model=cls(211,128,2,n_out=2).cuda();model.load_state_dict(weights)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
        trace=[]
        for step in range(11):
            p,_=model(x);loss=weighted_pooled_corr_loss(p,y,mask)
            if step==0:initial[mode]=p.detach().cpu()
            trace.append({'step':step,'loss':float(loss.detach()),'prediction_rms_drift':float((p.detach().cpu()-initial[mode]).square().mean().sqrt())})
            if step==10:break
            opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            trace[-1]['gradient_norm_before_clip']=float(norm);opt.step()
        records[mode]=trace
    error=float((initial['shifted']-initial['linear']).abs().max());assert error<1e-4,error
    result={'initial_function_max_abs':error,'trace':records,'scope':'Optimization conditioning diagnostic on two short reused sequences, not a validation score or candidate training.'}
    (D/'conditioning-probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
