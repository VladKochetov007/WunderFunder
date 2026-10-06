"""Embed 256+128+128 recurrent views into one dense trainable h512 GRU."""
from pathlib import Path
import json
import numpy as np
import torch
from wnn.nn import RNNModel
from wnn.data import SequenceFile
from wnn.features import event_features_torch
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave12'
def main():
    torch.set_num_threads(2)
    raw=SequenceFile(R/'datasets/valid.parquet').read(0).features[:2000]
    raw=torch.tensor(raw)[None];x=torch.cat([raw,event_features_torch(raw)],-1)
    reports={}
    for seed in (3,4):
        names=[('d256bc1k_f0' if seed==3 else 'd256bc1ks4_f0',3,.3,False),
               (f'cmp_fullft_s{seed}_ft',1,.2,False),
               ('cmp_rawwmse_s3_ft' if seed==3 else 'cmp_rawwmse65r2_s4_ft',1,.25,True)]
        cks=[torch.load(R/f'artifacts/runs/{n}/model_ep{ep}.pt',weights_only=False,map_location='cpu') for n,ep,_,_ in names]
        widths=[c['args']['hidden'] for c in cks];assert sum(widths)==512
        n_out=cks[0]['model']['head.2.weight'].shape[0]
        assert all(c['model']['head.2.weight'].shape[0]==n_out for c in cks)
        target=RNNModel(211,512,2,n_out=n_out);state={k:torch.zeros_like(v) for k,v in target.state_dict().items()}
        mean=cks[0]['model']['mean'];std=cks[0]['model']['std']
        state['mean']=mean.clone();state['std']=std.clone()
        offset=0;reference=0
        for ck,hidden,(name,epoch,coef,raw_only) in zip(cks,widths,names):
            src=ck['model'];sl=slice(offset,offset+hidden)
            w=src['inp.weight'].double().clone()
            if raw_only:w[:,112:]=0
            state['inp.weight'][sl]=(w*std.double()/src['std'].double()).float()
            state['inp.bias'][sl]=(src['inp.bias'].double()+w@((mean.double()-src['mean'].double())/src['std'].double())).float()
            for layer in (0,1):
                for gate in range(3):
                    sr=slice(gate*hidden,(gate+1)*hidden);dr=slice(gate*512+offset,gate*512+offset+hidden)
                    for typ in ('ih','hh'):
                        state[f'gru.weight_{typ}_l{layer}'][dr,sl]=src[f'gru.weight_{typ}_l{layer}'][sr]
                        state[f'gru.bias_{typ}_l{layer}'][dr]=src[f'gru.bias_{typ}_l{layer}'][sr]
            state['head.0.weight'][sl,sl]=src['head.0.weight'][:,:hidden]
            state['head.0.weight'][sl,512+offset:512+offset+hidden]=src['head.0.weight'][:,hidden:]
            state['head.0.bias'][sl]=src['head.0.bias']
            state['head.2.weight'][:,sl]=coef*src['head.2.weight']
            state['head.2.bias']+=coef*src['head.2.bias']
            original=RNNModel(211,hidden,2,n_out=n_out).eval();original.load_state_dict(src)
            inp=x.clone()
            if raw_only:inp[...,112:]=0
            with torch.no_grad():p,_=original(inp)
            reference=reference+coef*p
            offset+=hidden
        target.load_state_dict(state);target.eval()
        with torch.no_grad():p,_=target(x)
        err=float((p-reference).abs().max());assert err<3e-5,err
        out=R/f'artifacts/runs/w12merge_s{seed}';assert not out.exists();out.mkdir()
        args={**cks[0]['args'],'hidden':512,'arch':'gru','raw_only':False,'family_teachers':''}
        torch.save({'model':state,'args':args,'n_in':211,'epoch':0},out/'model_ep0.pt')
        reports[str(seed)]={'parts':names,'widths':widths,'torch_embedding_max_abs':err,'raw_event_columns_zeroed':True,'normalization_folded_to_first_parent':True}
    (D/'merge-manifest.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports),flush=True)
if __name__=='__main__':main()
