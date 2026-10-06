"""Compact chain-separated causal and privileged features for research-only teacher tests."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from wnn.data import SequenceFile
from train_rnn import build_inputs, make_model

R=Path(os.environ['WUNDER_RESEARCH_ROOT'])
D=R/'research/iterations/2026-10-05-wave3'
def main():
    torch.set_num_threads(3)
    cf=np.load(R/'cache/chain_folds.npz')
    seen=set(c for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold'])
        if (s=='valid' and f!=0) or (s=='train' and int(hashlib.sha256(f'half:{g}'.encode()).hexdigest(),16)%2==0))
    fit=set(np.load(R/'research/iterations/2026-10-05-wave2/complement_fit.npy'))
    hold=set(np.load(R/'research/iterations/2026-10-05-wave2/complement_hold.npy'))
    ck=torch.load(R/'artifacts/runs/cmp_fullft_s3_ft/model_ep1.pt',map_location='cpu',weights_only=False)
    model=make_model(ck['args'],ck['n_in'],ck['model']['head.2.weight'].shape[0]).cuda().eval()
    model.load_state_dict(ck['model'])
    assert not ck['args'].get('lookahead')
    buf={k:[] for k in ('x','future','y','weight','reference','base','chain','group','row','partition','market')}
    for split in ('train','valid'):
        z=np.load(R/f'cache/{split}_targets.npz')
        cm={g:(int(c),int(f)) for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if s==split}
        reference=np.load(R/f'cache/pred_w2_reference_{split}.npy',mmap_mode='r')
        prop=np.load(R/'cache/propcache_prop_f0_train.npy',mmap_mode='r') if split=='train' else None
        sf=SequenceFile(R/f'datasets/{split}.parquet')
        for i,g in enumerate(z['row_group']):
            c,f=cm[g]
            if split=='train':
                if g not in fit|hold:continue
                part=0 if g in fit else 1
            else:
                if f!=0 or c in seen:continue
                part=2
            seq=sf.read(int(g));xx=torch.from_numpy(seq.features)[None].cuda()
            # Each chosen row has equal inclusion probability within a sequence.
            offset=int(hashlib.sha256(f'probe:{split}:{g}'.encode()).hexdigest(),16)%32
            rows=np.arange(99+offset,19968,32)
            h=None;features=[];preds=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                inp=build_inputs(xx)
                for start in range(0,20000,2000):
                    v=model.encode((inp[:,start:start+2000]-model.mean)/model.std)
                    o,h=model.gru(v,h)
                    feat=torch.cat([o,v],-1)
                    features.append(feat[0].cpu().numpy());preds.append(model.head(feat)[0,:,:2].float().cpu().numpy())
            feat=np.concatenate(features);p=np.concatenate(preds)
            n=len(rows)
            values={'x':feat[rows],'future':feat[rows+32], 'y':seq.targets[rows],
                'weight':seq.scored[rows].astype('float32') if split=='valid' else np.asarray(prop[g,rows]),
                'reference':np.asarray(reference[i,rows]),'base':p[rows],
                'chain':np.full(n,c,np.int32),'group':np.full(n,g,np.int32),'row':rows.astype('int16'),
                'partition':np.full(n,part,np.uint8),'market':np.full(n,z['market_class'][i],np.uint8)}
            for k,v in values.items():buf[k].append(v)
            if len(buf['x'])%50==0:print(split,len(buf['x']),flush=True)
    arrays={k:np.concatenate(v) for k,v in buf.items()}
    for a in (0,1,2):
        for b in range(a+1,3):assert set(arrays['chain'][arrays['partition']==a]).isdisjoint(arrays['chain'][arrays['partition']==b])
    path=D/'probe.npz';assert not path.exists();np.savez_compressed(path,**arrays)
    meta={'rows':len(arrays['x']),'counts':{str(p):int((arrays['partition']==p).sum()) for p in (0,1,2)},
        'stride':32,'future_horizon':32,'backbone':'cmp_fullft_s3_ft@ep1','partitions':{'0':'fit train half1 chain excluded','1':'hold train half1 chain excluded','2':'valid fold0 chain excluded'},
        'caveat':'Sampled screening only; train scoring mask estimated; reused validation; organizer provenance uncertified.'}
    (D/'probe-manifest.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
if __name__=='__main__':main()
