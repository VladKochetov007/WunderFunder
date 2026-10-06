"""Feature-prediction-only streaming gradient adapter, reset at each sequence."""
import os
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
import torch
from train_rnn import build_inputs,make_model
from ssl_model import SSL_COLS
from wnn.data import SequenceFile
R=Path(os.environ['WUNDER_RESEARCH_ROOT']);D=R/'research/iterations/2026-10-05-wave9'

def adaptation(errors,sw,pw,rate):
    gram=sw@sw.T
    eigen,V=np.linalg.eigh(gram)
    lr=rate/max(float(eigen.max()),1e-6)
    coeff=.999-lr*eigen
    drive=errors@V
    u=np.column_stack([lfilter([lr],[1,-a],drive[:,j]) for j,a in enumerate(coeff)])
    correction=u@(pw@sw.T@V).T
    return np.clip(.2*correction,-.2,.2).astype('float32')

def main():
    torch.set_num_threads(3)
    rng=np.random.default_rng(3);sw=rng.normal(size=(16,128));pw=rng.normal(size=(2,128));err=rng.normal(size=(200,16));err[0]=0
    got=adaptation(err,sw,pw,.01)
    u=np.zeros(16);lr=.01/np.linalg.eigvalsh(sw@sw.T).max();out=[]
    for e in err:
        u=.999*u+lr*(e-sw@sw.T@u);out.append(np.clip(.2*(pw@sw.T@u),-.2,.2))
    assert np.allclose(got,out,atol=1e-7)
    assert np.array_equal(got[:99],adaptation(err[:99],sw,pw,.01))
    cf=np.load(R/'cache/chain_folds.npz');seen=set(c for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if (s=='valid' and f!=0) or (s=='train' and int(hashlib.sha256(f'half:{g}'.encode()).hexdigest(),16)%2==0))
    hold=set(np.load(R/'research/iterations/2026-10-05-wave2/complement_hold.npy'))
    counts={}
    for seed in (3,4):
        ck=torch.load(R/f'artifacts/runs/w7ssl_control_s{seed}_ft/model_ep1.pt',map_location='cpu',weights_only=False)
        model=make_model(ck['args'],ck['n_in'],ck['model']['head.2.weight'].shape[0]).cuda().eval();model.load_state_dict(ck['model'])
        sw=ck['model']['ssl_head.weight'].numpy().astype('float64');pw=ck['model']['head.2.weight'][:2].numpy().astype('float64')
        for split in ('valid','train'):
            groups=np.load(R/f'cache/{split}_targets.npz')['row_group']
            cm={g:(c,f) for s,g,c,f in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if s==split}
            ids=[i for i,g in enumerate(groups) if (g in hold if split=='train' else cm[g][1]==0 and cm[g][0] not in seen)]
            counts[split]=len(ids)
            final={mode:R/f'cache/pred_w9adapt_{mode}_s{seed}_{split}.npy' for mode in ('frozen','r01','r05')}
            assert all(not p.exists() for p in final.values())
            dest={m:np.lib.format.open_memmap(str(p)+'.partial',mode='w+',dtype=np.float32,shape=(len(groups),20000,2)) for m,p in final.items()}
            for arr in dest.values():arr[:]=np.nan
            sf=SequenceFile(R/f'datasets/{split}.parquet')
            for j,i in enumerate(ids):
                seq=sf.read(int(groups[i]));x=torch.from_numpy(seq.features)[None].cuda()
                pred=[];ssl=[];h=None
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    inp=build_inputs(x)
                    for start in range(0,20000,4000):
                        p,h=model(inp[:,start:start+4000],h)
                        pred.append(p[0,:,:2].float().cpu().numpy());ssl.append(model.ssl_output[0].float().cpu().numpy())
                p=np.concatenate(pred);f=np.concatenate(ssl)
                observed=np.diff(seq.features[:,SSL_COLS],axis=0)/ck['model']['std'][list(SSL_COLS)].numpy()
                errors=np.zeros((20000,len(SSL_COLS)),dtype='float64')
                errors[1:]=np.clip(observed,-3,3)-f[:-1]
                dest['frozen'][i]=p
                dest['r01'][i]=p+adaptation(errors,sw,pw,.01)
                dest['r05'][i]=p+adaptation(errors,sw,pw,.05)
                if j%100==0:print(seed,split,j,flush=True)
            for mode,arr in dest.items():arr.flush()
            del dest
            for p in final.values():Path(str(p)+'.partial').rename(p)
    (D/'adaptation-manifest.json').write_text(json.dumps({'counts':counts,'rates':[.01,.05],'decay':.999,'output_shrink':.2,'max_output_correction':.2,'tests':'one-row gradient recurrence, prefix invariance and sequence reset by construction','observations':'feature deltas at t compared with forecast from t-1; no labels or scoring mask enter adaptation','scope':'fixed SSL-control parents; full sequence replay on chain-excluded validation and train hold'},indent=2))
if __name__=='__main__':main()
