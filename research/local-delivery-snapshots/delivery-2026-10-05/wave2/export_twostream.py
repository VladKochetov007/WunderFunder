"""Exact block-diagonal embedding of two GRUs into the existing native GRU engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

R=Path.home()/"WunderFunder-creative"
sys.path.insert(0,str(R/'scripts'))
from train_rnn import make_model
from wnn.nn import RNNModel
from wnn.data import SequenceFile
from wnn.features import event_features
from wnn.gmetric import global_wp

def convert(ck):
    a=ck['args']; src=ck['model']
    h0=src['inp0.weight'].shape[0]; h1=src['inp1.weight'].shape[0]; h=h0+h1
    layers=a['layers']; no=src['head.2.weight'].shape[0]
    target=RNNModel(ck['n_in'],h,layers,0,n_out=no).eval()
    s=target.state_dict()
    for v in s.values():v.zero_()
    s['mean'].copy_(src['mean']);s['std'].copy_(src['std'])
    for i,off,hh in ((0,0,h0),(1,h0,h1)):
        idx=src[f'idx{i}'].long()
        assert len(idx.unique())==len(idx)
        s['inp.weight'][off:off+hh,idx]=src[f'inp{i}.weight']
        s['inp.bias'][off:off+hh]=src[f'inp{i}.bias']
        for layer in range(layers):
            for typ in ('ih','hh'):
                for gate in range(3):
                    dstrows=slice(gate*h+off,gate*h+off+hh)
                    srcrows=slice(gate*hh,(gate+1)*hh)
                    s[f'gru.weight_{typ}_l{layer}'][dstrows,off:off+hh]=src[f'gru{i}.weight_{typ}_l{layer}'][srcrows]
                    s[f'gru.bias_{typ}_l{layer}'][dstrows]=src[f'gru{i}.bias_{typ}_l{layer}'][srcrows]
    k=src['head.0.weight'].shape[0]
    s['head.0.weight'][:k]=src['head.0.weight'];s['head.0.bias'][:k]=src['head.0.bias']
    s['head.2.weight'][:,:k]=src['head.2.weight'];s['head.2.bias']=src['head.2.bias'].clone()
    target.load_state_dict(s)
    return target

def main():
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint');ap.add_argument('output');ap.add_argument('--rows',type=int,default=20000)
    ap.add_argument('--export-only',action='store_true')
    ap.add_argument('--engine-dir',default=str(R/'research/iterations/2026-10-02-big-ensemble/run_vlad_pair_bw/v3'))
    args=ap.parse_args(); torch.set_num_threads(1)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    original=make_model(ck['args'],ck['n_in'],ck['model']['head.2.weight'].shape[0]).eval()
    original.load_state_dict(ck['model'])
    target=convert(ck)
    weights={k:v.detach().numpy().copy() for k,v in target.state_dict().items()}
    weights['head.2.weight']=weights['head.2.weight'][:2].copy();weights['head.2.bias']=weights['head.2.bias'][:2].copy()
    dst=Path(args.output);assert not dst.exists(),dst
    np.savez_compressed(dst,**weights)
    seq=SequenceFile(R/'datasets/valid.parquet').read(0)
    x=seq.features[:args.rows]
    inp=torch.from_numpy(np.concatenate([x,event_features(x)],axis=-1))[None]
    with torch.no_grad():
        ref,_=original(inp);mapped,_=target(inp)
    ref=ref[0,:,:2].numpy();mapped=mapped[0,:,:2].numpy()
    exact=float(np.max(np.abs(ref-mapped)))
    assert exact<2e-5,exact
    report={'checkpoint':args.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'rows':len(x),'torch_embedding_max_abs':exact}
    np.savez_compressed(dst.with_suffix('.probe.npz'),x=x,reference=ref,y=seq.targets[:len(x)],mask=seq.scored[:len(x)])
    dst.with_suffix('.embedding.json').write_text(json.dumps(report,indent=2))
    if args.export_only:
        print(json.dumps(report,indent=2),flush=True)
        return
    sys.path.insert(0,args.engine_dir)
    import wnn_cpp
    report={'checkpoint':args.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'rows':len(x),'torch_embedding_max_abs':exact,'native':{}}
    for quant in (False,True):
        engine=wnn_cpp.Ensemble([weights],[1.0],fast_activations=True,half_weights=False,dynamic_int8=quant,quant_backend='auto')
        got=np.empty_like(ref);t0=time.perf_counter()
        for i,row in enumerate(x):got[i]=engine.step(np.ascontiguousarray(row))
        elapsed=time.perf_counter()-t0
        engine.reset();repeat=np.array([engine.step(np.ascontiguousarray(row)) for row in x[:300]])
        assert np.array_equal(got[:300],repeat)
        corr=[float(np.corrcoef(ref[:,j],got[:,j])[0,1]) for j in range(2)]
        score=lambda p:global_wp(seq.targets[None,:len(x)],p[None],seq.scored[None,:len(x)])['wp']
        report['native']['int8' if quant else 'fp32']={'max_abs':float(np.max(np.abs(ref-got))),'corr':corr,
            'wp_delta':float(score(got)-score(ref)),'us_per_row':elapsed/len(x)*1e6,'deterministic':True}
        assert min(corr)>.999,report
    dst.with_suffix('.audit.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(dst.with_suffix('.probe.npz'),x=x[:2000],reference=ref[:2000])
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
