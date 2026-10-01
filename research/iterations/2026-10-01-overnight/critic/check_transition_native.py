"""Independent native/feature oracle, nonzero-head parity, and CPU timing (no labels)."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
TRANS=HERE.parent/'transition'

def oracle(raw, nuisance, arm):
    # Explicit prefix lists and integer Counter objects, independent of actor array cumsums.
    prev=None; marks=[]; at=[]; observed=Counter(); pairs=Counter(); past_out=Counter(); out=[]
    c=np.asarray(nuisance['centers'],np.float64); pi=np.asarray(nuisance['pi']); R=np.asarray(nuisance['R'])
    held=np.zeros(68,np.float32);held[35:51]=pi
    for row,x in enumerate(np.asarray(raw,np.float32)):
        good=lambda v:v is not None and all(np.isfinite(float(z)) and abs(z)<np.float32(5.19) for z in v)
        e=(good(x) and good(prev) and any(float(a)!=float(b) for a,b in zip(x,prev)) and np.float32(max(x)-min(x))>np.float32(1e-6))
        held[34]=bool(e)
        if e:
            mark=sum((1<<j) if float(x[j])>c[j] else 0 for j in range(4))
            if arm=='F':prob=np.array([(observed[j]+16*pi[j])/(len(marks)+16) for j in range(16)])
            elif not marks:prob=pi.copy()
            else:prob=np.array([(pairs[marks[-1],j]+16*R[marks[-1],j])/(past_out[marks[-1]]+16) for j in range(16)])
            held[:32]=0;held[mark]=1
            if marks:held[16+marks[-1]]=1
            held[35:51]=prob;held[51:67]=-prob;held[51+mark]=1-prob[mark]
            held[67]=min(-np.log(prob[mark]),10)/10
            if marks:pairs[marks[-1],mark]+=1;past_out[marks[-1]]+=1
            observed[mark]+=1;marks.append(mark);at.append(row)
            held[32]=np.log1p(min(len(marks),20000))/np.log(20001)
        held[33]=np.log1p(min(row-(at[-1] if at else 0),10000))/np.log(10001)
        prev=x.copy();out.append(held.copy())
    return np.array(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--native',type=Path,default=HERE/'native_transition/build/v3');ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();sys.path.insert(0,str(args.native));sys.path.insert(0,str(TRANS));sys.path.insert(0,str(ROOT/'src'))
    import wnn_cpp
    from features import batch_features
    import torch
    from model import FeatureHeadModel
    from wnn.nn import RNNModel
    from wnn.features import event_features_torch
    torch.set_num_threads(1)
    rng=np.random.default_rng(333611)
    nuisance={'centers':[-.1,.2,-.3,.4],'pi':rng.dirichlet(np.ones(16)).tolist(),'R':rng.dirichlet(np.ones(16),size=16).tolist()}
    raw=rng.normal(size=(20003,4)).astype(np.float32)
    raw[::7]=0;raw[1::11]=raw[::11][:len(raw[1::11])]
    edge=np.array([[0,0,0,0],[-0.,0,0,0],[1e-6,0,0,0],[np.nextafter(np.float32(1e-6),np.float32(np.inf)),0,0,0],[5.19,0,1,0],[np.nextafter(np.float32(5.19),np.float32(0)),0,1,0],[np.nan,0,1,0],[1,2,3,4],[1,2,3,4],[np.inf,0,0,1],[1,2,3,4],[.1,.1,.1,.1],[.2,.3,.4,.5]],np.float32)
    raw[:len(edge)]=edge
    results={}
    for arm in ['F','T']:
        config={'schema':'transhead01-68-v1','arm':arm,'nuisance':nuisance}
        expect=oracle(raw,nuisance,arm);batch=batch_features(raw,nuisance,arm)
        state=wnn_cpp.TransitionFeatures(config);got=np.array([state.update(x) for x in raw])
        error=float(np.max(np.abs(expect-got)));berror=float(np.max(np.abs(expect-batch)))
        assert error<=1e-7 and berror<=1e-7,(error,berror)
        state.reset();rerun=np.array([state.update(x) for x in raw[:407]])
        assert np.array_equal(rerun,got[:407])
        changed=raw.copy();changed[407:]=rng.normal(size=changed[407:].shape)
        assert np.array_equal(batch_features(changed,nuisance,arm)[:407],batch[:407])
        results[arm]={'native_feature_max_abs':error,'actor_batch_max_abs':berror,'events':int(got[:,34].sum()),'reset_and_prefix_exact':True}
    # Nonzero extra weights distinguish real channel use from zero-weight identity.
    torch.manual_seed(99913)
    model=FeatureHeadModel(211,hidden=128,layers=2,n_out=23).eval()
    model.head_extra.weight.data.normal_(0,.03)
    x=rng.normal(size=(517,112)).astype(np.float32)*.5
    x[0]=0;x[34:71,48:52]=0
    xr=torch.from_numpy(x)[None]
    with torch.no_grad():baseinputs=torch.cat([xr,event_features_torch(xr)],dim=-1)
    weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
    for arm in ['F','T']:
        config={'schema':'transhead01-68-v1','arm':arm,'nuisance':nuisance}
        feat=torch.from_numpy(batch_features(x[:,48:52],nuisance,arm))[None]
        with torch.no_grad():ref=model(torch.cat([baseinputs,feat],-1))[0][0,:,:2].numpy()
        for half,tol in [(False,2e-5),(True,.002)]:
            engine=wnn_cpp.Predictor(weights,fast_activations=True,half_weights=half,feature_config=config)
            got=engine.run(x);diff=float(np.max(np.abs(got-ref)));assert diff<=tol,(arm,half,diff)
            engine.reset();stream=np.array([engine.step(row) for row in x]);assert np.array_equal(got,stream)
            results[arm]['prediction_max_abs_half' if half else 'prediction_max_abs_fp32']=diff
    # Known one-hot feature single nonzero head path also checks channel ordering.
    single=FeatureHeadModel(211,hidden=16,layers=2,n_out=23).eval()
    for p in single.parameters():p.data.zero_()
    single.head_extra.weight.data[0,34]=1.7
    single.head[2].weight.data[0,0]=1
    config={'schema':'transhead01-68-v1','arm':'T','nuisance':nuisance}
    sw={k:v.detach().numpy() for k,v in single.state_dict().items()}
    ge=wnn_cpp.Predictor(sw,fast_activations=False,feature_config=config).run(x)
    event=batch_features(x[:,48:52],nuisance,'T')[:,34]
    expected=torch.nn.functional.gelu(torch.tensor(event*1.7)).numpy()
    assert np.max(np.abs(ge[:,0]-expected))<2e-6 and np.all(ge[:,1]==0)
    # Short synthetic per-row benchmark, fixed model sizes; ratio is a feasibility proxy.
    plain={k:v for k,v in weights.items() if k!='head_extra.weight'}
    cost={};bx=np.tile(x,(8,1))
    for name,members,cfgs in [('two_plain',[plain,plain],[None,None]),('two_extra',[weights,weights],[config,config])]:
        values=[]
        for repeat in range(3):
            ens=wnn_cpp.Ensemble(members,[.5,.5],half_weights=True,feature_configs=cfgs)
            t=time.perf_counter()
            for row in bx:ens.step(row)
            values.append((time.perf_counter()-t)*1e6/len(bx))
        cost[name]=values
    result={'status':'PASS','checks':results,'known_event_channel_nonzero_path':'PASS','timing_us_row':cost,'median_ratio':float(np.median(cost['two_extra'])/np.median(cost['two_plain'])),'torch':torch.__version__,'scope':'synthetic correctness/CPU timing only; no target reads, empirical fits or WP','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'feature_sha256':hashlib.sha256((TRANS/'features.py').read_bytes()).hexdigest(),'native_binary_sha256':hashlib.sha256(Path(wnn_cpp.__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
