"""Independent CPU/synthetic TRANSHEAD review. No competition dataset or fitted model read."""
from __future__ import annotations
import hashlib, importlib, json, math, sys, tempfile, types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ACTOR=HERE.parent/'transition'
sys.path[:0]=[str(ACTOR),str(ROOT/'src'),str(ROOT/'scripts')]
feat=importlib.import_module('features')
fit=importlib.import_module('fit_nuisance')
mod=importlib.import_module('model')
trainer=importlib.import_module('train_adapter')
predictor=importlib.import_module('predict_adapter')
from wnn.nn import RNNModel


def oracle(raw, nuisance, arm):
    """Recompute all history from lists, no actor detector/encoder/counting calls."""
    x=np.asarray(raw,dtype=np.float32)
    center=list(map(float,nuisance['centers']));prior=np.array(nuisance['pi'],np.float64);matrix=np.array(nuisance['R'],np.float64)
    records=[];marks=[];last_event=0;held=[0.]*68
    held[35:51]=prior.tolist()
    for row in range(len(x)):
        now=list(x[row]);old=None if row==0 else list(x[row-1])
        valid=lambda vals: vals is not None and all(math.isfinite(float(v)) and abs(v)<np.float32(5.19) for v in vals)
        event=valid(now) and valid(old) and any(a!=b for a,b in zip(now,old)) and np.float32(max(now)-min(now))>np.float32(1e-6)
        held[34]=float(event)
        if event:
            mark=sum((1<<j) for j,val in enumerate(now) if float(val)>center[j])
            if arm=='F': history=marks;initial=prior
            elif not marks:history=[];initial=prior
            else:
                history=[b for a,b in zip(marks[:-1],marks[1:]) if a==marks[-1]]
                initial=matrix[marks[-1]]
            counts=np.array([history.count(category) for category in range(16)],np.float64)
            p=(counts+16*initial)/(len(history)+16)
            held[16:32]=[float(bool(marks) and category==marks[-1]) for category in range(16)]
            held[:16]=[float(category==mark) for category in range(16)]
            held[35:51]=p.tolist();held[51:67]=[float(category==mark)-p[category] for category in range(16)]
            held[67]=min(-math.log(p[mark]),10)/10
            marks.append(mark);held[32]=math.log1p(min(len(marks),20000))/math.log(20001)
            last_event=row
        held[33]=math.log1p(min(row-last_event,10000))/math.log(10001)
        records.append(held.copy())
    return np.array(records,dtype=np.float32).reshape(len(x),68)


def feature_checks():
    rng=np.random.default_rng(898123);pi=rng.random(16)+.001;pi/=pi.sum();R=rng.random((16,16))+.001;R/=R.sum(1,keepdims=True)
    nuisance={'centers':[0.,.1,-.15,.2],'pi':pi.tolist(),'R':R.tolist()}
    pathological=np.array([[0,0,0,0],[1,2,3,4],[1.1,2,3,4],[2,3,4,1],[0,0,0,0],[2,3,4,1],[np.nan,2,3,4],[2,3,4,1],[3,2,4,1],[0,-0.,0,0],[-0.,0,0,0],[0,0,0,np.float32(1e-6)],[0,0,0,np.nextafter(np.float32(1e-6),np.float32(1))],[5.19,2,3,4],[np.nextafter(np.float32(5.19),np.float32(0)),2,3,4],[4,3,2,1]],np.float32)
    random=rng.uniform(-5.5,5.5,(420,4)).astype(np.float32);random[::7]=0;random[40:44]=random[39];random[100,0]=np.inf
    raw=np.concatenate([pathological,random]); results={}
    for arm in ('F','T'):
        want=oracle(raw,nuisance,arm);got=feat.batch_features(raw,nuisance,arm)
        np.testing.assert_allclose(got,want,rtol=0,atol=1e-7)
        state=feat.StreamingFeatures(nuisance,arm);stream=np.array([state.update(v) for v in raw]);np.testing.assert_allclose(stream,want,rtol=0,atol=1e-7)
        # All prefix cuts over pathologies and independent later suffix changes.
        for cut in list(range(18))+[111,173,len(raw)]:
            np.testing.assert_array_equal(feat.batch_features(raw[:cut],nuisance,arm),got[:cut])
        changed=raw.copy();changed[43:]=rng.normal(size=changed[43:].shape);np.testing.assert_array_equal(feat.batch_features(changed,nuisance,arm)[:43],got[:43])
        state.reset();again=np.array([state.update(v) for v in raw]);np.testing.assert_array_equal(again,stream)
        assert list(np.flatnonzero(got[:4,34]))==[1,2,3]
        assert int(got[2,:16].argmax())==int(got[1,:16].argmax())==15
        # Second same-category event sees no 15->15 count until after it is encoded.
        if arm=='T':
            np.testing.assert_allclose(got[2,35:51],R[15],rtol=0,atol=3e-8)
            np.testing.assert_allclose(got[3,35:51],(16*R[15]+np.eye(16)[15])/17,rtol=0,atol=3e-8)
        assert np.isfinite(got).all();assert np.max(abs(got[:,35:51].sum(1)-1))<2e-7
        results[arm]={'rows':len(raw),'events':int(got[:,34].sum()),'max_oracle_abs':float(abs(got-want).max())}
    np.testing.assert_array_equal(feat.batch_features(raw,nuisance,'F')[:,:35],feat.batch_features(raw,nuisance,'T')[:,:35])
    return results,nuisance


def nuisance_checks():
    cf={'split':np.array(['valid','valid','valid','valid','train','valid','valid']), 'fold':np.array([0,1,1,2,1,2,0]),'chain':np.array([10,20,20,30,30,30,40]),'row_group':np.array([8,10,3,6,1,4,9])}
    assert fit.select_groups(cf,0)==[{'chain':20,'row_group':3},{'chain':30,'row_group':4}]
    assert fit.select_groups(cf,-1)==[{'chain':10,'row_group':8},{'chain':20,'row_group':3},{'chain':30,'row_group':4},{'chain':40,'row_group':9}]
    bad={k:v.copy() for k,v in cf.items()};bad['chain'][0]=20
    try:fit.select_groups(bad,0)
    except ValueError:pass
    else:raise AssertionError('cross-fold chain overlap not rejected')
    # Centers from replicated rows must respect equal sequence mass; priors checked from explicit loops.
    sequences=[np.array([[1,2,3,4]]*100,np.float32),np.array([[-2,-1,-3,-4]],np.float32),np.empty((0,4),np.float32)]
    fitted=fit.fit_values(sequences)
    # Exact left weighted median is defined by source summation; verify legal coordinates and priors independently.
    center=np.array(fitted['centers']);codes=[]
    for seq in sequences:codes.append([sum(1<<j for j,v in enumerate(row) if v>center[j]) for row in seq])
    freq=np.zeros(16);trans=np.zeros((16,16));n=0
    for marks in codes:
        if marks:
            n+=1
            for v in marks:freq[v]+=1/len(marks)
        if len(marks)>1:
            for a,b in zip(marks[:-1],marks[1:]):trans[a,b]+=1/(len(marks)-1)
    pi=(freq+1/16)/(n+1);R=(trans+pi)/(trans.sum(1,keepdims=True)+1)
    np.testing.assert_allclose(fitted['pi'],pi,rtol=0,atol=1e-14);np.testing.assert_allclose(fitted['R'],R,rtol=0,atol=1e-14)
    return {'chain_overlap_rejected':True,'min_valid_rg_per_allowed_chain':True,'train_rows_excluded':True,'sequence_equal_priors_scalar_oracle':True}


def model_checks(nuisance):
    torch.manual_seed(831);base=RNNModel(211,hidden=9,layers=2,n_out=23);want_rng=torch.get_rng_state().clone()
    torch.manual_seed(831);model=mod.FeatureHeadModel(211,hidden=9,layers=2,n_out=23);assert torch.equal(want_rng,torch.get_rng_state())
    model.load_state_dict(base.state_dict(),strict=True)
    raw=torch.randn(2,33,112);extra=torch.tensor(np.stack([feat.batch_features(a[:,48:52],nuisance,'T') for a in raw.numpy()]))
    full=trainer.base.build_inputs(torch.cat([raw,extra],-1));plain=trainer.base.build_inputs(raw)
    assert torch.equal(full[:,:,:211],plain);assert torch.equal(full[:,:,211:],extra)
    y,h=base(plain);out,state=model(full);assert torch.equal(y,out) and torch.equal(h,state)
    for name,p in model.named_parameters():
        if name.startswith(('inp','gru')):p.requires_grad_(False)
    frozen={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('inp','gru'))}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0005)
    opt.zero_grad(set_to_none=True);model(full)[0].square().mean().backward()
    assert model.head_extra.weight.grad.abs().sum()>0 and model.head[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for n,p in model.named_parameters() if n.startswith(('inp','gru')))
    opt.step();assert all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items())
    # Independent recurrent streaming partition, after adapter is nonzero.
    whole,wh=model(full);parts=[];h=None
    for begin,end in [(0,7),(7,8),(8,33)]:
        q,h=model(full[:,begin:end],h);parts.append(q)
    torch.testing.assert_close(torch.cat(parts,1),whole,rtol=1e-6,atol=1e-7);torch.testing.assert_close(h,wh,rtol=1e-6,atol=1e-7)
    loader=DataLoader(TensorDataset(torch.arange(77)),batch_size=16,num_workers=0,shuffle=False)
    torch.manual_seed(348);list(loader);want=torch.get_rng_state().clone();torch.manual_seed(348);res=trainer.skip_evaluation_preserving_rng(None,loader,None,None);assert torch.equal(want,torch.get_rng_state()) and res['evaluation_skipped']
    return {'extra_layer_rng_preserved':True,'base211_plus_extra68_layout':True,'initial_forecast_state_bitexact':True,'frozen_backbone_unchanged_after_AdamW':True,'both_head_branches_receive_gradients':True,'chunk_state_parity':True,'skipped_eval_rng_preserved':True}


def prediction_provenance_checks(nuisance):
    # Exercise actual prediction main against synthetic checkpoint/index/data/model,
    # full1873 output shape and sparse row mapping, without allocating a300MB file or CUDA.
    groups=np.arange(1873,dtype=np.int64)[::-1];chosen=[3,17,100,555,1600]
    cf={'split':np.array(['valid']*1873),'row_group':np.arange(1873),'fold':np.where(np.isin(np.arange(1873),chosen),0,1)}
    metadata={'schema':feat.SCHEMA,'source_fold':0,'arm':'T','nuisance':nuisance}
    ck={'epoch':1,'args':{'transition_features':metadata,'hidden':9,'layers':2,'dropout':0.,'valid_fold':0},'n_in':211,'model':{'head.2.weight':np.zeros((23,9))}}
    class FakeModel:
        def __init__(self,*args):pass
        def cuda(self):return self
        def eval(self):return self
        def load_state_dict(self,state):pass
    class FakeInputs:
        def __init__(self, selected, meta):self.groups=list(selected)
        def __len__(self):return len(self.groups)
        def __getitem__(self,i):return torch.tensor([self.groups[i]],dtype=torch.float32)
    writes={};sentinel=[]
    class FakeMemmap:
        def __setitem__(self,indices,values):
            if isinstance(indices,slice):assert np.isnan(values);sentinel.append(True)
            else:
                for pos,val in zip(indices,values):writes[int(pos)]=int(val[0,0])
        def flush(self):pass
    def fake_memmap(path,mode,dtype,shape):
        assert mode=='w+' and shape==(1873,20000,2) and dtype==np.float32;Path(path).write_bytes(b'synthetic full-index output');return FakeMemmap()
    def fake_load(path):
        return {'row_group':groups} if str(path).endswith('valid_targets.npz') else cf
    def fake_predict(model,x,chunk,device):
        return np.broadcast_to(x.numpy()[:,0,None,None],(len(x),20000,2)).copy()
    with tempfile.TemporaryDirectory(prefix='trans-review-') as td:
        p=Path(td);checkpoint=p/'mock.pt';checkpoint.write_bytes(b'synthetic checkpoint');output=p/'result.npy'
        argv=['predict_adapter.py','--checkpoint',str(checkpoint),'--output',str(output),'--workers','0','--batch','3']
        with patch.object(sys,'argv',argv),patch.object(predictor.torch,'load',return_value=ck),patch.object(predictor,'FeatureHeadModel',FakeModel),patch.object(predictor,'InputSequences',FakeInputs),patch.object(predictor.np,'load',side_effect=fake_load),patch.object(predictor.np.lib.format,'open_memmap',side_effect=fake_memmap),patch.object(predictor.base,'configure_torch'),patch.object(predictor.base,'predict',side_effect=fake_predict):predictor.main()
        receipt=json.loads(output.with_suffix('.json').read_text());positions=np.flatnonzero(np.isin(groups,chosen)).tolist()
        assert sentinel==[True];assert sorted(writes)==positions
        assert receipt['included_positions']==positions and receipt['row_groups']==groups[positions].tolist()
        assert all(writes[pos]==int(groups[pos]) for pos in positions)
        assert receipt['cache_row_groups_sha256']==hashlib.sha256(groups.tobytes()).hexdigest()
        assert receipt['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert receipt['epoch']==1 and receipt['WP_queries']==0 and not receipt['targets_read'] and not receipt['mask_read']
    return {'actual_main_mocked_1873_layout':True,'all_unselected_rows_NaN':True,'selected_indices_and_groups_exact':True,'checkpoint_and_row_order_hashes':True,'zero_WP_readouts':True,'test_scope':'mocked IO/CUDA, real controller/index/provenance code'}


def main():
    torch.set_num_threads(2)
    records,nuisance=feature_checks()
    result={'status':'PASS','independent_scalar_features':records,'nuisance':nuisance_checks(),'model':model_checks(nuisance),'prediction':prediction_provenance_checks(nuisance),'scope':{'empirical_data_read':False,'GPU_jobs':0,'WP_queries':0},'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ACTOR.glob('*.py'))},'review_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
