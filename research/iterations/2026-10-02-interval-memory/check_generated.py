"""Generated-only CPU readiness: independent literal stream and parameter VJP."""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
import numpy as np
import torch
from torch.nn import functional as F
from wnn.nn import RNNModel
from interval_memory import IntervalMemoryModel, interval_means


def literal(model, x, events, state=None):
    """Independent row loop; no interval helper or event packing/gather."""
    z=model.encode((x-model.mean)/model.std)
    b,t,_=z.shape
    if state is None:
        row=None;slow=z.new_zeros(b,64);total=z.new_zeros(b,32)
        count=torch.zeros(b,dtype=torch.long);events=events.clone();events[:,0]=False
    else:row,slow,total,count=state
    h,row=model.gru(z,row);v=torch.tanh(model.interval_proj(h));outputs=[]
    for i in range(t):
        total=total+v[:,i];count=count+1
        u=total/count[:,None] if model.arm=='accumulated' else v[:,i]
        # Per-stream one-event call; no padded future events can enter state.
        slow_rows=[]
        for k in range(b):
            if bool(events[k,i]):
                inp=torch.cat([u[k],torch.log1p(count[k].to(z.dtype))[None]])
                _,s=model.event_gru(inp[None,None],slow[k][None,None])
                slow_rows.append(s[0,0])
            else:slow_rows.append(slow[k])
        slow=torch.stack(slow_rows)
        pre=model.head[0](torch.cat([h[:,i],z[:,i]],-1))
        pre=pre+F.linear(torch.cat([u,slow],-1),model.extra_weight)
        outputs.append(model.head[2](model.head[1](pre)))
        total=torch.where(events[:,i,None],torch.zeros_like(total),total)
        count=torch.where(events[:,i],torch.zeros_like(count),count)
    return torch.stack(outputs,1),(row,slow,total,count)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    torch.set_num_threads(1);torch.manual_seed(20261002);start=time.monotonic();checks=[]
    result=dict(status='RUNNING',real_data=False,checkpoints_read=False,forecast_fits=0,GPU=False)
    def close(name,left,right,atol=2e-11,rtol=2e-9):
        torch.testing.assert_close(left,right,atol=atol,rtol=rtol)
        checks.append(dict(name=name,max_abs=float((left-right).abs().max()) if left.numel() else 0.))
    try:
        rng=torch.get_rng_state();candidate=IntervalMemoryModel('accumulated');after=torch.get_rng_state()
        torch.set_rng_state(rng);control=IntervalMemoryModel('instantaneous')
        assert torch.equal(after,torch.get_rng_state())
        assert candidate.state_dict().keys()==control.state_dict().keys()
        assert all(torch.equal(v,control.state_dict()[k]) for k,v in candidate.state_dict().items())
        plain=RNNModel(211,256,1,0.,n_out=23)
        with torch.no_grad():plain.mean.normal_(0,.1);plain.std.uniform_(.5,2.)
        candidate.load_plain_baseline(plain.state_dict());control.load_plain_baseline(plain.state_dict())
        x=torch.randn(2,29,211);events=torch.zeros(2,29,dtype=torch.bool)
        events[0,[3,4,17,28]]=True;events[1,[0,9]]=True
        x[...,178]=events.to(x.dtype)
        baseline,bh=plain(x)
        for name,m in [('candidate',candidate),('control',control)]:
            out,state=m(x)
            assert torch.equal(out,baseline) and torch.equal(state[0],bh)
            assert torch.equal(m.mean,plain.mean) and torch.equal(m.std,plain.std)
            checks.append(dict(name=name+'_zero_extra_exact_baseline',max_abs=0.))
        try:candidate.load_plain_baseline({k:v for k,v in plain.state_dict().items() if k!='mean'})
        except ValueError:pass
        else:raise AssertionError('partial warmstart was accepted')
        # Nonzero extra weights make feature/state errors visible at the output.
        for name,initial in [('candidate',candidate),('control',control)]:
            model=copy.deepcopy(initial).double()
            with torch.no_grad():model.extra_weight.normal_(0,.025)
            ref=copy.deepcopy(model)
            xx=x.double().requires_grad_();xr=x.double().requires_grad_()
            out,state=model(xx);expected,es=literal(ref,xr,events)
            close(name+'_literal_outputs',out,expected)
            for i,(u,v) in enumerate(zip(state,es)):close(name+'_literal_state'+str(i),u,v)
            loss=out.square().mean()+.013*state[1].square().mean()+.007*state[2].square().mean()
            el=expected.square().mean()+.013*es[1].square().mean()+.007*es[2].square().mean()
            grads=torch.autograd.grad(loss,[xx,*model.parameters()]);eg=torch.autograd.grad(el,[xr,*ref.parameters()])
            for i,(g,r) in enumerate(zip(grads,eg)):close(name+'_VJP'+str(i),g,r,atol=5e-11)
            with torch.no_grad():
                inp=x.double();allout,alls=model(inp)
                carry=None;parts=[]
                for lo,hi in [(0,1),(1,4),(4,5),(5,5),(5,13),(13,29)]:
                    p,carry=model(inp[:,lo:hi],carry);parts.append(p)
                close(name+'_arbitrary_chunks',torch.cat(parts,1),allout)
                for i,(u,v) in enumerate(zip(carry,alls)):close(name+'_chunk_state'+str(i),u,v)
                one=None;parts=[]
                for i in range(29):p,one=model(inp[:,i:i+1],one);parts.append(p)
                close(name+'_one_row',torch.cat(parts,1),allout)
                for end in (1,4,10,21):close(name+'_prefix'+str(end),model(inp[:,:end])[0],allout[:,:end])
                future=inp.clone();future[:,11:]+=4;close(name+'_future_causality',model(future)[0][:,:11],allout[:,:11])
                close(name+'_sequence_reset',model(inp)[0],allout)
        # Event input witness: completed mean != instantaneous value, held values
        # and count survive arbitrary boundaries; no model head can hide this.
        values=torch.tensor([[[1.,0.],[3.,2.],[8.,4.],[6.,8.]]],dtype=torch.float64)
        ev=torch.tensor([[False,False,True,False]])
        means,ts,n=interval_means(values,ev,torch.zeros(1,2,dtype=torch.float64),torch.zeros(1,dtype=torch.long))
        close('event_closes_inclusive_interval',means[:,2],torch.tensor([[4.,2.]],dtype=torch.float64))
        assert not torch.equal(means[:,2],values[:,2]);close('after_event_new_interval',means[:,3],values[:,3])
        assert int(n[0])==1;close('after_event_carry',ts,values[:,3])
        # Long quiet20k including a carry-in; measure FP32 scan drift, not bit parity.
        gen=np.random.default_rng(9);vv=gen.uniform(-1,1,(1,20000,32)).astype(np.float32)
        vv[...,0]=.999;inc=np.full((1,32),.125,np.float32);counter=37
        sums=inc.copy();oracle=np.empty_like(vv)
        for i in range(20000):sums+=vv[:,i];oracle[:,i]=sums/(counter+i+1)
        vals=torch.from_numpy(vv);ev=torch.zeros(1,20000,dtype=torch.bool)
        mean,total,count=interval_means(vals,ev,torch.from_numpy(inc),torch.tensor([counter]))
        close('longquiet_FP32_literal',mean,torch.from_numpy(oracle),atol=4e-4,rtol=1e-4)
        carry=torch.from_numpy(inc);ct=torch.tensor([counter]);parts=[]
        for lo,hi in [(0,7),(7,2000),(2000,8193),(8193,20000)]:
            mm,carry,ct=interval_means(vals[:,lo:hi],ev[:,lo:hi],carry,ct);parts.append(mm)
        close('longquiet_FP32_chunks',torch.cat(parts,1),mean,atol=4e-4,rtol=1e-4)
        assert int(count[0])==int(ct[0])==20037
        # Frozen-H2 roles: only existing head and extra readout may change.
        model=copy.deepcopy(candidate);model.freeze_trunk();before={k:v.clone() for k,v in model.state_dict().items()}
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        assert set(names)=={'extra_weight','head.0.weight','head.0.bias','head.2.weight','head.2.bias'}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        out,_=model(x);out.square().mean().backward();optimizer.step()
        frozen=[k for k in before if not k.startswith('head.') and k!='extra_weight']
        assert all(torch.equal(before[k],model.state_dict()[k]) for k in frozen)
        assert not torch.equal(before['extra_weight'],model.extra_weight)
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        result.update(status='PASS_GENERATED_INTERVAL_CPU',checks=checks,matched_constructor_RNG_and_parameters=True,
            strict_plain_warmstart=True,baseline_initial_outputs_exact=True,H2_frozen_tensors=frozen,H2_trainable=names,
            parameters=sum(p.numel() for p in model.parameters()),ordinary_parameters=sum(p.numel() for p in plain.parameters()),
            interval_FP32_policy='4e-4 absolute +1e-4 relative numerical envelope for generated20k; observed errors separately recorded, not a forecast utility criterion',
            limitations='CPU only; no CUDA AMP, GPU memory/runtime, native parity, forecast data or fit. FP32 prefix-sum grouping need not equal streaming bits.')
    except BaseException as e:
        result.update(status='FAILED_GENERATED_NO_FORECAST',failure=repr(e));raise
    finally:
        result.update(seconds=time.monotonic()-start,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'interval_memory.py',Path(__file__))})
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result),flush=True)
if __name__=='__main__':main()
