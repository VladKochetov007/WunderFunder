"""Fixed generated CPU matrix; failures retained, no coefficient/math retry."""
from pathlib import Path
import copy
import gc
import hashlib
import json
import os
import resource
import time
import numpy as np
import torch
from torch.nn import functional as F
from direct_scan import DirectMinimalRNN, direct_scan
from torch_reference import MinimalRNN, blocked_scan, streaming_states, positive_g

P=Path(__file__).resolve().parent
OUT=P/'generated-01'
checks=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def close(name,a,b,extreme=False,exact=False):
    a,b=a.detach().double(),b.detach().double()
    atol=0 if exact else (2e-4 if extreme else 2e-5)
    rtol=0 if exact else 2e-5
    delta=(a-b).abs();bound=atol+rtol*b.abs()
    passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and (delta<=bound).all())
    record=dict(name=name,passed=passed,max_abs=float(delta.max()),atol=atol,rtol=rtol,
                max_error_over_allowed=float((delta/bound.clamp_min(1e-300)).max()))
    checks.append(record)
    return record


def model_literal(model,x,h0,dtype):
    sd={k:v.to(dtype) for k,v in model.state_dict().items()}
    x=x.to(dtype);carry=[h0[i].to(dtype) for i in range(model.layers)]
    rows=[]
    for row in x.unbind(1):
        z=F.gelu(F.linear((row-sd['mean'])/sd['std'],sd['inp.weight'],sd['inp.bias']))
        value=z
        for i in range(model.layers):
            key=f'cells.{i}.'
            k=F.linear(value,sd[key+'gate.weight'],sd[key+'gate.bias'])
            v=F.linear(value,sd[key+'candidate.weight'],sd[key+'candidate.bias'])
            update=torch.sigmoid(k);carry[i]=(1-update)*carry[i]+update*positive_g(v);value=carry[i]
        hidden=F.gelu(F.linear(torch.cat((value,z),-1),sd['head.0.weight'],sd['head.0.bias']))
        rows.append(F.linear(hidden,sd['head.2.weight'],sd['head.2.bias']))
    return torch.stack(rows,1),torch.stack(carry)


def main():
    start=time.perf_counter()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and list(os.sched_getaffinity(0))==[34]
    cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[2].lstrip('/')
    limits={k:(cg/k).read_text().strip() for k in ['cpu.max','memory.max','memory.swap.max']}
    assert limits=={'cpu.max':'200000 100000','memory.max':str(4*(1<<30)),'memory.swap.max':'0'},limits
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(2045)
    k=torch.randn(2,20000,9);v=torch.randn_like(k);initial=torch.rand(2,9)*2+.1
    x=torch.randn(1,20000,211);h=torch.rand(2,1,256)*2+.1
    model=MinimalRNN(n_out=23).eval();model.mean.copy_(.2*torch.randn(211));model.std.copy_(.5+torch.rand(211))
    direct=DirectMinimalRNN(n_out=23).eval();direct.load_state_dict(model.state_dict())
    np.savez(OUT/'generated-inputs.npz',k=k.numpy(),v=v.numpy(),initial=initial.numpy(),x=x.numpy(),h=h.numpy())
    np.savez(OUT/'generated-model.npz',**{n:p.numpy() for n,p in model.state_dict().items()})
    inputs={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in [*[P/f for f in ['direct_scan.py','torch_reference.py','check_direct.py','plan.json']],*sorted(OUT.glob('generated-*.npz'))]}
    (OUT/'inputs-before-check.json').write_text(json.dumps(inputs,indent=2)+'\n')
    retention=[]
    with torch.no_grad():
        for length in [2000,20000]:
            for label,h0 in [('zero',torch.zeros_like(initial)),('positive',initial)]:
                kk,vv=k[:,:length],v[:,:length]
                a,end=direct_scan(kk,vv,h0)
                for name,fn,dtype in [('step',streaming_states,torch.float32),('oldlog',blocked_scan,torch.float32),('double',streaming_states,torch.float64)]:
                    want,last=fn(kk.to(dtype),vv.to(dtype),h0.to(dtype))
                    close(f'normal_{length}_{label}_{name}',a,want)
                    close(f'normal_end_{length}_{label}_{name}',end,last)
                pieces=[];carry=h0
                cuts=[0,1,63,64,65,511,999,length]
                for lo,hi in zip(cuts[:-1],cuts[1:]):
                    part,carry=direct_scan(kk[:,lo:hi],vv[:,lo:hi],carry);pieces.append(part)
                close(f'split_{length}_{label}',torch.cat(pieces,1),a)
                altered=vv.clone();altered[:,1001:]+=10
                future,_=direct_scan(kk,altered,h0);close(f'causal_{length}_{label}',future[:,:1001],a[:,:1001],exact=True)
            # Fixed long retention stress exposes coefficient rounding rather than concealing it.
            levels=torch.tensor([-80.,-20.,-17.,0.,20.,80.])
            kk=levels.repeat(1,length,1);vv=torch.tensor([-80.,-20.,-17.,0.,8.,80.]).repeat(1,length,1)
            for label,h0 in [('zero',torch.zeros(1,6)),('positive20',torch.full((1,6),20.))]:
                a,end=direct_scan(kk,vv,h0);old,old_end=blocked_scan(kk,vv,h0)
                step,step_end=streaming_states(kk,vv,h0);double,double_end=streaming_states(kk.double(),vv.double(),h0.double())
                for name,want in [('step',step),('oldlog',old),('double',double)]:close(f'extreme_{length}_{label}_{name}',a,want,extreme=True)
                retention.append({'T':length,'initial':label,'gate':levels.tolist(),'direct_end':end.tolist(),'step_end':step_end.tolist(),'oldlog_end':old_end.tolist(),'double_end':double_end.tolist(),
                                  'oldlog_vs_double_max_abs':float((old.double()-double).abs().max()),'step_vs_double_max_abs':float((step.double()-double).abs().max())})
        for length in [2000,20000]:
            for label,h0 in [('zero',torch.zeros_like(h)),('positive',h)]:
                xx=x[:,:length];a,end=direct(xx,h0);old,old_end=model(xx,h0)
                rows=[];carry=h0
                for row in xx.unbind(1):
                    out,carry=model.step(row,carry);rows.append(out)
                close(f'model_{length}_{label}_actual_step',a,torch.stack(rows,1));close(f'model_end_{length}_{label}_actual_step',end,carry)
                close(f'model_{length}_{label}_oldlog',a,old);close(f'model_end_{length}_{label}_oldlog',end,old_end)
        small=MinimalRNN(n_in=11,hidden=8,layers=2,n_out=23).eval()
        small_direct=DirectMinimalRNN(n_in=11,hidden=8,layers=2,n_out=23).eval();small_direct.load_state_dict(small.state_dict())
        sx=torch.randn(2,131,11);sh=torch.rand(2,2,8)+.1
        a,end=small_direct(sx,sh);double,double_end=model_literal(small,sx,sh,torch.float64)
        close('small_model_double',a,double);close('small_model_end_double',end,double_end)
    # Cell VJPs against three explicit references; both exact zero and positive h0.
    for extreme in [False,True]:
        for initial_label in ['zero','positive']:
            if extreme:
                kg=torch.tensor([-80.,-20.,-17.,0.,20.,80.,1.]).repeat(2,131,1)
                vg=torch.tensor([-80.,-8.,-.2,0.,8.,80.,.5]).repeat(2,131,1)
            else:kg=.6*torch.randn(2,131,7);vg=.6*torch.randn(2,131,7)+.07
            ig=torch.zeros(2,7) if initial_label=='zero' else torch.rand(2,7)+.2
            weight=torch.randn_like(kg)
            def cell_vjp(fn,dtype):
                args=[z.to(dtype).detach().requires_grad_() for z in [kg,vg,ig]]
                y,_=fn(*args);grad=torch.autograd.grad((y*weight.to(dtype)).sum(),args)
                return grad
            actual=cell_vjp(direct_scan,torch.float32)
            for ref,fn,dtype in [('step',streaming_states,torch.float32),('oldlog',blocked_scan,torch.float32),('double',streaming_states,torch.float64)]:
                expected=cell_vjp(fn,dtype)
                for part,a,b in zip(['gate','candidate','initial'],actual,expected):close(f'VJP_cell_{extreme}_{initial_label}_{ref}_{part}',a,b,extreme=extreme)
    # Full model VJPs use actual FP32 step plus old scan, all parameters, x and h0.
    sx=torch.randn(2,23,11);coef=torch.randn(2,23,23)
    for label,h0 in [('zero',torch.zeros(2,2,8)),('positive',torch.rand(2,2,8)+.1)]:
        def model_vjp(m,step=False):
            xx=sx.clone().requires_grad_();hh=h0.clone().requires_grad_()
            if step:
                out=[];carry=hh
                for row in xx.unbind(1):value,carry=m.step(row,carry);out.append(value)
                y=torch.stack(out,1)
            else:y,_=m(xx,hh)
            return torch.autograd.grad((y*coef).sum(),[xx,hh,*m.parameters()])
        actual=model_vjp(small_direct)
        for name,expected in [('actual_step',model_vjp(small,True)),('oldlog',model_vjp(small))]:
            for idx,(a,b) in enumerate(zip(actual,expected)):close(f'VJP_model_{label}_{name}_{idx}',a,b)
        # Independent double recurrence retains parameter autograd here, no state_dict detachment.
        dm=copy.deepcopy(small).double();xx=sx.double().requires_grad_();hh=h0.double().requires_grad_();z=F.gelu(dm.inp((xx-dm.mean)/dm.std));value=z;states=[]
        for i,cell in enumerate(dm.cells):value,last=streaming_states(cell.gate(value),cell.candidate(value),hh[i]);states.append(last)
        yy=dm.head(torch.cat((value,z),-1));expected=torch.autograd.grad((yy*coef.double()).sum(),[xx,hh,*dm.parameters()])
        for idx,(a,b) in enumerate(zip(actual,expected)):close(f'VJP_model_{label}_double_{idx}',a,b)
    # Descriptive underflow: distinguish a==0 from a positive product becoming zero.
    underflow=[]
    for level in [-80.,-20.,-17.,-10.,0.,2.,4.,20.,80.]:
        logits=torch.full((64,),level);a=1-torch.sigmoid(logits);a64=1-torch.sigmoid(logits.double());prod=torch.cumprod(a,0);prod64=torch.cumprod(a64,0)
        idx=torch.nonzero((prod==0)&(a>0)).flatten()
        underflow.append(dict(gate=level,coefficient=float(a[0]),coefficient_double=float(a64[0]),product64=float(prod[-1]),product64_double=float(prod64[-1]),first_positive_coefficient_product_zero=None if not len(idx) else int(idx[0])+1,omitted_h20_contribution_at64=float(prod64[-1]*20) if prod[-1]==0 else 0.))
    # Fixed CPU forward/backward timings, no optimizer or parameter updates.
    times={}
    for name,m in [('oldlog',model),('direct',direct)]:
        values=[]
        for repeat in range(4):
            m.zero_grad(set_to_none=True);t=time.perf_counter();yy,_=m(x[:,:2000],h);yy.square().mean().backward();elapsed=time.perf_counter()-t
            if repeat:values.append(elapsed)
        times[name]=values
    gc.collect()
    for name,rec in inputs.items():
        q=P/name if (P/name).exists() else OUT/name
        assert sha(q)==rec['sha256'],name
    failed=[r for r in checks if not r['passed']]
    result=dict(status='PASS_GENERATED_CPU_ONLY' if not failed else 'FAIL_FIXED_NUMERICAL_TOLERANCE',checks=checks,failures=failed,retention_stress=retention,coefficient_underflow=underflow,CPU_forward_backward_B1_T2000_H256_L2_O23_seconds=times,
        seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits=limits,CPU_affinity=[34],seed=2045,torch_version=torch.__version__,source_sha256={f:sha(P/f) for f in ['direct_scan.py','torch_reference.py','check_direct.py','plan.json']},input_inventory_sha256=sha(OUT/'inputs-before-check.json'),GPU_used=False,real_data_or_trained_checkpoints_read=False,forecast_fit_or_metric=False,
        limitations=['Fixed generated FP32 tests; no AMP, GPU performance, full training or forecast qualification.','Direct coefficients intentionally match literal FP32 step; saturated sigmoid/subtraction and associative rounding can differ from old log scan or real-arithmetic oracle.','Long retention stress is part of the fixed test matrix; failed checks are not removed or rescued.','Timing is CPU B1 synthetic scalar forward/backward, not GPU B32 or an optimizer step.'])
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','failures','seconds','peak_RSS_KiB','CPU_forward_backward_B1_T2000_H256_L2_O23_seconds']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        (OUT/'failure.json').write_text(json.dumps({'status':'EXECUTION_FAILURE_NO_RETRY','exception':repr(exc),'checks_completed':checks},indent=2)+'\n')
        raise
