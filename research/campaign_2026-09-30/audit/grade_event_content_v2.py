"""Independent D1 equations and serial gradient/long-sequence proof."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent/'event_content'
CASES=[]


def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()


def serial(model:nn.Module,x:torch.Tensor,state:tuple|None=None,scaled:bool=True)->tuple:
    """Independent direct recurrence, no actor state/position/reference helpers."""
    b=x.shape[0];dtype=model.mean.dtype
    if state is None:
        h=torch.zeros(1,b,96,dtype=dtype);s=torch.zeros(b,4,8,16,dtype=dtype);z=torch.zeros(b,4,8,dtype=dtype)
        row=torch.zeros(b,dtype=torch.long);count=row.clone();previous=torch.zeros(b,22,dtype=torch.float32)
    else:h,s,z,row,count,previous=state
    outputs=[]
    for t in range(x.shape[1]):
        prices=x[:,t,:22].float()
        mark=x[:,t,110].float()==torch.tensor(-5.199337482452393,dtype=torch.float32)
        event=((prices!=previous).any(-1)&(row>0)) if model.arm=='raw-code-content' else mark
        count=count+event.long()
        encoded=F.gelu(F.linear((x[:,t]-model.mean)/model.std,model.inp.weight,model.inp.bias))
        hidden,h=model.gru(encoded.unsqueeze(1),h);hidden=hidden[:,0]
        if model.arm=='endpoint-position':
            ordinals=[row,row,torch.maximum(count-1,torch.zeros_like(count)),torch.maximum(count-1,torch.zeros_like(count))]
            sources=torch.stack([torch.tensor([[1-2*((int(n)%128&j).bit_count()%2) for j in range(96)] for n in ordinal],dtype=dtype) for ordinal in ordinals],1)
        else:sources=hidden.unsqueeze(1).expand(-1,4,-1)
        q=torch.stack([F.elu(F.linear(sources[:,j],model.query.weight[8*j:8*j+8]).clamp(-8,8))+1 for j in range(4)],1)
        k=torch.stack([F.elu(F.linear(sources[:,j],model.key.weight[8*j:8*j+8]).clamp(-8,8))+1 for j in range(4)],1)
        value=F.linear(hidden,model.value.weight,model.value.bias).tanh().reshape(b,4,16)
        active=torch.stack([torch.ones_like(event),torch.ones_like(event),event,event],1).to(dtype)
        alpha=model.base_decay.to(dtype)*active+(1-active)
        scale=model.norm_scale.to(dtype) if scaled else torch.ones(4,dtype=dtype)
        s=alpha[:,:,None,None]*s+active[:,:,None,None]*scale[None,:,None,None]*k[:,:,:,None]*value[:,:,None,:]
        z=alpha[:,:,None]*z+active[:,:,None]*scale[None,:,None]*k
        retrieval=(s*q[:,:,:,None]).sum(2)/torch.maximum((z*q).sum(2),scale[None,:]*1e-6)[:,:,None]
        read=F.linear(hidden,model.read_gate.weight,model.read_gate.bias).sigmoid()*retrieval.flatten(1)
        outputs.append(model.head(torch.cat([hidden,encoded,read],-1)))
        row=row+1;previous=prices
    return torch.stack(outputs,1),(h,s,z,row,count,previous)


def check(name:str,fn)->None:
    """Preserve every numerical failure and continue independent cases."""
    try:CASES.append({'name':name,'pass':True,**fn()})
    except Exception as error:CASES.append({'name':name,'pass':False,'error':str(error),'traceback':traceback.format_exc()})
    print(json.dumps(CASES[-1]),flush=True)


def compare(a:torch.Tensor,b:torch.Tensor,tol:float)->float:
    maximum=float((a-b).abs().max()) if a.numel() else 0.
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert maximum<=tol,f'maximum absolute error {maximum} > {tol}'
    return maximum


def states(a:tuple,b:tuple,tol:float)->float:
    worst=max(compare(a[i],b[i],tol) for i in range(3))
    for i in range(3,6):assert torch.equal(a[i],b[i]),f'flat state coordinate{i} mismatch'
    return worst


def sample(length:int,batch:int=2,dtype=torch.float32)->torch.Tensor:
    x=torch.randn(batch,length,211,dtype=dtype)*.3
    x[:,:,110]=0
    for t in [0,1,17,63,64,127,128,129]:
        if t<length:x[:,t,110]=-5.199337482452393
    return x


def main()->None:
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(12973)
    contract=json.loads((HERE/'event_content_grade_contract_v2.json').read_text())
    assert sha(ACTOR/'model.py')==contract['model_sha256'] and sha(ACTOR/'preregistration.json')==contract['preregistration_sha256']
    scope=os.environ['WNN_SCOPE_UNIT']+'.scope'
    prop=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show',scope,'-p','MemoryMax','-p','MemorySwapMax','-p','CPUQuotaPerSecUSec','-p','Slice'],text=True).splitlines())
    assert prop=={'MemoryMax':'2147483648','MemorySwapMax':'0','CPUQuotaPerSecUSec':'1s','Slice':'wnn-campaign.slice'}
    spec=importlib.util.spec_from_file_location('frozen_event_model',ACTOR/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def model(arm,dtype=torch.float32):
        torch.manual_seed(534);return module.EventContentModel(211,23,arm).to(dtype)
    for arm in module.ARMS:
        for dtype in [torch.float64,torch.float32]:
            def gradient_case(arm=arm,dtype=dtype):
                a=model(arm,dtype);b=copy.deepcopy(a);x=sample(37,dtype=dtype).requires_grad_();y=x.detach().clone().requires_grad_()
                st=list(a.initial_state(x));st[0]=torch.randn_like(st[0])*.05
                st[1]=torch.randn_like(st[1])*.05;st[2]=torch.rand_like(st[2])+.5;st[3][:]=13;st[4][:]=4;st[5]=x[:,0,:22].detach().float().clone()
                sa=tuple(t.detach().clone().requires_grad_(True) if i<3 else t.clone() for i,t in enumerate(st));sb=tuple(t.detach().clone().requires_grad_(True) if i<3 else t.clone() for i,t in enumerate(st))
                oa,fa=a(x,sa);ob,fb=serial(b,y,sb)
                ftol=1e-9 if dtype==torch.float64 else 3e-5;gtol=1e-8 if dtype==torch.float64 else .0003
                e=compare(oa,ob,ftol);se=states(fa,fb,1e-9 if dtype==torch.float64 else .003)
                weights=torch.randn_like(oa);loss=lambda o,s:(o*weights).sum()+sum((v*v).mean() for v in s[:3])
                ga=torch.autograd.grad(loss(oa,fa),(x,*sa[:3],*a.parameters()));gb=torch.autograd.grad(loss(ob,fb),(y,*sb[:3],*b.parameters()))
                ge=max(compare(v,w,gtol) for v,w in zip(ga,gb))
                return {'forward_max':e,'state_max':se,'gradient_max':ge,'dtype':str(dtype)}
            check(f'gradient_{arm}_{dtype}',gradient_case)
        def partitions(arm=arm):
            a=model(arm);x=sample(211)
            with torch.no_grad():
                full,fs=a(x);worst=0.
                for size in [1,17,64,2000]:
                    out=[];st=None
                    for first in range(0,len(x[0]),size):o,st=a(x[:,first:first+size],st);out.append(o)
                    worst=max(worst,compare(full,torch.cat(out,1),3e-5));states(fs,st,.003)
                altered=x.clone();altered[:,113:]=torch.randn_like(altered[:,113:])*3
                future,_=a(altered);compare(full[:,:113],future[:,:113],3e-5)
                again,reset=a(x,None);assert torch.equal(full,again)
                for i in range(2):o,st=a(x[i:i+1]);compare(full[i:i+1],o,3e-5)
            return {'chunk_max':worst,'future_prefix_exact':True,'reset_exact':True,'batch_isolation':True}
        check(f'partitions_causality_reset_{arm}',partitions)
        def boundary(arm=arm):
            a=model(arm);x=torch.zeros(1,130,211);endpoint=torch.tensor(-5.199337482452393,dtype=torch.float32)
            x[:,[1,63,64,127,128],110]=endpoint;x[:,[0,17,129],110]=torch.nextafter(endpoint,torch.tensor(0.))
            x[:,65:,0]=.00001;x[:,129:,21]=.7
            a.mean[110]=200.;a.std[110]=7.
            with torch.no_grad():
                _,s=a(x);_,r=serial(a,x);states(s,r,.003)
                expected=2 if arm=='raw-code-content' else 5
                assert int(s[4])==expected
                empty=torch.zeros(1,1,211);_,zero=a(empty);assert torch.count_nonzero(zero[1][:,2:])==0 and torch.count_nonzero(zero[2][:,2:])==0
                empty[0,0,110]=endpoint;_,origin=a(empty)
                assert int(origin[4])==(0 if arm=='raw-code-content' else 1)
            return {'expected_writes':expected,'origin_no_artificial_write':True,'source_predicate_before_normalization':True}
        check(f'boundaries_{arm}',boundary)
        def long_case(arm=arm):
            a=model(arm);quiet=torch.zeros(1,20000,211);quiet[0,1,110]=-5.199337482452393;quiet[:,1:,0]=.2
            with torch.no_grad():
                o,s=a(quiet);r,rs=serial(a,quiet);e=compare(o,r,3e-5);se=states(s,rs,.003)
                assert int(s[3])==20000 and int(s[4])==1
                # Also test long genuine eventful recurrence, absent from author proof.
                eventful=torch.zeros(1,20000,211);eventful[:,:,110]=-5.199337482452393;eventful[0,:,0]=torch.arange(20000)%2
                eo,es=a(eventful);er,ers=serial(a,eventful);ee=compare(eo,er,3e-5);ese=states(es,ers,.003)
                assert int(es[4])==(19999 if arm=='raw-code-content' else 20000)
            return {'quiet_forward_max':e,'quiet_state_max':se,'eventful_forward_max':ee,'eventful_state_max':ese}
        check(f'long20k_serial_{arm}',long_case)
        def active_grad(arm=arm):
            a=model(arm);x=sample(256);x[:,:,110]=-5.199337482452393
            out,s=a(x);loss=(out*torch.randn_like(out)).sum()+sum(v.square().mean() for v in s[:3]);loss.backward()
            minima={}
            for name,p in a.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum())>1e-12,name
            for name in ['query','key']:
                g=getattr(a,name).weight.grad.reshape(4,8,96).abs().sum(1);minima[name]=float(g.min());assert g.min()>1e-12
            return {'column_gradient_min':minima,'all_parameters_active':True}
        check(f'active_grad_{arm}',active_grad)
        def bounds_amp(arm=arm):
            a=model(arm);a.read_gate.weight.data.zero_();a.read_gate.bias.data.zero_();a.head=nn.Identity();x=sample(257)
            with torch.no_grad():
                out,s=a(x);reference,rs=serial(a,x);compare(out,reference,3e-5)
                retrieval=2*out[:,:,-64:];assert retrieval.abs().max()<=1+3e-5
                # Vary values across time; positive normalized memory must remain
                # inside each observed head/channel envelope, including zero.
                encoded=F.gelu(a.inp((x-a.mean)/a.std));h,_=a.gru(encoded);values=a.value(h).tanh()
                lower=torch.minimum(values.cummin(1).values,torch.zeros_like(values));upper=torch.maximum(values.cummax(1).values,torch.zeros_like(values))
                assert torch.all(retrieval>=lower-3e-5) and torch.all(retrieval<=upper+3e-5)
                limit=torch.full((4,),9.)
                assert torch.all(s[2]>=0) and torch.all(s[2]<=limit[None,:,None]+.003)
                assert torch.all(s[1].abs()<=limit[None,:,None,None]+.003)
                a.query.weight.data.fill_(1000);a.key.weight.data.fill_(-1000);extreme,st=a(x);assert torch.isfinite(extreme).all() and extreme[:,:,-64:].abs().max()<=.5+3e-5
                with torch.autocast('cpu',dtype=torch.bfloat16):autocast,ast=a(x)
                assert ast[1].dtype==ast[2].dtype==torch.float32 and torch.isfinite(autocast).all()
            return {'variable_value_convex_hull':True,'memory_bounds':True,'extreme_clamped_logits_finite':True,'autocast_fp32_memory':True}
        check(f'bounds_amp_{arm}',bounds_amp)
        def equivalence(arm=arm):
            a=model(arm,torch.float64);x=sample(19,dtype=torch.float64)
            original=list(a.initial_state(x));original[1]=torch.randn_like(original[1])*.001;original[2]=torch.rand_like(original[2])*.02
            original[3][:]=131;original[4][:]=127
            scaled=list(original);scaled[1]=original[1]*a.norm_scale[None,:,None,None];scaled[2]=original[2]*a.norm_scale[None,:,None]
            with torch.no_grad():
                parallel,ps=a(x,tuple(scaled));unscaled,us=serial(a,x,tuple(original),scaled=False)
                err=compare(parallel,unscaled,1e-9)
                compare(ps[1],us[1]*a.norm_scale[None,:,None,None],1e-9)
                compare(ps[2],us[2]*a.norm_scale[None,:,None],1e-9)
                # Independent algebraic ratio/floor case exercises denominators
                # above AND below floor for every head with varied values.
                q=torch.rand(2,4,8,dtype=torch.float64)+.001
                z=torch.rand(2,4,8,dtype=torch.float64);z[0]*=1e-12
                s=z[:,:,:,None]*torch.rand(2,4,8,16,dtype=torch.float64)*.7
                numerator=(q[:,:,:,None]*s).sum(2);denominator=(q*z).sum(2)
                old=numerator/denominator.clamp_min(1e-6)[:,:,None]
                c=a.norm_scale[None,:]
                new=(numerator*c[:,:,None])/torch.maximum(denominator*c,c*1e-6)[:,:,None]
                floor_err=compare(old,new,1e-9)
                assert torch.all(denominator[0]<1e-6) and torch.all(denominator[1]>1e-6)
            return {'model_vs_unscaled_real_reference_max':err,'scaled_floor_equivalence_max':floor_err,'old_state_grammar_converted_explicitly':True}
        check(f'real_ratio_scaled_state_equivalence_{arm}',equivalence)
    def walsh():
        independent=torch.tensor([[1-2*((n&j).bit_count()%2) for j in range(96)] for n in range(128)],dtype=torch.float64)
        assert torch.equal(independent.T@independent,torch.eye(96,dtype=torch.float64)*128)
        assert torch.equal(module.walsh_table().double(),independent)
        assert torch.linalg.matrix_rank(independent)==96
        return {'rank':96,'gram_exact':True,'period':128,'actual_TRAIN_exposure_measured':False}
    check('walsh_rank',walsh)
    assert not torch.cuda.is_initialized()
    result={'status':'pass' if all(c['pass'] for c in CASES) else 'fail','cases':CASES,'contract_sha256':sha(HERE/'event_content_grade_contract_v2.json'),'model_sha256':sha(ACTOR/'model.py'),'test_source_sha256':sha(Path(__file__)),'scope_properties':prop,'cuda_initialized':False,'competition_data_read':False,'scorer_mutated':False,'wall_seconds':time.monotonic()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    with (HERE/'event_content_grade_v2.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
