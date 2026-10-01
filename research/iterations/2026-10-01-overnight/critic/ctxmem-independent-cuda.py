"""Bounded independent synthetic AMP/fused optimizer audit; no empirical inputs."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,hashlib,json,resource,signal,sys,time
import torch
ap=argparse.ArgumentParser();ap.add_argument('--actor',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7' and not a.output.exists()
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120-second cap')));signal.alarm(120)
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.10)
free,total=torch.cuda.mem_get_info();assert free>=6200*2**20,(free,total)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=True
sys.path.insert(0,str(a.actor.resolve()));from model import ContextMemoryModel
from wnn.nn import RNNModel
start=time.monotonic();torch.cuda.reset_peak_memory_stats()
torch.manual_seed(627193);base=RNNModel(211,128,2,0,n_out=23);after_cpu=torch.get_rng_state().clone();after_gpu=torch.cuda.get_rng_state().clone()
torch.manual_seed(627193);mem=ContextMemoryModel(arm='memory')
assert torch.equal(torch.get_rng_state(),after_cpu) and torch.equal(torch.cuda.get_rng_state(),after_gpu)
mem.load_state_dict(base.state_dict());ctl=ContextMemoryModel(arm='current');ctl.load_state_dict(base.state_dict())
models=[base,mem,ctl]
for m in models:m.inp.requires_grad_(False);m.gru.requires_grad_(False);m.cuda()
g=torch.Generator(device='cuda').manual_seed(193761)
x=torch.randn(2,169,211,device='cuda',generator=g)*.2;x[...,178]=0;x[:,::11,178]=1;x[:,0,178]=0
target=torch.randn(2,169,23,device='cuda',generator=g)*.3
cpu_before=torch.get_rng_state().clone();gpu_before=torch.cuda.get_rng_state().clone()
states=[None]*3;zero_checks=[]
for lo,hi in [(0,101),(101,169)]:
    outputs=[];grads=[]
    for i,m in enumerate(models):
        m.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):out,states[i]=m(x[:,lo:hi],states[i])
        outputs.append(out.detach());loss=(out.float()-target[:,lo:hi]).square().mean();(loss*1024).backward()
        grads.append({k:p.grad.detach().clone() for k,p in m.named_parameters() if k.startswith('head.')})
        states[i]=tuple(v.detach() for v in states[i]) if isinstance(states[i],tuple) else states[i].detach()
    for out in outputs[1:]:assert torch.equal(out,outputs[0]),'zero-extra AMP output differs'
    for other in grads[1:]:
        for k in grads[0]:assert torch.equal(other[k],grads[0][k]),('zero-extra head gradient',k)
    assert torch.equal(states[0],states[1][0]) and torch.equal(states[0],states[2][0])
    assert all(s[1].dtype==torch.float64 and s[2].dtype==torch.float64 and s[3].dtype==torch.bool and s[1].device.type=='cpu' for s in states[1:])
    zero_checks.append({'rows':[lo,hi],'outputs_head_gradients_and_GRU_hidden':'bitexact'})
assert torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(gpu_before,torch.cuda.get_rng_state())
updates=[]
for m in [mem,ctl]:
    frozen={k:v.clone() for k,v in m.state_dict().items() if k.startswith(('inp.','gru.')) or k in ['mean','std','ctx_projection']}
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0005,weight_decay=.01,fused=True)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024);state=None
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):out,state=m(x,state)
        loss=(out.float()-target).square().mean();scaler.scale(loss).backward();scaler.unscale_(opt)
        assert m.head_extra.weight.grad is not None and torch.isfinite(m.head_extra.weight.grad).all() and m.head_extra.weight.grad.abs().max()>0
        assert all(p.grad is None for k,p in m.named_parameters() if k.startswith(('inp.','gru.')))
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
        scaler.step(opt);scaler.update();state=tuple(v.detach() for v in state)
        for k,v in frozen.items():assert torch.equal(v,m.state_dict()[k]),('frozen changed',m.arm,k)
        if m.arm=='current':assert torch.equal(m.head_extra.weight[:,:16],m.head_extra.weight[:,16:])
    assert m.head_extra.weight.abs().max()>0 and scaler.get_scale()==1024
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        full,_=m(x);left,state=m(x[:,:101]);right,_=m(x[:,101:],state);chunk=torch.cat([left,right],1)
        delta=float((full.float()-chunk.float()).abs().max());assert delta<=5e-4,('chunk AMP error',m.arm,delta)
        prefix,_=m(x[:,:73]);prefix_delta=float((prefix.float()-full[:,:73].float()).abs().max());assert prefix_delta<=5e-4
        repeat,_=m(x);assert torch.equal(full,repeat)
    updates.append({'arm':m.arm,'fused_steps':3,'extra_max_abs':float(m.head_extra.weight.detach().abs().max()),'backbone_norm_P_frozen_bitexact':True,'finite_extra_grad_and_update':True,'final_scaler':scaler.get_scale(),'chunk_max_abs':delta,'prefix_max_abs':prefix_delta,'repeat_bitexact':True})
torch.cuda.synchronize();result={'status':'PASS','GPU':'7','device':torch.cuda.get_device_name(),'synthetic_only':True,'checkpoint_reads':0,'empirical_rows':0,'WP_queries':0,'constructor_CPU_CUDA_RNG_bitexact':True,'forward_backward_consumes_no_CPU_CUDA_RNG':True,'zero_extra_AMP_checks':zero_checks,'updates':updates,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'initial_free_bytes':free,'memory_fraction':.10,'CPU_threads':torch.get_num_threads(),'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':time.monotonic()-start,'torch':torch.__version__,'sources':{n:hashlib.sha256((a.actor/n).read_bytes()).hexdigest() for n in ['memory.py','model.py']},'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
assert result['max_RSS_KiB']<4*2**20 and result['peak_reserved_bytes']<.1*total
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));signal.alarm(0)
