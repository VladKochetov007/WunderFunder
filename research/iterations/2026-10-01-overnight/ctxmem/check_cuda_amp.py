"""Single authorized CUDA AMP/fused-AdamW synthetic correctness pass; no inputs from disk."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import torch
from wnn.nn import RNNModel
from model import ContextMemoryModel

HERE=Path(__file__).resolve().parent


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
    output=HERE/'cuda-amp-parity.json'
    if output.exists():raise FileExistsError(output)
    def timeout(signum,frame):raise TimeoutError('Fixed120second CUDA parity cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    free=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=6200
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.cuda.set_per_process_memory_fraction(.34)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=True
    torch.manual_seed(20261005)
    baseline=RNNModel(211,128,2,0,n_out=23)
    baseline_cpu_rng=torch.get_rng_state().clone();baseline_gpu_rng=torch.cuda.get_rng_state_all()
    torch.manual_seed(20261005)
    memory=ContextMemoryModel(arm='memory')
    assert torch.equal(torch.get_rng_state(),baseline_cpu_rng)
    assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),baseline_gpu_rng))
    memory.load_state_dict(baseline.state_dict())
    control=ContextMemoryModel(arm='current');control.load_state_dict(baseline.state_dict())
    cases=[('baseline',baseline),('current',control),('memory',memory)]
    for _,model in cases:model.inp.requires_grad_(False);model.gru.requires_grad_(False);model.cuda()
    generator=torch.Generator(device='cuda').manual_seed(20261006)
    chunks=[torch.randn(3,61,211,device='cuda',generator=generator)*.2 for _ in range(3)]
    for x in chunks:x[...,178]=(torch.arange(61,device='cuda')%5==0).float()
    chunks[0][:,0,178]=0
    optimizers={name:torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0005,weight_decay=.01,fused=True) for name,model in cases}
    scalers={name:torch.amp.GradScaler() for name,_ in cases}
    states={name:None for name,_ in cases};snapshot={k:v.detach().cpu().clone() for k,v in baseline.state_dict().items() if not k.startswith('head.')}
    before_rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state_all())
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();first={};norms={};checks=[]
    for chunk_index,x in enumerate(chunks):
        for name,model in cases:
            optimizer,scaler=optimizers[name],scalers[name]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                out,state=model(x,states[name]);loss=out.float().square().mean()
            assert torch.isfinite(out).all() and torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            grads={k:p.grad.detach().clone() for k,p in model.named_parameters() if k.startswith('head.')}
            if chunk_index==0:first[name]={'out':out.detach().clone(),'grads':grads}
            total=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(total)
            if chunk_index==0:
                norms[name]=float(total)
                assert float(total)<1., 'Predeclared parity fixture must stay below common clipping threshold'
            scaler.step(optimizer);scaler.update()
            if chunk_index==0:first[name]['weights']={k:p.detach().clone() for k,p in model.named_parameters() if k.startswith('head.')}
            states[name]=tuple(t.detach() for t in state) if isinstance(state,tuple) else state.detach()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert all(p.grad is None for k,p in model.named_parameters() if k.startswith(('inp.','gru.')))
            assert all(torch.equal(v.cpu(),snapshot[k]) for k,v in model.state_dict().items() if k in snapshot)
        baseline_state=states['baseline']
        assert torch.equal(baseline_state,states['current'][0]) and torch.equal(baseline_state,states['memory'][0])
        checks.append({'chunk':chunk_index,'frozen_backbone_exact':True,'recurrent_state_exact':True,'all_finite':True})
    for arm in ('current','memory'):
        assert torch.equal(first['baseline']['out'],first[arm]['out'])
        for key in first['baseline']['grads']:
            assert torch.equal(first['baseline']['grads'][key],first[arm]['grads'][key]), (arm,key,'gradient')
            assert torch.equal(first['baseline']['weights'][key],first[arm]['weights'][key]), (arm,key,'AdamW update')
    assert torch.equal(torch.get_rng_state(),before_rng[0])
    assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),before_rng[1]))
    assert memory.head_extra.weight.abs().max()>0 and control.head_extra.weight.abs().max()>0
    result={'status':'PASS','synthetic_only':True,'forecast_fits':0,'WP_queries':0,'GPU6_free_MiB_before':free,
            'zero_extra_initial_outputs_bitexact':True,'initial_original_head_gradients_bitexact':True,
            'initial_original_head_fused_AdamW_update_bitexact':True,'constructor_CPU_and_GPU_RNG_exact':True,
            'forward_backward_RNG_unchanged':True,'first_step_total_gradient_norm':norms,
            'gradient_clipping_fixture':'All initial total norms<1; claim does not equate clipping when extra gradients change its scale',
            'three_chunks':checks,'extra_maps_received_nonzero_updates':True,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'max_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'seconds':time.monotonic()-start,
            'torch_version':torch.__version__,
            'sources':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['memory.py','model.py','check_cuda_amp.py']}}
    assert result['peak_reserved_bytes']<=5.5*2**30
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps({k:result[k] for k in ['status','seconds','peak_reserved_bytes']}))


if __name__=='__main__':main()
