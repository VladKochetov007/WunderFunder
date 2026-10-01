"""One authorized <=120s GPU6 synthetic transfer/step probe; no empirical inputs."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
import torch
from wnn.nn import RNNModel
from model import ContextMemoryModel

HERE=Path(__file__).resolve().parent


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
    output=HERE/'synthetic-gpu-result.json'
    if output.exists():raise FileExistsError(output)
    def deadline(signum,frame):raise TimeoutError('Fixed synthetic GPU120-second cap')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(120)
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.cuda.set_per_process_memory_fraction(.34)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    torch.manual_seed(20261003)
    parent=RNNModel(211,128,2,0,n_out=23)
    baseline=RNNModel(211,128,2,0,n_out=23);baseline.load_state_dict(parent.state_dict())
    current=ContextMemoryModel(arm='current');current.load_state_dict(parent.state_dict())
    memory=ContextMemoryModel(arm='memory');memory.load_state_dict(parent.state_dict())
    models=[('baseline',baseline),('current',current),('memory',memory)]
    for _,model in models:
        model.inp.requires_grad_(False);model.gru.requires_grad_(False);model.cuda()
    generator=torch.Generator(device='cuda').manual_seed(20261004)
    x=torch.randn(16,2000,211,device='cuda',generator=generator)*.2
    # A fixed dense event test (all later rows are events), no selected event rate.
    x[...,178]=1.;x[:,0,178]=0.
    results=[];start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for name,model in models:
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0005,weight_decay=.01,fused=True)
        scaler=torch.amp.GradScaler()
        state=None;times=[]
        for step in range(15):
            if step%10==0:state=None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();begin=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16):
                out,state=model(x,state)
                loss=out.float().square().mean()
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            elapsed=time.perf_counter()-begin
            if step>=5:times.append(elapsed)
            if isinstance(state,tuple):state=tuple(t.detach() for t in state)
            else:state=state.detach()
        results.append({'arm':name,'timed_steps':10,'batch':16,'chunk':2000,
                        'median_seconds':float(np.median(times)),'seconds':times})
    result={'status':'PASS','synthetic_only':True,'forecast_fits':0,'WP_queries':0,'GPU_visible':'6',
            'input':'fixed seeded random B16xT2000x211; event column178 true after first row',
            'warmup_steps_per_arm':5,'results':results,'wall_seconds':time.monotonic()-start,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'GPU_device_total_bytes':torch.cuda.get_device_properties(0).total_memory,
            'GPU_name':torch.cuda.get_device_name(0),'max_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'torch_version':torch.__version__,'CPU_threads':torch.get_num_threads(),
            'sources':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['memory.py','model.py','benchmark_synthetic_gpu.py']},
            'interpretation':'Implementation/transfer cost only; no forecast quality or data-profile inference'}
    assert result['peak_reserved_bytes']<=5.5*2**30
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','wall_seconds','peak_allocated_bytes','peak_reserved_bytes','max_rss_KiB']}))
    signal.alarm(0)


if __name__=='__main__':main()
