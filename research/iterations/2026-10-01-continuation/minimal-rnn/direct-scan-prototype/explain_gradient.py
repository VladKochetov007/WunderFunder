"""Post-result scalar explanation only; no alternate model or tolerance changes."""
import hashlib,json,os
from pathlib import Path
import torch
from direct_scan import direct_scan
from torch_reference import blocked_scan,streaming_states,positive_g,log_g
P=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and list(os.sched_getaffinity(0))==[34]
torch.set_num_threads(1)
gates=torch.tensor([-80.,-20.,-17.,0.,20.,80.,1.])
values=torch.tensor([-80.,-8.,-.2,0.,8.,80.,.5])
out={'scope':'post-result deterministic explanation of fixed extreme candidate-VJP mismatch; no replacement recipe','gates':gates.tolist(),'candidate_values':values.tolist(),'activation':{},'cell':{}}
for name,fn in [('positive_g',positive_g),('exp_log_g',lambda x:log_g(x).exp())]:
    v=values.clone().requires_grad_();y=fn(v);g,=torch.autograd.grad(y.sum(),[v]);out['activation'][name]={'value':y.tolist(),'gradient':g.tolist()}
for length in [1,131]:
    for name,fn,dtype in [('direct',direct_scan,torch.float32),('oldlog',blocked_scan,torch.float32),('step',streaming_states,torch.float32),('double',streaming_states,torch.float64)]:
        k=gates.repeat(1,length,1).to(dtype).requires_grad_();v=values.repeat(1,length,1).to(dtype).requires_grad_();h=torch.zeros(1,7,dtype=dtype).requires_grad_()
        y,_=fn(k,v,h);gg,gv,gh=torch.autograd.grad(y.sum(),[k,v,h])
        out['cell'][f'{length}_{name}']={'first_candidate_gradient':gv[0,0].tolist(),'last_candidate_gradient':gv[0,-1].tolist(),'last_output':y[0,-1].tolist(),'initial_gradient':gh.tolist()}
out['source_sha256']={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['direct_scan.py','torch_reference.py','explain_gradient.py']}
(P/'generated-01/gradient-explanation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
