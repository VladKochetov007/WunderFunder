"""Generated CPU audit-helper branch tests; no GPU or competition inputs."""
from pathlib import Path
import copy,hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from gpu_step_audit import snapshot,step_audit,final_audit
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.manual_seed(791)
m=torch.nn.Linear(2,2);o=torch.optim.AdamW(m.parameters(),lr=.001);before=snapshot(m,o)
for p in m.parameters():o.state[p]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
r={'step':0,'loss':1.,'gradient_norm':float('inf'),'scale_before':65536.,'scale_after':32768.}
checks,n=step_audit(before,m,o,r,0);assert checks['PASS'] and n==0
cases=['first_skip_allocates_zero_moments']
state=copy.deepcopy(o.state_dict());params=copy.deepcopy(m.state_dict())
for case in ['parameter_changed','moment_changed','wrong_step_counter','timed_skip','nonfinite_loss']:
 m.load_state_dict(params);o.load_state_dict(copy.deepcopy(state));rr=dict(r)
 if case=='parameter_changed':
  with torch.no_grad():m.weight.add_(1.)
 if case=='moment_changed':o.state[m.weight]['exp_avg'].add_(1.)
 if case=='wrong_step_counter':o.state[m.weight]['step'].add_(1.)
 if case=='timed_skip':rr['step']=3
 if case=='nonfinite_loss':rr['loss']=float('nan')
 result,_=step_audit(before,m,o,rr,0);assert not result['PASS'],case;cases.append('reject_'+case)
m.load_state_dict(params);o.load_state_dict(copy.deepcopy(state))
prev=snapshot(m,o)
for p in m.parameters():p.grad=torch.ones_like(p)
o.step()
r={'step':3,'loss':1.,'gradient_norm':1.,'scale_before':32768.,'scale_after':32768.}
result,n=step_audit(prev,m,o,r,0);assert result['PASS'] and n==1;cases.append('timed_actual_update')
prev=snapshot(m,o);r.update(step=2,gradient_norm=float('inf'),scale_after=16384.)
result,n=step_audit(prev,m,o,r,1);assert result['PASS'] and n==1;cases.append('later_skip_unchanged_moments_counters')
assert not torch.cuda.is_initialized()
receipt={'status':'PASS','cases':cases,'CPU_only':True,'empirical_inputs_read':False,'audit_source_sha256':hashlib.sha256((P/'gpu_step_audit.py').read_bytes()).hexdigest(),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(P/'gpu-v2-step-audit-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
