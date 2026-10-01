"""Independent source/protocol and generated state-checker checks; no GPU/model forward."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys
P=Path(__file__).resolve().parent.parent;ROOT=P.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((P/'scalar/gpu-v2-proposal.json').read_text())
for f,h in plan['sources'].items():assert sha(ROOT/f)==h,f
old=(P/'scalar/gpu_probe.py').read_text();new=(P/'scalar/gpu_probe_v2.py').read_text()
def region(text,left,right):return text.split(left,1)[1].split(right,1)[0]
assert region(old,'torch.manual_seed(3)\n','records=[];')==region(new,'torch.manual_seed(3)\n','records=[];')
assert region(old,' sl=slice(s,s+2000);',' records.append(')==region(new,' sl=slice(s,s+2000);',' records.append(')
tree=ast.parse(new)
assert new.index(' before=snapshot(')<new.index(' sl=slice(s,s+2000);')
assert new.index('audit,completed_updates=step_audit(')>new.index('end.record();torch.cuda.synchronize()')
import torch
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('auditor',P/'scalar/gpu_step_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
model=torch.nn.Linear(1,1,bias=False);opt=torch.optim.AdamW(model.parameters(),lr=.1)
parameter=next(model.parameters())
with torch.no_grad():parameter.fill_(1.)
opt.state[parameter]={'step':torch.tensor(4.),'exp_avg':torch.tensor([[2.]]),'exp_avg_sq':torch.tensor([[3.]])}
before=m.snapshot(model,opt);checks=[]
for step in range(10):
 record={'step':step,'loss':1.,'gradient_norm':float('inf'),'scale_before':32768.,'scale_after':16384.}
 result,n=m.step_audit(before,model,opt,record,4)
 assert result['PASS']==(step<3) and n==4
 checks.append(f'skip_at_step_{step}_accepted_{step<3}')
record={'step':3,'loss':1.,'gradient_norm':2.,'scale_before':16384.,'scale_after':16384.}
for increment in [0,1,2]:
 with torch.no_grad():parameter.fill_(1.25)
 opt.state[parameter]['step'].fill_(4+increment)
 result,n=m.step_audit(before,model,opt,record,4)
 assert result['PASS']==(increment==1)
 checks.append(f'finite_update_increment_{increment}_accepted_{increment==1}')
with torch.no_grad():parameter.fill_(1.)
opt.state[parameter]['step'].fill_(5.)
assert not m.step_audit(before,model,opt,record,4)[0]['PASS'];checks.append('counter_only_without_parameter_update_rejected')
with torch.no_grad():parameter.fill_(float('nan'))
assert not m.step_audit(before,model,opt,record,4)[0]['PASS'];checks.append('nonfinite_parameter_rejected')
with torch.no_grad():parameter.fill_(1.25)
opt.state[parameter]['exp_avg'].fill_(float('inf'))
assert not m.step_audit(before,model,opt,record,4)[0]['PASS'];checks.append('nonfinite_moment_rejected')
assert not torch.cuda.is_initialized()
result={'status':'PASS','checker_sha256':sha(__file__),'proposal_sha256':sha(P/'scalar/gpu-v2-proposal.json'),'freeze_sha256':sha(P/'scalar/GPU-V2-PREP-FREEZE.json'),'all_frozen_sources_verified':len(plan['sources']),'ordinary_setup_and_timed_update_regions_byte_exact':True,'readbacks_outside_timed_region':True,'generated_helper_checks':checks,'GPU_used':False,'model_forward_calls':0,'empirical_data_reads':0,'forecast_fits':0,'WP_queries':0}
(P/'critic/scalar-v2-preparation-check-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
