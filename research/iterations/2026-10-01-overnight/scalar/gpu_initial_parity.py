"""Generated 32x2000 AMP initial-function/gradient/RNG check; no optimizer step."""
import torch
from scalar_model import ScalarRNNModel
from wnn.nn import RNNModel

def check_initial_pair(raw_cpu,knots,counts,build_inputs):
 with torch.random.fork_rng(devices=[0]):
  torch.manual_seed(3);base=RNNModel(211,128,2,0.,n_out=23)
  cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
  torch.manual_seed(3);candidate=ScalarRNNModel(211,knots=knots,counts=counts)
  assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
  for name,value in base.state_dict().items():assert torch.equal(value,candidate.state_dict()[name]),name
  inputs=build_inputs(raw_cpu[:,:2000].cuda())
  assert list(inputs.shape)==[32,2000,211]
  base=base.cuda()
  with torch.autocast('cuda',dtype=torch.float16):yb,hb=base(inputs)
  (yb.float().square().mean()*65536).backward()
  by=yb.detach().cpu();bh=hb.detach().cpu();gradients={name:p.grad.detach().cpu() for name,p in base.named_parameters()}
  assert all(torch.isfinite(g).all() for g in gradients.values())
  del base,yb,hb;torch.cuda.empty_cache()
  candidate=candidate.cuda()
  with torch.autocast('cuda',dtype=torch.float16):yc,hc=candidate(inputs)
  (yc.float().square().mean()*65536).backward()
  assert torch.equal(by,yc.detach().cpu()) and torch.equal(bh,hc.detach().cpu())
  max_error=0.
  for name,p in candidate.named_parameters():
   if name.startswith('scalar.'):continue
   observed=p.grad.detach().cpu();max_error=max(max_error,float((observed-gradients[name]).abs().max()))
   assert torch.equal(observed,gradients[name]),name
  assert torch.count_nonzero(candidate.scalar.table.grad)==0
  assert torch.isfinite(candidate.scalar.projection.grad).all() and candidate.scalar.projection.grad.abs().sum()>0
  result={'shape':[32,2000,211],'AMP_dtype':'float16','ordinary_tensor_initialization_exact':True,'CPU_CUDA_constructor_RNG_exact':True,'outputs_hidden_exact_numerical':True,'ordinary_parameter_gradients_exact_numerical':True,'base_gradient_max_abs':max_error,'zero_table_gradient':True,'nonzero_finite_projection_gradient':True,'signed_zero_byte_identity_claimed':False,'optimizer_update_parity_claimed':False,'normalization':'generated initial mean0/std1; full-step probe separately runs ordinary fit_normalization'}
 del candidate,yc,hc,inputs;torch.cuda.empty_cache()
 return result
