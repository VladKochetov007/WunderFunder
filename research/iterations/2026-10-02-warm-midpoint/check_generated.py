"""Generated-only exact arithmetic, frozen buffers, strict-load and rejection checks."""
import sys
from midpoint import *
sys.path.insert(0,str(ROOT/'src'))
def main():
 torch.set_num_threads(1)
 from wnn.nn import RNNModel
 torch.manual_seed(31);model=RNNModel(211,256,1,0.,n_out=23)
 old={k:v.clone() for k,v in model.state_dict().items()};warm={k:v.clone() for k,v in old.items()}
 for k in PARAMETERS:warm[k]=warm[k]+torch.linspace(-.02,.03,warm[k].numel()).reshape(warm[k].shape)
 old_before={k:v.clone() for k,v in old.items()};warm_before={k:v.clone() for k,v in warm.items()}
 rng=torch.get_rng_state().clone();out=midpoint(old,warm);assert torch.equal(rng,torch.get_rng_state())
 assert len(PARAMETERS)==10 and all(same(out[k],torch.add(old[k],warm[k]).mul(.5)) for k in PARAMETERS)
 assert all(same(out[k],old[k]) and same(out[k],warm[k]) for k in BUFFERS)
 assert all(same(old[k],old_before[k]) and same(warm[k],warm_before[k]) for k in old)
 model.load_state_dict(out,strict=True);assert set(dict(model.named_parameters()))==PARAMETERS
 rejected=[]
 for reason in ('mean','std','negative_zero','dtype','keys','nonfinite'):
  bad={k:v.clone() for k,v in warm.items()}
  if reason in ('mean','std'):bad[reason][0]+=1
  elif reason=='negative_zero':bad['mean'][0]=-0.0;assert old['mean'][0]==0
  elif reason=='dtype':bad['head.2.bias']=bad['head.2.bias'].double()
  elif reason=='keys':bad['unexpected']=torch.zeros(1)
  else:bad['head.2.bias'][0]=float('nan')
  try:midpoint(old,bad)
  except AssertionError:rejected.append(reason)
  else:raise AssertionError('Invalid source accepted: '+reason)
 print(json.dumps(dict(status='PASS_GENERATED_FIXED_WARM_MIDPOINT',all10_parameter_arithmetic_exact=True,
  mean_std_byte_identical=True,source_tensors_unchanged=True,no_RNG_consumption=True,ordinary_strict_load=True,
  rejection_cases=rejected,midpoint_source_sha256=sha(HERE/'midpoint.py'),test_source_sha256=sha(Path(__file__)),forecast_fits=0,data_reads=0,metrics=0)))
if __name__=='__main__':main()
