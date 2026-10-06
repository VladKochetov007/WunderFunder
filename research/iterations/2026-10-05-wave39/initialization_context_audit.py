from pathlib import Path
import json,torch
from train_rnn import make_model
from wnn.propensity import load_propensity
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave39';torch.set_num_threads(2);result=[]
for seed in (3,4):
 init=torch.load(R/f'artifacts/runs/w39head_init_s{seed}/model_ep0.pt',weights_only=False,map_location='cpu')
 args=dict(init['args'],family_head_hidden=0)
 torch.manual_seed(seed);load_propensity(R/'artifacts/runs/prop_f0/model.pt','cpu');plain=make_model(args,211,23);rng=torch.get_rng_state().clone()
 torch.manual_seed(seed);load_propensity(R/'artifacts/runs/prop_f0/model.pt','cpu');wide=make_model(init['args'],211,23);assert torch.equal(rng,torch.get_rng_state())
 err=float((init['model']['family_head.weight'][:,:256]-plain.family_head.weight).abs().max());bias=float((init['model']['family_head.bias']-plain.family_head.bias).abs().max())
 assert err==0 and bias==0,(err,bias)
 result.append({'seed':seed,'training_constructor_rng_still_matched_in_propensity_context':True,'initial_family_weight_max_abs_vs_actual_control':err,'initial_family_bias_max_abs_vs_actual_control':bias})
r={'findings':result,'conclusion':'Auxiliaryfamily head is bit-exact withactualwave19 control initialization afterPropensityNet constructor; trainingconstructor RNG matched. Correctedv2.'}
(D/'initialization-context-audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
