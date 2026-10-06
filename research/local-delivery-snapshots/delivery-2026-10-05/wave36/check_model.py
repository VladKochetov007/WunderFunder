from pathlib import Path
import torch,json
from train_rnn import make_model,build_inputs
from wnn.data import SequenceFile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave36';torch.set_num_threads(2)
x=build_inputs(torch.tensor(SequenceFile(R/'datasets/valid.parquet').read(0).features[:500])[None]);report=[]
for seed in (3,4):
 ck=torch.load(R/f'artifacts/runs/w19distilled_s{seed}/model_ep1.pt',weights_only=False,map_location='cpu');states=[];pred=[]
 for mode in ('off','uniform','soft'):
  torch.manual_seed(seed);m=make_model(dict(ck['args'],family_class_mode=mode),211,23).eval();states.append(torch.get_rng_state().clone());missing,unexpected=m.load_state_dict(ck['model'],strict=False);assert sorted(missing)==['class_res.bias','class_res.weight'] and not unexpected
  with torch.no_grad():pred.append(m(x)[0])
  if mode=='off':
   m(x)[0][...,:2].square().mean().backward();assert m.class_res.weight.grad is None
 assert all(torch.equal(states[0],s) for s in states);assert all(torch.equal(pred[0],p) for p in pred)
 report.append({'seed':seed,'all_three_constructor_rng_identical':True,'all_three_initial_output_bit_exact':True,'off_head_unused_no_gradient':True})
(D/'model-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
