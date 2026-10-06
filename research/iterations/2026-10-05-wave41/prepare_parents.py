from pathlib import Path
import json,torch
from train_rnn import make_model,build_inputs
from wnn.data import SequenceFile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave41';torch.set_num_threads(2);x=build_inputs(torch.tensor(SequenceFile(R/'datasets/valid.parquet').read(0).features[:2000])[None]);report=[]
for kind,prefix in (('full','w19distilled'),('comp','w24comp_distilled')):
 for seed in (3,4):
  ck=torch.load(R/f'artifacts/runs/{prefix}_s{seed}/model_ep1.pt',weights_only=False,map_location='cpu');old=make_model(ck['args'],211,23).eval();old.load_state_dict(ck['model'])
  args=dict(ck['args'],family_teachers='',name=f'w41plain_{kind}_s{seed}');state={k:v for k,v in ck['model'].items() if not k.startswith('family_head.')};new=make_model(args,211,23).eval();new.load_state_dict(state)
  with torch.no_grad():a,h=old(x);b,j=new(x);assert torch.equal(a,b) and torch.equal(h,j)
  out=R/f'artifacts/runs/w41plain_{kind}_s{seed}';assert not out.exists();out.mkdir();torch.save({'model':state,'args':args,'n_in':211,'epoch':1},out/'model_ep1.pt')
  report.append({'kind':kind,'seed':seed,'main23_and_state_bit_exact':True,'unused_family_only_removed':True})
(D/'parent-parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
