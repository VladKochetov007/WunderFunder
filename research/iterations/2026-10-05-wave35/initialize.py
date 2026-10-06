from pathlib import Path
import json,torch
from train_rnn import make_model,build_inputs
from wnn.data import SequenceFile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave35';torch.set_num_threads(2)
xs=[build_inputs(torch.tensor(SequenceFile(R/'datasets/valid.parquet').read(g).features[:2000])[None]) for g in (0,5)]
report=[]
for seed in (3,4):
 ck=torch.load(R/f'artifacts/runs/cmp_tstream_s{seed}_ft/model_ep1.pt',weights_only=False,map_location='cpu');src=ck['model'];n=src['head.2.weight'].shape[0];assert n==23
 old=make_model(ck['args'],ck['n_in'],n).eval();old.load_state_dict(src)
 args=torch.load(R/f'artifacts/runs/w19distilled_s{seed}/model_ep1.pt',weights_only=False,map_location='cpu')['args'].copy()
 args.update(name=f'w35init_s{seed}')
 torch.manual_seed(seed);target=make_model(args,211,n).eval();state=target.state_dict()
 state['mean'].copy_(src['mean']);state['std'].copy_(src['std']);state['inp.weight'].zero_();state['inp.bias'].zero_()
 for k in state:
  if k.startswith('gru.'):state[k].zero_()
 for i,off in ((0,0),(1,128)):
  idx=src[f'idx{i}'].long();state['inp.weight'][off:off+128,idx]=src[f'inp{i}.weight'];state['inp.bias'][off:off+128]=src[f'inp{i}.bias']
  for layer in (0,1):
   for typ in ('ih','hh'):
    for gate in range(3):
     dr=slice(gate*256+off,gate*256+off+128);sr=slice(gate*128,(gate+1)*128)
     state[f'gru.weight_{typ}_l{layer}'][dr,off:off+128]=src[f'gru{i}.weight_{typ}_l{layer}'][sr]
     state[f'gru.bias_{typ}_l{layer}'][dr]=src[f'gru{i}.bias_{typ}_l{layer}'][sr]
 state['head.0.weight'][:128]=src['head.0.weight'];state['head.0.bias'][:128]=src['head.0.bias']
 state['head.2.weight'].zero_();state['head.2.weight'][:,:128]=src['head.2.weight'];state['head.2.bias'].copy_(src['head.2.bias'])
 target.load_state_dict(state);errs=[]
 with torch.no_grad():
  for x in xs:
   p,(h0,h1)=old(x);q,h=target(x);err=float((p-q).abs().max());he=float((h-torch.cat((h0,h1),-1)).abs().max());assert max(err,he)<3e-5;errs.append({'all23_max_abs':err,'state_max_abs':he})
 target.zero_grad();target(xs[0][:,:100])[0][...,:2].square().mean().backward();assert target.head[2].weight.grad[:,128:].norm()>0
 # Verify that formerly cross-stream-zero recurrent blocks can start learning.
 assert target.gru.weight_hh_l0.grad[:128,128:].norm()>0
 out=R/f'artifacts/runs/w35init_s{seed}';assert not out.exists();out.mkdir();torch.save({'model':target.state_dict(),'args':args,'n_in':211,'epoch':0},out/'model_ep0.pt')
 report.append({'seed':seed,'parent':f'cmp_tstream_s{seed}_ft@ep1','parity':errs,'fresh_head_and_cross_stream_gradients_nonzero':True})
(D/'initialization-parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
