from pathlib import Path
import json,numpy as np,torch
from train_rnn import make_model,build_inputs
from wnn.data import SequenceFile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave32';torch.set_num_threads(2)
report=[]
x=build_inputs(torch.tensor(SequenceFile(R/'datasets/valid.parquet').read(0).features[:2000])[None])
for seed in (3,4):
 ck=torch.load(R/f'artifacts/runs/w19distilled_s{seed}/model_ep1.pt',weights_only=False,map_location='cpu');args=ck['args']
 torch.manual_seed(seed);old=make_model(args,211,23).eval();rng=torch.get_rng_state().clone();old.load_state_dict(ck['model'])
 torch.manual_seed(seed);new=make_model(dict(args,family_codebook=str(D/'codebook.npz')),211,23).eval()
 assert torch.equal(rng,torch.get_rng_state())
 missing,unexpected=new.load_state_dict(ck['model'],strict=False);assert not unexpected and all(k.startswith(('code_keys','code_emb.','code_proj.')) for k in missing)
 with torch.no_grad():
  a,_=old(x);b,_=new(x);assert torch.equal(a,b)
  new.code_proj.weight.normal_(0,.01)
  b,_=new(x);p,h=new(x[:,:1000]);q,_=new(x[:,1000:],h);err=float((b-torch.cat((p,q),1)).abs().max());assert err<5e-5
  raw=x.numpy();hi=np.ascontiguousarray(raw[...,0]).view(np.uint32).astype(np.uint64);lo=np.ascontiguousarray(raw[...,11]).view(np.uint32).astype(np.uint64)
  key=((hi<<np.uint64(32))|lo).view(np.int64);assert np.array_equal(key,new.pair_key(x,0,11).numpy())
 new.zero_grad();new(x[:,:100])[0][...,:2].square().mean().backward();assert new.code_proj.weight.grad.norm()>0
 report.append({'seed':seed,'initial_output_bit_exact':True,'constructor_rng_exact':True,'numpy_key_bit_exact':True,'prefix_chunk_max_abs':err,'projection_gradient':float(new.code_proj.weight.grad.norm())})
 # GPU key conversion and mixed-precision training smoke check, two seeds.
 new=new.cuda().train();new.zero_grad();xx=x[:,:100].cuda();assert np.array_equal(key[:,:100],new.pair_key(xx,0,11).cpu().numpy())
 with torch.autocast('cuda',dtype=torch.float16):p,_=new(xx);loss=p.square().mean()
 assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None);report[-1]['gpu_key_exact_amp_backward_finite']=True
(D/'model-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
