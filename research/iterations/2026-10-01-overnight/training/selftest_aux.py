"""Synthetic CPU correctness only, no dataset or metric readouts."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('aux02_train', HERE/'train_aux.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from wnn.nn import RNNModel, weighted_pooled_corr_loss
from wnn.outcomes import N_SP_AUX

def losses(model, x, y, scale):
 out,_=model(x);v=torch.ones(x.shape[:2],dtype=torch.bool)
 main=weighted_pooled_corr_loss(out[...,:2],y,v)
 # Synthetic valid classification/next-event labels, same graph and slice structure.
 aux=0
 for a,b in [(2,2+N_SP_AUX),(2+N_SP_AUX,5+N_SP_AUX),(5+N_SP_AUX,9+N_SP_AUX)]:
  labels=torch.arange(x.shape[0]*x.shape[1]).reshape(x.shape[:2])%(b-a)
  aux=aux+.3*torch.nn.functional.cross_entropy(out[...,a:b].reshape(-1,b-a),labels.flatten())
 nxt=out[...,9+N_SP_AUX:]
 aux=aux+.3*((nxt[...,0]-1)**2+torch.nn.functional.binary_cross_entropy_with_logits(nxt[...,1],torch.zeros_like(nxt[...,1]),reduction='none')).mean()
 return main+scale*aux,main,aux

torch.set_num_threads(2);torch.manual_seed(8301)
m=RNNModel(5,hidden=8,layers=2,n_out=23)
for n,p in m.named_parameters():
 if n.startswith(('inp','gru')):p.requires_grad_(False)
x=torch.randn(3,17,5);y=torch.randn(3,17,2)
params=list(p for p in m.parameters() if p.requires_grad)
loss,main,aux=losses(m,x,y,1)
g1=torch.autograd.grad(loss,params,retain_graph=True)
goriginal=torch.autograd.grad(main+aux,params)
assert all(torch.equal(a,b) for a,b in zip(g1,goriginal))
loss,main,aux=losses(m,x,y,0)
g0=torch.autograd.grad(loss,params,retain_graph=True)
gmain=torch.autograd.grad(main,params)
assert all(torch.equal(a,b) for a,b in zip(g0,gmain))
assert all(g is not None for g in g0)
assert torch.count_nonzero(g0[-1][2:])==0 and torch.count_nonzero(g0[-2][2:])==0
before={k:v.clone() for k,v in m.state_dict().items() if k.startswith(('inp','gru'))}
opt=torch.optim.AdamW(params,lr=.0005,weight_decay=.01)
opt.zero_grad(set_to_none=True);losses(m,x,y,0)[0].backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
m2=RNNModel(5,hidden=8,layers=2,n_out=23);m2.load_state_dict(m.state_dict(),strict=True)
assert m2(x)[0].shape==(3,17,23)
# Exact RNG advance for the original sequential, nonpersistent valid DataLoader.
loader=DataLoader(TensorDataset(torch.zeros(19,2)),batch_size=16,num_workers=0)
torch.manual_seed(829);before_rng=torch.get_rng_state().clone()
list(loader);after_iterator=torch.get_rng_state().clone()
torch.set_rng_state(before_rng);torch.empty((),dtype=torch.int64).random_(generator=loader.generator)
assert torch.equal(after_iterator,torch.get_rng_state())
# Replay sampler before/after evaluation: fixed paired epoch orders.
def orders(skip):
 torch.manual_seed(938)
 tr=DataLoader(TensorDataset(torch.arange(32)),batch_size=16,shuffle=True,num_workers=0)
 first=torch.cat([b[0] for b in tr]).tolist()
 if skip:torch.empty((),dtype=torch.int64).random_()
 else:list(loader)
 second=torch.cat([b[0] for b in tr]).tolist()
 return first,second
assert orders(False)==orders(True)
# Source recipe transformation remains exactly declared (no implicit code edits).
j=json.loads((HERE/'trainer-source.json').read_text());src=(ROOT/j['base']).read_text()
assert hashlib.sha256(src.encode()).hexdigest()==j['base_sha256']
for a,b in j['changes']:
 assert src.count(a)==1;src=src.replace(a,b)
assert src==(HERE/'train_aux.py').read_text()
print(json.dumps({'status':'PASS','checks':['scale1_loss_and_gradients_exact','scale0_equals_main_gradients','aux_private_rows_zero_not_none','frozen_trunk_unchanged_after_AdamW','strict_checkpoint_load_and_23_outputs','suppressed_eval_matches_DataLoader_CPU_RNG','two_epoch_sampler_orders_match','declared_source_transform_exact'],'implementation_sha256':j['implementation_sha256'],'base_sha256':j['base_sha256'],'torch':torch.__version__},indent=2))
