"""Fixed ordinary-GRU parameter midpoint; no fitting, data or random draws."""
from pathlib import Path
import hashlib,json
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SHAPES={'mean':(211,),'std':(211,),'inp.weight':(256,211),'inp.bias':(256,),
 'gru.weight_ih_l0':(768,256),'gru.weight_hh_l0':(768,256),'gru.bias_ih_l0':(768,),'gru.bias_hh_l0':(768,),
 'head.0.weight':(256,512),'head.0.bias':(256,),'head.2.weight':(23,256),'head.2.bias':(23,)}
BUFFERS={'mean','std'};PARAMETERS=set(SHAPES)-BUFFERS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def validate_state(state):
 assert set(state)==set(SHAPES)
 for k,shape in SHAPES.items():
  v=state[k];assert tuple(v.shape)==shape and v.dtype==torch.float32 and v.device.type=='cpu' and torch.isfinite(v).all(),k
 assert (state['std']>0).all()
def midpoint(old,warm):
 validate_state(old);validate_state(warm)
 assert all(same(old[k],warm[k]) for k in BUFFERS),'mean/std must be byte-identical'
 result={k:old[k].clone() if k in BUFFERS else (old[k].float()+warm[k].float())*.5 for k in old}
 validate_state(result)
 return result
def plan():
 p=json.loads((HERE/'commands.json').read_text())
 assert p['alpha']==.5 and p['counts']==dict(fits=0,OOF=4)
 for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==h,rel
 return p
def load_source(binding):
 path=ROOT/binding['path'];assert sha(path)==binding['sha256']
 ck=torch.load(path,map_location='cpu',weights_only=False)
 assert sha(path)==binding['sha256']
 assert ck['args']==binding['resolved_args'] and ck['epoch']==1 and ck['n_in']==211
 assert ck['readout'] is None and ck['readout_tilted'] is False
 assert len(ck['args'])==103 and not ck['args']['lookahead'] and ck['args']['hidden']==256 and ck['args']['layers']==1
 validate_state(ck['model']);return ck
def provenance(pair):
 return dict(kind='fixed_parameter_midpoint_not_a_fit',alpha=.5,seed=pair['seed'],fold=pair['fold'],
  old_checkpoint=dict(path=pair['old']['path'],sha256=pair['old']['sha256'],epoch=1),
  warm_checkpoint=dict(path=pair['warm']['path'],sha256=pair['warm']['sha256'],epoch=1),
  averaged_parameters=sorted(PARAMETERS),unchanged_buffers=sorted(BUFFERS),
  arithmetic='(old.float()+warm.float())*.5 on CPU float32',config_args_source='old H2 with name replaced; provenance, not a new training recipe',
  forecast_fits=0,targets_read=False,metrics=0)
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
