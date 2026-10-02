"""Fixed final seed3 parameter midpoint. No fitting, data, inference or metrics."""
from pathlib import Path
import hashlib,importlib.util,json,os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
M=ROOT/'research/iterations/2026-10-02-warm-midpoint/midpoint.py'
M_SHA='a6ef8e8378724db48278fe263bd23c34d8a4f04768619cd5c33858d9978fef71'
ENDPOINT=ROOT/'artifacts/runs/oct02_wm_final_s3h2/model_ep1.pt'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def helpers():
 assert sha(M)==M_SHA
 sp=importlib.util.spec_from_file_location('frozen_midpoint_math',M);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def verify_training():
 c=read(HERE/'commands.json');r=read(HERE/'execution-01/result.json');a=read(HERE/'training-allocation.json')
 assert read(HERE/'execution-01/allocation.json')==a
 assert r['allocation_sha256']==sha(HERE/'training-allocation.json') and a['sources'][str((HERE/'commands.json').relative_to(ROOT))]==sha(HERE/'commands.json')
 assert r['status']==c['success_status'] and r['fits']==2 and r['OOF_jobs']==0 and r['metric_calls']==0
 assert [j['name'] for j in r['jobs']]==[j['name'] for j in c['jobs']]
 for j in r['jobs']:
  assert j['exit_code']==0 and not j.get('cleanup_error') and not j['remaining_scope_pids']
  p=ROOT/j['worker_result']['path'];assert sha(p)==j['worker_result']['sha256'];w=read(p);assert w['status']=='COMPLETE'
  for rel,b in w['outputs'].items():assert sha(ROOT/rel)==b['sha256']
 for rel,h in c['source_sha256'].items():assert sha(ROOT/rel)==h,rel
 job=c['jobs'][1];path=ROOT/job['endpoint'];cp=HERE/'execution-01'/f'{job["name"]}.check.json';check=read(cp)
 assert check['status']=='PASS_ORDINARY_WARM_FIXED_ENDPOINT_AND_LINEAGE' and check['checkpoint']['sha256']==sha(path)
 assert check['normalization_matches_parent'] and check['all103_args_match'] and check['model_pt_state_byte_equals_endpoint']
 assert check['commands_sha256']==sha(HERE/'commands.json') and check['validator_sha256']==sha(HERE/'check_checkpoint.py')
 return c,dict(result_sha256=sha(HERE/'execution-01/result.json'),allocation_sha256=sha(HERE/'training-allocation.json'),warm_check_sha256=sha(cp))

def expected():
 import torch
 m=helpers();c,proof=verify_training();b=read(HERE/'midpoint-binding.json');assert b['alpha']==.5 and b['final_seed']==3
 assert sha(ROOT/b['old']['path'])==b['old']['sha256']
 old=torch.load(ROOT/b['old']['path'],map_location='cpu',weights_only=False)
 warm_path=ROOT/c['jobs'][1]['endpoint'];wd=sha(warm_path);warm=torch.load(warm_path,map_location='cpu',weights_only=False)
 assert sha(warm_path)==wd and sha(ROOT/b['old']['path'])==b['old']['sha256']
 assert old['args']==b['old']['resolved_args'] and warm['args']==c['jobs'][1]['resolved_args']
 for ck in (old,warm):
  assert ck['epoch']==1 and ck['n_in']==211 and ck['readout'] is None and ck['readout_tilted'] is False
  m.validate_state(ck['model'])
 assert {k for k in old['args'] if old['args'][k]!=warm['args'][k]}=={'name','init'}
 assert old['args']['init']==c['jobs'][0]['resolved_args']['init']=='cont_sh256_final_s3@ep3'
 assert warm['args']['init']=='oct02_ow_final_s3@ep0' and old['args']['seed']==warm['args']['seed']==3
 assert old['args']['valid_fold']==warm['args']['valid_fold']==-1
 state=m.midpoint(old['model'],warm['model']);args=dict(old['args'],name='oct02_wm_final_s3h2')
 provenance=dict(kind='fixed_parameter_midpoint_not_a_fit',alpha=.5,seed=3,fold=-1,
  old_checkpoint=dict(path=b['old']['path'],sha256=b['old']['sha256'],epoch=1),
  warm_checkpoint=dict(path=str(warm_path.relative_to(ROOT)),sha256=wd,epoch=1),
  averaged_parameters=sorted(m.PARAMETERS),unchanged_buffers=sorted(m.BUFFERS),
  arithmetic='(old.float()+warm.float())*.5 on CPU float32',forecast_fits=0,targets_read=False,metrics=0,
  midpoint_math_sha256=M_SHA,midpoint_binding_sha256=sha(HERE/'midpoint-binding.json'),**proof)
 return state,args,provenance

def validate_constructed():
 import torch
 state,args,prov=expected();m=helpers();digest=sha(ENDPOINT);ck=torch.load(ENDPOINT,map_location='cpu',weights_only=False)
 assert sha(ENDPOINT)==digest and ck['epoch']==1 and ck['n_in']==211 and ck['args']==args and ck['construction']==prov
 assert ck['readout'] is None and ck['readout_tilted'] is False;m.validate_state(ck['model'])
 assert all(m.same(ck['model'][k],v) for k,v in state.items())
 receipt=read(ENDPOINT.parent/'construction.json')
 assert receipt['status']=='PASS_FIXED_FINAL_MIDPOINT_CONSTRUCTION' and receipt['checkpoint']==dict(path=str(ENDPOINT.relative_to(ROOT)),sha256=digest,epoch=1)
 assert receipt['construction']==prov and receipt['constructor_sha256']==sha(Path(__file__))
 return ck,receipt

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
 import sys,torch
 torch.set_num_threads(1);sys.path.insert(0,str(ROOT/'src'))
 assert not ENDPOINT.parent.exists() and not ENDPOINT.parent.is_symlink()
 state,args,prov=expected()
 from wnn.nn import RNNModel
 model=RNNModel(211,256,1,0.,n_out=23);model.load_state_dict(state,strict=True)
 m=helpers();assert set(dict(model.named_parameters()))==m.PARAMETERS and set(dict(model.named_buffers()))==m.BUFFERS
 ck=dict(model=state,args=args,n_in=211,epoch=1,readout=None,readout_tilted=False,construction=prov)
 ENDPOINT.parent.mkdir()
 with ENDPOINT.open('xb') as f:torch.save(ck,f)
 receipt=dict(status='PASS_FIXED_FINAL_MIDPOINT_CONSTRUCTION',checkpoint=dict(path=str(ENDPOINT.relative_to(ROOT)),sha256=sha(ENDPOINT),epoch=1),
  construction=prov,mean_std_byte_identical=True,all10_parameters_exact_midpoint=True,ordinary_strict_load=True,constructor_sha256=sha(Path(__file__)),commands_sha256=sha(HERE/'commands.json'))
 write(ENDPOINT.parent/'construction.json',receipt);validate_constructed();print(json.dumps(receipt))
if __name__=='__main__':main()
