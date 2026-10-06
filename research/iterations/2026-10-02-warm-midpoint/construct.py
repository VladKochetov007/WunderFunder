"""Construct four preregistered fixed midpoints in fresh run directories."""
import argparse,os,sys
from pathlib import Path
from midpoint import *
sys.path.insert(0,str(ROOT/'src'))
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
 torch.set_num_threads(1);p=plan()
 # Check every destination before loading or writing any checkpoint.
 for pair in p['pairs']:
  path=ROOT/pair['endpoint'];assert not path.parent.exists() and not path.parent.is_symlink()
 from wnn.nn import RNNModel
 results=[]
 for pair in p['pairs']:
  old=load_source(pair['old']);warm=load_source(pair['warm']);state=midpoint(old['model'],warm['model'])
  model=RNNModel(211,256,1,0.,n_out=23);model.load_state_dict(state,strict=True)
  assert set(dict(model.named_parameters()))==PARAMETERS and set(dict(model.named_buffers()))==BUFFERS
  args=old['args'].copy();args['name']=pair['name'];assert args==pair['resolved_args']
  construction=provenance(pair)
  ck=dict(model=state,args=args,n_in=211,epoch=1,readout=None,readout_tilted=False,construction=construction)
  path=ROOT/pair['endpoint'];path.parent.mkdir()
  with path.open('xb') as f:torch.save(ck,f)
  receipt=dict(status='CONSTRUCTED_FIXED_PARAMETER_MIDPOINT',name=pair['name'],**construction,
   checkpoint=dict(path=pair['endpoint'],sha256=sha(path),epoch=1),mean_std_byte_identical=True,
   all10_parameters_exact_midpoint=True,ordinary_strict_load=True,commands_sha256=sha(HERE/'commands.json'),
   constructor_sha256=sha(Path(__file__)),midpoint_source_sha256=sha(HERE/'midpoint.py'))
  write(path.parent/'construction.json',receipt);results.append(receipt)
 print(json.dumps(dict(status='CONSTRUCTED_4_FIXED_MIDPOINTS_NO_FITS',checkpoints=results,forecast_fits=0,metrics=0)))
if __name__=='__main__':main()
