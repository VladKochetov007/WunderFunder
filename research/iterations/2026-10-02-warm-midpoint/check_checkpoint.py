"""Verify fixed CPU midpoint arithmetic and all eight source lineages."""
import argparse,sys
from midpoint import *
sys.path.insert(0,str(ROOT/'src'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert a.output.resolve().is_relative_to(HERE) and not a.output.exists() and not a.output.is_symlink()
 torch.set_num_threads(1);p=plan();records=[]
 from wnn.nn import RNNModel
 for pair in p['pairs']:
  old=load_source(pair['old']);warm=load_source(pair['warm']);path=ROOT/pair['endpoint'];before=sha(path)
  ck=torch.load(path,map_location='cpu',weights_only=False)
  assert sha(path)==before and ck['args']==pair['resolved_args'] and ck['epoch']==1 and ck['n_in']==211
  assert ck['readout'] is None and ck['readout_tilted'] is False and ck['construction']==provenance(pair)
  validate_state(ck['model']);assert all(same(old['model'][k],warm['model'][k]) and same(ck['model'][k],old['model'][k]) for k in BUFFERS)
  assert all(same(ck['model'][k],(old['model'][k].float()+warm['model'][k].float())*.5) for k in PARAMETERS)
  model=RNNModel(211,256,1,0.,n_out=23);model.load_state_dict(ck['model'],strict=True)
  receipt_path=path.parent/'construction.json';r=json.loads(receipt_path.read_text())
  assert r['status']=='CONSTRUCTED_FIXED_PARAMETER_MIDPOINT' and r['checkpoint']['sha256']==before
  assert r['commands_sha256']==sha(HERE/'commands.json') and r['constructor_sha256']==sha(HERE/'construct.py')
  assert all(r[k]==v for k,v in provenance(pair).items())
  records.append(dict(name=pair['name'],seed=pair['seed'],fold=pair['fold'],checkpoint=dict(path=pair['endpoint'],sha256=before,epoch=1),
   old_checkpoint=ck['construction']['old_checkpoint'],warm_checkpoint=ck['construction']['warm_checkpoint'],alpha=.5,
   mean_std_byte_identical=True,all10_parameters_exact_midpoint=True,ordinary_strict_load=True,
   construction_receipt_sha256=sha(receipt_path)))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 result=dict(status='PASS_FIXED_WARM_MIDPOINT_CHECKPOINTS',checkpoints=records,commands_sha256=sha(HERE/'commands.json'),
  checker_sha256=sha(Path(__file__)),forecast_fits=0,model_forward_calls=0,data_reads=0,metrics=0)
 write(a.output,result);print(json.dumps(result))
if __name__=='__main__':main()
