"""Preparation only: externally allocated fixed three-member scalar export, no inference."""
from pathlib import Path
import argparse,hashlib,json,sys
P=Path(__file__).resolve().parents[1];ROOT=P.parents[3];sys.path[:0]=[str(P),str(ROOT/'src')]
from training_recipe import sha
from scalar_model import load_scalar_model,SCHEMA
import numpy as np
import torch

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);a=ap.parse_args();alloc=json.loads(a.allocation.read_text());assert alloc.get('SCALARPLE_native_export_authorized') is True
 for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
 assert alloc['stage'] in ['heldfold_seed3','final_seed3'];fold=alloc['fold'];assert fold in [0,1] if alloc['stage']=='heldfold_seed3' else fold==-1
 assert len(alloc['children'])==3 and [c['coef'] for c in alloc['children']]==[1/3]*3
 fitentry=alloc['approved_scalar_fit'];assert sha(ROOT/fitentry['receipt_path'])==fitentry['receipt_sha256'];fit=json.loads((ROOT/fitentry['receipt_path']).read_text());assert sha(ROOT/fit['knots_path'])==fit['knots_file_sha256']
 out=ROOT/alloc['export_directory'];assert out.resolve().is_relative_to((P/'native-deployment').resolve());out.mkdir(parents=True,exist_ok=False);children=[]
 for slot,child in enumerate(alloc['children']):
  path=ROOT/child['checkpoint'];assert sha(path)==child['checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=False);cfg=ck['args'];assert ck['epoch']==1 and cfg['valid_fold']==fold and ck['n_in']==211 and ck['readout'] is None and not ck['readout_tilted']
  assert cfg['name']==child['run'] and cfg['hidden']==128 and cfg['layers']==2 and cfg['seed']==[3,4,3][slot]
  assert cfg['arch']=='gru' and cfg['cell']=='gru' and cfg['enc']=='linear' and cfg['dropout']==0
  assert not any(cfg.get(k) for k in ['lookahead','class_heads','heuristics','quant_feats','prop_input','atom_head','online_norm','transition_features','state_reset'])
  if slot==0:
   assert cfg['scalar_schema']==SCHEMA and cfg['scalar_arm']=='ple' and cfg['scalar_stage']=='head'
   assert cfg['scalar_fit_receipt_sha256']==fitentry['receipt_sha256'] and cfg['scalar_training_plan_sha256']==child['training_plan_sha256']
   proof=child['independent_integrity'];assert sha(ROOT/proof['path'])==proof['sha256'];review=json.loads((ROOT/proof['path']).read_text())
   assert review['status']==('PASS_INDEPENDENT_SCALAR8_FIT4_OOF_INTEGRITY' if alloc['stage']=='heldfold_seed3' else 'PASS_INDEPENDENT_SCALAR_FINAL_INTEGRITY')
   assert review['checkpoints'][child['run']]['endpoint_sha256']==child['checkpoint_sha256'] and review['plan_sha256']==child['training_plan_sha256']
   assert review['WP_queries']==0 and review['targets_decoded'] is False
   load_scalar_model(ck,expected_fit=fit)
  else:assert not any(k.startswith('scalar.') for k in ck['model']) and not cfg.get('scalar_schema')
  arrays={k:v.detach().cpu().numpy().copy() for k,v in ck['model'].items()};assert arrays['head.2.weight'].shape==(23,128)
  arrays['head.2.weight']=arrays['head.2.weight'][:2].copy();arrays['head.2.bias']=arrays['head.2.bias'][:2].copy()
  assert all(np.isfinite(v).all() for v in arrays.values()) and (arrays['std']>0).all()
  for key,value in arrays.items():assert value.dtype==(np.int64 if key in ['scalar.counts','scalar.raw_indices'] else np.float32),key
  dest=out/f'weights_{slot}.npz';np.savez(dest,**arrays);assert sha(path)==child['checkpoint_sha256']
  children.append({**child,'epoch':1,'fold':fold,'export':str(dest.relative_to(ROOT)),'export_sha256':sha(dest),'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in arrays.items()}})
 receipt={'status':'EXACT_FIXED_CHECKPOINT_EXPORT_NO_INFERENCE','stage':alloc['stage'],'fold':fold,'children':children,'approved_scalar_fit':fitentry,'allocation_sha256':sha(a.allocation),'script_sha256':sha(__file__),'sources':alloc['sources'],'targets_read':False,'WP_queries':0}
 (out/'exports.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'directory':str(out),'receipt_sha256':sha(out/'exports.json')}))
if __name__=='__main__':main()
