"""Only after root's separate runtime-pass allocation: export exact OOF children."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import numpy as np
import torch
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[0,1],required=True);ap.add_argument('--allocation',type=Path,required=True);a=ap.parse_args()
 alloc=json.loads(a.allocation.read_text());assert alloc.get('Q3_heldfold_predictions_authorized') is True and alloc.get('runtime_gate')=='PASS'
 assert alloc['zip_sha256']==sha(ROOT/'submissions/overnight_quant3/quant3_fixed.zip')
 d=P/'oof'/f'fold{a.fold}';d.mkdir(parents=True,exist_ok=False)
 groups=np.load(ROOT/'cache/valid_targets.npz')['row_group'];assert len(groups)==1873
 children=[]
 for i,name in enumerate(['on_ctld256','on_ctl128s3','on_ctlc500']):
  ckpath=ROOT/'artifacts/runs'/f'{name}_f{a.fold}'/'model_ep1.pt';digest=sha(ckpath);ck=torch.load(ckpath,map_location='cpu',weights_only=False);cfg=ck['args']
  assert int(ck['epoch'])==1 and int(cfg['valid_fold'])==a.fold and ck['n_in']==211
  assert not cfg.get('lookahead') and cfg.get('arch','gru')=='gru' and cfg.get('cell','gru')=='gru' and cfg.get('enc','linear')=='linear'
  assert not any(cfg.get(k) for k in ['class_heads','heuristics','quant_feats','prop_input','atom_head','online_norm','transition_features','state_reset'])
  arr={k:v.detach().cpu().float().numpy() for k,v in ck['model'].items()};arr['head.2.weight']=arr['head.2.weight'][:2].copy();arr['head.2.bias']=arr['head.2.bias'][:2].copy()
  assert arr['inp.weight'].shape==([256,128,128][i],211) and all(np.isfinite(v).all() for v in arr.values()) and (arr['std']>0).all()
  export=d/f'weights_{i}.npz';np.savez(export,**arr)
  children.append({'run':f'{name}_f{a.fold}','checkpoint':str(ckpath),'checkpoint_sha256':digest,'epoch':1,'fold':a.fold,'export':str(export),'export_sha256':sha(export),'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in arr.items()},'coef_float64':1/3,'coef_float32_hex':np.float32(1/3).tobytes().hex()})
  assert sha(ckpath)==digest
 for mode in ['quant','fp16','quant_avx2']:
  level='avx2' if mode=='quant_avx2' else 'v3'
  nd=P/'native/build'/level if mode.startswith('quant') else P.parent/'critic/native_order_v2/build/v3';bins=list(nd.glob('wnn_cpp.cpython-311-*.so'));assert len(bins)==1
  entry=f'{level}/{bins[0].name}' if mode.startswith('quant') else None
  if entry:
   with zipfile.ZipFile(ROOT/'submissions/overnight_quant3/quant3_fixed.zip') as z:assert hashlib.sha256(z.read(entry)).hexdigest()==sha(bins[0])
  manifest={'schema':'q3-composite-checkpoint-v1','kind':'ensemble_recipe_manifest_not_torch_checkpoint','mode':mode,'fold':a.fold,'epoch':1,'children':children,'sum':'original member order float32 fused multiply-add per member; coefficients float32(1/3)','native':str(bins[0]),'native_sha256':sha(bins[0]),'native_cpu_level':level,'zip_native_entry':entry,'quant_backend':'avx2' if mode=='quant_avx2' else 'vnni' if mode=='quant' else None,'same_composition':True,'zip':str(ROOT/'submissions/overnight_quant3/quant3_fixed.zip'),'zip_sha256':alloc['zip_sha256'],'allocation':str(a.allocation.resolve()),'allocation_sha256':sha(a.allocation),'source_row_group_order_sha256':hashlib.sha256(groups.tobytes()).hexdigest(),'chain_folds_sha256':sha(ROOT/'cache/chain_folds.npz'),'data_reader_source':str(ROOT/'src/wnn/data.py'),'data_reader_source_sha256':sha(ROOT/'src/wnn/data.py'),'predictor_source_sha256':sha(P/'predict_oof.py'),'exporter_source_sha256':sha(__file__),'targets_read':False,'avx2_audit_requires_additional_allocation':mode=='quant_avx2'}
  (d/f'{mode}.checkpoint.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'fold':a.fold,'directory':str(d),'children':[c['run'] for c in children],'targets_read':False,'WP_queries':0}))
if __name__=='__main__':main()
