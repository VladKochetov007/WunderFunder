"""Allocated fixed endpoint integrity; no predictions or decoded competition arrays."""
from pathlib import Path
import argparse,json,sys
P=Path(__file__).resolve().parent;ROOT=P.parents[3];sys.path[:0]=[str(P),str(ROOT/'src')]
from training_recipe import sha,verify_job
import torch
from scalar_model import load_scalar_model,SCHEMA,TABLE_SEED

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);ap.add_argument('--run',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists();alloc=json.loads(a.allocation.read_text());assert alloc.get('SCALARPLE_forecast_fit_authorized') is True
 for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
 job=alloc['runs'][a.run];expected=verify_job(job,a.run);stage=job['stage'];epoch=3 if stage=='parent' else 1;path=ROOT/job['endpoint'];ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['epoch']==epoch and ck['n_in']==211 and ck['readout'] is None and not ck['readout_tilted']
 expected.update(scalar_arm=job['arm'],scalar_stage=stage,scalar_allocation_sha256=sha(a.allocation),scalar_training_plan_sha256=alloc['training_plan_sha256'],scalar_parent_sha256=job.get('parent',{}).get('sha256'))
 fitted=alloc['approved_scalar_fit'];assert sha(ROOT/fitted['receipt_path'])==fitted['receipt_sha256'];fit=json.loads((ROOT/fitted['receipt_path']).read_text())
 if job['arm']=='ple':
  expected.update(scalar_schema=SCHEMA,scalar_fit_receipt_sha256=fitted['receipt_sha256'],scalar_knots_array_sha256=fit['knots_array_sha256'],scalar_counts_array_sha256=fit['counts_array_sha256'])
  model=load_scalar_model(ck,expected_fit=fit)
 else:
  from wnn.nn import RNNModel
  model=RNNModel(211,128,2,0,n_out=23);model.load_state_dict(ck['model'],strict=True)
 assert ck['args']==expected,{k:(expected.get(k),ck['args'].get(k)) for k in set(expected)|set(ck['args']) if expected.get(k)!=ck['args'].get(k)}
 assert all(torch.isfinite(t).all() for t in ck['model'].values())
 for name,value in ck['model'].items():assert value.dtype==(torch.int64 if name in ['scalar.counts','scalar.raw_indices'] else torch.float32),name
 latest=torch.load(path.with_name('model.pt'),map_location='cpu',weights_only=False);assert latest['args']==ck['args'] and set(latest['model'])==set(ck['model'])
 same=lambda a,b:a.dtype==b.dtype and a.shape==b.shape and a.detach().cpu().numpy().tobytes()==b.detach().cpu().numpy().tobytes()
 assert all(same(t,latest['model'][k]) for k,t in ck['model'].items())
 history=json.loads(path.with_name('history.json').read_text());assert len(history)==epoch+1
 assert all(row['epoch']==i and row['evaluation_skipped'] and row['metric_placeholder_not_a_readout'] and row['weighted_pearson']==0 for i,row in enumerate(history))
 assert sorted(x.name for x in path.parent.glob('model_ep*.pt'))==[f'model_ep{i}.pt' for i in range(epoch+1)]
 parent_hash=None;frozen_exact=None;scalar_learning=None
 if stage=='parent' and job['arm']=='ple':
  gen=torch.Generator(device='cpu').manual_seed(TABLE_SEED);initial=torch.randn((44,16,16),generator=gen,dtype=torch.float32)
  table=ck['model']['scalar.table'];projection=ck['model']['scalar.projection'];assert torch.count_nonzero(projection)>0 and not same(table,initial),'inert scalar adapter'
  scalar_learning={'projection_norm':float(torch.linalg.vector_norm(projection)),'table_change_norm':float(torch.linalg.vector_norm(table-initial))}
 if stage=='head':
  parent=ROOT/job['parent']['path'];parent_hash=sha(parent);assert parent_hash==job['parent']['sha256'];prior=torch.load(parent,map_location='cpu',weights_only=False)
  assert prior['epoch']==3 and prior['args']['valid_fold']==job['fold'] and prior['args']['seed']==job['seed'] and prior['args']['scalar_arm']==job['arm']
  assert set(prior['model'])==set(ck['model']);assert all(same(v,ck['model'][k]) for k,v in prior['model'].items() if not k.startswith('head.'))
  assert any(not same(v,ck['model'][k]) for k,v in prior['model'].items() if k.startswith('head.'));frozen_exact=True
 result={'status':'PASS','run':a.run,'stage':stage,'epoch':epoch,'fold':job['fold'],'seed':job['seed'],'arm':job['arm'],'checkpoint_path':str(path.relative_to(ROOT)),'checkpoint_sha256':sha(path),'model_pt_exact_endpoint':True,'all_state_finite':True,'full_parsed_recipe_exact':True,'parent_sha256':parent_hash,'head_frozen_nonhead_state_byte_exact':frozen_exact,'scalar_fit_receipt_sha256':fitted['receipt_sha256'],'training_plan_sha256':alloc['training_plan_sha256'],'allocation_sha256':sha(a.allocation),'script_sha256':sha(__file__),'history_sha256':sha(path.with_name('history.json')),'scalar_learning':scalar_learning,'targets_read':False,'prediction_arrays_read':False,'WP_queries':0}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
