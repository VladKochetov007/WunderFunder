"""Allocated fixedH2 OOF producer; raw inputs only, no decoded target/mask arrays."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3];sys.path[:0]=[str(P),str(ROOT/'src')]
from training_recipe import sha,verify_job,verify_input_snapshot
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from wnn.data import SequenceFile
from wnn.resources import configure_torch
from scalar_model import load_scalar_model
class Inputs(Dataset):
 def __init__(self,groups):self.groups=list(groups);self.reader=None
 def __len__(self):return len(self.groups)
 def __getitem__(self,index):
  if self.reader is None:self.reader=SequenceFile(ROOT/'datasets/valid.parquet')
  return torch.from_numpy(self.reader.read(int(self.groups[index]),targets=False,mask=False).features)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);ap.add_argument('--run',required=True);a=ap.parse_args();alloc=json.loads(a.allocation.read_text());assert alloc.get('SCALARPLE_OOF_authorized') is True
 for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
 job=alloc['runs'][a.run];verify_job(job,a.run);assert job['stage']=='head' and job['fold'] in [0,1];expected=verify_input_snapshot(alloc)
 names=['datasets/valid.parquet','cache/valid_targets.npz','cache/chain_folds.npz'];before={n:sha(ROOT/n) for n in names};assert all(before[n]==expected['files'][n]['sha256'] for n in names)
 started=time.monotonic();configure_torch();path=ROOT/job['endpoint'];assert sha(path)==job['endpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=False)
 assert ck['epoch']==1 and ck['args']['valid_fold']==job['fold'] and ck['args']['seed']==job['seed'] and ck['args']['scalar_arm']==job['arm'] and ck['args']['scalar_stage']=='head'
 assert ck['args']['scalar_training_plan_sha256']==alloc['training_plan_sha256'] and ck['n_in']==211 and ck['readout'] is None and not ck['readout_tilted']
 spec=importlib.util.spec_from_file_location('scalar_predict_trainer',P.parent/'training/train_aux.py');trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
 fitted=alloc['approved_scalar_fit'];assert sha(ROOT/fitted['receipt_path'])==fitted['receipt_sha256'];fit=json.loads((ROOT/fitted['receipt_path']).read_text())
 if job['arm']=='ple':
  assert ck['args']['scalar_fit_receipt_sha256']==fitted['receipt_sha256'];model=load_scalar_model(ck,expected_fit=fit)
 else:
  model=trainer.make_model(ck['args'],ck['n_in'],23);model.load_state_dict(ck['model'],strict=True)
 model=model.cuda().eval()
 with np.load(ROOT/'cache/valid_targets.npz',allow_pickle=False) as z:groups=z['row_group']
 with np.load(ROOT/'cache/chain_folds.npz',allow_pickle=False) as z:chosen=z['row_group'][(z['split']=='valid')&(z['fold']==job['fold'])]
 rows=np.flatnonzero(np.isin(groups,chosen));selected=groups[rows];assert len(groups)==1873 and len(rows)==(467 if job['fold']==0 else 469)
 loader=DataLoader(Inputs(selected),batch_size=16,shuffle=False,num_workers=2,prefetch_factor=1)
 final=ROOT/job['prediction'];assert final.parent.resolve()==(P.parent/'cache').resolve();temporary=final.with_suffix('.npy.partial');receipt=final.with_suffix('.inputs.json');assert not any(x.exists() for x in [final,temporary,receipt]);final.parent.mkdir(exist_ok=True)
 out=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(1873,20000,2));out[:]=np.nan;written=0
 for raw in loader:
  pred=trainer.predict(model,raw,4000,'cuda');assert np.isfinite(pred).all();out[rows[written:written+len(pred)]]=pred;written+=len(pred)
 assert written==len(rows);out.flush();del out
 after={n:sha(ROOT/n) for n in names};assert after==before and sha(path)==job['endpoint_sha256'];temporary.replace(final)
 result={'run':a.run,'epoch':1,'fold':job['fold'],'seed':job['seed'],'arm':job['arm'],'checkpoint':str(path),'checkpoint_sha256':sha(path),'prediction_path':str(final),'prediction_sha256':sha(final),'rows':rows.tolist(),'row_groups':selected.tolist(),'source_row_group_order_sha256':hashlib.sha256(groups.tobytes()).hexdigest(),'chain_folds_sha256':before['cache/chain_folds.npz'],'outside_fold':'NaN','allocation_sha256':sha(a.allocation),'training_plan_sha256':alloc['training_plan_sha256'],'fit_receipt_sha256':fitted['receipt_sha256'],'input_hashes_before':before,'input_hashes_after':after,'targets_read':False,'masks_read':False,'metadata_arrays_only':['valid_targets.row_group','chain_folds.row_group','chain_folds.split','chain_folds.fold'],'metrics_computed':0,'WP_queries':0,'seconds':time.monotonic()-started,'script_sha256':sha(__file__),'sources':alloc['sources'],'input_identity':alloc['input_identity']}
 receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','row_groups']}))
if __name__=='__main__':main()
