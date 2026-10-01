"""Whole-byte input identity check; never decodes labels, masks or raw arrays."""
from pathlib import Path
import argparse,datetime,json
from training_recipe import sha
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists();alloc=json.loads(a.allocation.read_text());assert alloc.get('SCALARPLE_forecast_fit_authorized') is True
 for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
 prior=alloc['historical_input_identity'];assert sha(ROOT/prior['path'])==prior['sha256'];historical=json.loads((ROOT/prior['path']).read_text());assert historical['status']=='PASS';files={}
 for rel,expected in historical['files'].items():
  path=ROOT/rel;before=path.stat();digest=sha(path);after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  assert digest==expected['sha256'] and before.st_size==expected['bytes'],rel
  files[rel]={'sha256':digest,'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'resolved_path':str(path.resolve())};print(json.dumps({'file':rel,**files[rel]}),flush=True)
 for fold in [0,1]:
  for split in ['train','valid']:
   metadata=json.loads((ROOT/f'cache/propcache_prop_f{fold}_{split}.json').read_text());assert metadata['model_sha256']==files[f'artifacts/runs/prop_f{fold}/model.pt']['sha256']
 result={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'historical_receipt_sha256':prior['sha256'],'allocation_sha256':sha(a.allocation),'script_sha256':sha(__file__),'decoded_competition_arrays':False,'metrics':0}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
