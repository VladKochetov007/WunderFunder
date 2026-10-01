"""Assemble immutable three-partner manifests after root-generated legal training caches."""
import argparse,json
from pathlib import Path
from support import sha
HERE=Path(__file__).resolve().parent
SLOTS=['128s3','128s4','d256','c500']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True,choices=[0,1]);a=ap.parse_args()
 entries={slot:json.loads((HERE/'cache'/f'pred_on_ctl{slot}_f{a.fold}_training.json').read_text()) for slot in SLOTS}
 row_hashes={entry['source_row_group_order_sha256'] for entry in entries.values()};map_hashes={entry['chain_folds_sha256'] for entry in entries.values()}
 if len(row_hashes)!=1 or len(map_hashes)!=1:raise ValueError('partner source-map mismatch')
 for slot,entry in entries.items():
  if entry['fold']!=a.fold or entry['epoch']!=1 or entry['role']!='eligible_training_rows' or entry['run']!=f'on_ctl{slot}_f{a.fold}':raise ValueError('wrong control provenance')
  if sha(entry['prediction_path'])!=entry['prediction_sha256'] or sha(entry['checkpoint'])!=entry['checkpoint_sha256']:raise ValueError('partner file changed')
 for slot in ['128s3','128s4']:
  dest=HERE/'cache'/f'partners_{slot}_f{a.fold}.json'
  meta={'schema':'enshead01-partners-v1','fold':a.fold,'own_slot':slot,'source_row_group_order_sha256':next(iter(row_hashes)),'chain_folds_sha256':next(iter(map_hashes)),'partners':[v for k,v in entries.items() if k!=slot],'assembly_code_sha256':sha(__file__)}
  with dest.open('x') as f:json.dump(meta,f,indent=2);f.write('\n')
  print(str(dest))
if __name__=='__main__':main()
