"""Independent post-cache provenance/population audit, no metric or model fit."""
import hashlib,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def main():
 started=time.monotonic();plan=json.loads((HERE/'cache-execution-plan.json').read_text());groups=np.load(ROOT/'cache/valid_targets.npz')['row_group'];cf=np.load(ROOT/'cache/chain_folds.npz');v=cf['split']=='valid';foldmap={int(g):int(f) for g,f in zip(cf['row_group'][v],cf['fold'][v])};chainmap={int(g):int(c) for g,c in zip(cf['row_group'][v],cf['chain'][v])}
 ledger=[json.loads(line) for line in (HERE/'job-ledger.jsonl').read_text().splitlines()];completed=[z for z in ledger if z['event']=='completed'];assert len(completed)==10;assert not any(z['event'] in ('failed','resource_blocked','runner_exception') for z in ledger)
 checked=[]
 for job in plan['cache_jobs']:
  r=json.loads((ROOT/job['receipt']).read_text());f=r['fold'];expected=np.array([i for i,g in enumerate(groups) if foldmap[int(g)]!=f]);held=np.array([i for i,g in enumerate(groups) if foldmap[int(g)]==f])
  assert r['rows']==expected.tolist() and r['row_groups']==groups[expected].tolist()
  trainchains={chainmap[int(g)] for g in groups[expected]};heldchains={chainmap[int(g)] for g in groups[held]};assert trainchains.isdisjoint(heldchains)
  assert r['epoch']==1 and r['role']=='eligible_training_rows' and r['WP_queries']==0 and not r['target_columns_read'] and not r['mask_read']
  assert r['source_row_group_order_sha256']==hashlib.sha256(groups.tobytes()).hexdigest() and r['chain_folds_sha256']==digest(ROOT/'cache/chain_folds.npz')
  assert digest(r['prediction_path'])==r['prediction_sha256'] and digest(r['checkpoint'])==r['checkpoint_sha256']
  arr=np.load(r['prediction_path'],mmap_mode='r');assert arr.shape==(len(groups),20000,2) and arr.dtype==np.float32
  for start in range(0,len(expected),8):assert np.isfinite(arr[expected[start:start+8]]).all()
  for start in range(0,len(held),8):assert np.isnan(arr[held[start:start+8]]).all()
  checked.append({'run':r['run'],'fold':f,'training_sequences':len(expected),'held_sequences':len(held),'training_chains':len(trainchains),'held_chains':len(heldchains),'prediction_sha256':r['prediction_sha256'],'checkpoint_sha256':r['checkpoint_sha256'],'all_training_predictions_finite':True,'all_held_predictions_NaN':True,'chain_sets_disjoint':True})
 manifests=[]
 for f in (0,1):
  for seed in (3,4):
   path=HERE/'cache'/f'partners_128s{seed}_f{f}.json';m=json.loads(path.read_text());wanted={f'on_ctl{slot}_f{f}' for slot in ('128s3','128s4','d256','c500') if slot!=f'128s{seed}'}
   assert m['fold']==f and m['own_slot']==f'128s{seed}';assert {r['run'] for r in m['partners']}==wanted
   assert all(r['fold']==f and r['epoch']==1 for r in m['partners'])
   manifests.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path),'fold':f,'seed':seed,'partners':sorted(wanted)})
 receipt={'status':'PASS','source_execution_plan_sha256':digest(HERE/'cache-execution-plan.json'),'cache_jobs_completed':8,'manifest_jobs_completed':2,'GPU_fits':0,'WP_queries':0,'checks':checked,'manifests':manifests,'sampled_peak_GPU_MiB':max(z.get('sampled_peak_GPU_MiB',0) for z in completed),'job_seconds_total':sum(z['seconds'] for z in completed),'audit_seconds':time.monotonic()-started,'audit_code_sha256':digest(__file__)}
 dest=HERE/'cache-completion-audit.json'
 with dest.open('x') as out:json.dump(receipt,out,indent=2);out.write('\n')
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
