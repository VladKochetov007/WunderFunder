"""Independent completed native/input artifact audit; no targets or model execution."""
from pathlib import Path
import argparse,hashlib,io,json,statistics,zipfile
import numpy as np

P=Path(__file__).resolve().parent.parent;ROOT=P.parents[2];S=P/'scalar'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(p):return json.loads(Path(p).read_text())
def same(a,b):assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
 allocation=S/'native-input-allocation.json';assert sha(allocation)=='9752adaa17412d5a2a347dd6ce962d9b244b1ccdc5a87ac116658904b474196d'
 alloc=get(allocation);proposal=get(S/'native-input-proposal.json');freeze=get(S/'NATIVE-INPUT-FREEZE.json')
 assert alloc['proposal_sha256']==sha(S/'native-input-proposal.json')=='2be2360c8d9c288dce308df77714cb590f8f151f183f3e791faa586df47594b1'
 assert alloc['freeze_sha256']==sha(S/'NATIVE-INPUT-FREEZE.json')=='70e615a5b2a59142b3ea5b62886e2349d9b06ad802081e2be501d4a9b9575e72'
 assert alloc['sources']==proposal['sources']==freeze['sources']
 for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
 result=get(S/'native-execution/result.json');assert result['status']=='PASS_INPUT_AND_NATIVE_PILOT_NO_QUALITY_CLAIM',result['status']
 assert result['failure'] is None and result['allocation_sha256']==sha(allocation) and result['source_sha256']==sha(S/'run_native_pilot.py')
 assert result['elapsed_seconds']<=900 and result['new_forecast_fits']==result['WP_queries']==result['target_reads']==0
 jobs=result['jobs'];assert len({j['label'] for j in jobs})==len(jobs)
 assert [j['label'] for j in jobs[:5]]==['fit-knots','prepare-package','correctness-v2','correctness-avx2','correctness-v3']
 for j in jobs:
  assert j['exit_code']==0 and j['PID']==j['PGID'] and j['start_ticks']>0
  assert j['argv'][:3]==['taskset','-c','22'] and j['max_sampled_RSS_KiB']<4*1024*1024
  assert j['log_sha256']==sha(S/'native-execution'/f"{j['label']}.log")
  assert '--allocation' in j['argv'] and Path(j['argv'][j['argv'].index('--allocation')+1])==allocation
 fit=get(S/'knots/fit.json');manifest=get(P/'renewal/selected-chains.json')
 assert fit['status']=='PASS' and fit['allocation_sha256']==sha(allocation) and fit['source_sha256']==sha(S/'fit_knots.py')
 assert fit['selected']==manifest['selected'] and len(fit['selected'])==256 and fit['selected_count']==256
 assert fit['source_before']==fit['source_after'] and fit['sample_shape']==[256000,44]
 assert fit['columns']==[f'i{i}_p{j}' for i in range(2) for j in range(22)] and fit['sample_rows_per_chain']==list(range(0,20000,20))
 assert not any(fit[k] for k in ['targets_read','masks_read','validation_chains_read','forecast_fits','WP_queries'])
 assert fit['affinity']==[22] and fit['max_rss_KiB']<4*1024*1024 and fit['wall_seconds']<480
 assert sha(ROOT/fit['knots_path'])==fit['knots_file_sha256']
 with np.load(ROOT/fit['knots_path'],allow_pickle=False) as z:k=z['knots'];c=z['counts']
 assert k.dtype==np.float32 and k.shape==(44,16) and c.dtype==np.int64 and c.shape==(44,) and np.isfinite(k).all()
 assert hashlib.sha256(k.tobytes()).hexdigest()==fit['knots_array_sha256'] and hashlib.sha256(c.tobytes()).hexdigest()==fit['counts_array_sha256']
 assert c.tolist()==fit['unique_counts']
 for row,n in zip(k,c):assert 1<=n<=16 and np.all(np.diff(row[:n].astype(np.float64))>0) and np.all(row[n:]==row[n-1]) and not np.signbit(row[row==0]).any()
 package=get(S/'native-pilot/package.json');base=ROOT/proposal['prototype']['baseline_zip'];candidate=S/'native-pilot/scalar_nonzero_prototype.zip'
 assert sha(base)==package['baseline_zip_sha256']==proposal['prototype']['baseline_sha256'] and sha(candidate)==package['zip_sha256']
 assert package['allocation_sha256']==sha(allocation) and package['fit_receipt_sha256']==sha(S/'knots/fit.json') and package['fit_knots_sha256']==fit['knots_file_sha256']
 with zipfile.ZipFile(base) as bz,zipfile.ZipFile(candidate) as cz:
  assert set(bz.namelist())==set(cz.namelist())==set(package['files'])
  members=json.loads(bz.read('members.json'));assert len(members)==3 and [m['coef'] for m in members]==[1/3]*3
  for name in cz.namelist():
   assert hashlib.sha256(cz.read(name)).hexdigest()==package['files'][name]
   if name.endswith('.so'):
    level=name.split('/')[0];assert hashlib.sha256(cz.read(name)).hexdigest()==alloc['sources'][str((S/'native_v2/build'/level/Path(name).name).relative_to(ROOT))]
   elif name!=members[0]['weights']:assert bz.read(name)==cz.read(name)
  with np.load(io.BytesIO(bz.read(members[0]['weights'])),allow_pickle=False) as old,np.load(io.BytesIO(cz.read(members[0]['weights'])),allow_pickle=False) as new:
   assert set(new.files)-set(old.files)=={'scalar.knots','scalar.counts','scalar.raw_indices','scalar.table','scalar.projection'}
   for key in old.files:same(old[key],new[key])
   same(k,new['scalar.knots']);same(c,new['scalar.counts']);same(np.array(list(range(22))+list(range(52,74)),np.int64),new['scalar.raw_indices'])
   rng=np.random.default_rng(2026100159);same(rng.normal(0,1,(44,16,16)).astype(np.float32),new['scalar.table']);same(rng.normal(0,.01,(128,16)).astype(np.float32),new['scalar.projection'])
 refs={level:get(S/f'native-pilot/reference-{level}.json') for level in ['v2','avx2','v3']}
 for level,r in refs.items():
  assert r['status']=='PASS' and r['rows']==(100000 if level=='v3' else 512)
  assert r['prototype_zip_sha256']==sha(candidate) and r['baseline_zip_sha256']==sha(base)
  assert r['native_sha256']==package['files'][f'{level}/wnn_cpp.cpython-311-x86_64-linux-gnu.so']
  assert all(r[key] for key in ['original_arrays_exact','plain_and_zero_extra_vs_archived_exact','nonzero_changes_predictions','reset_step_prefix_future_exact'])
  assert r['raw_sha256']==proposal['raw_fixture']['sha256'] and not r['targets_read'] and r['WP_queries']==0
 pairs=result['pairs'];quiet=[p for p in pairs if p['quiet']];assert [(p['index'],p['rows']) for p in quiet]==[(0,100000),(1,100000),(2,100000),(3,1000000)]
 evidence=[]
 for pair in pairs:
  assert pair['attempt'] in [0,1] and pair['order']==(['baseline','prototype'] if pair['index'] in [0,2] else ['prototype','baseline'])
  rr={arm:get(ROOT/path) for arm,path in pair['receipts'].items()}
  for arm,r in rr.items():
   assert r['arm']==arm and r['rows']==pair['rows'] and r['allocation_sha256']==sha(allocation) and r['source_sha256']==sha(S/'benchmark_native_pilot.py')
   assert r['zip_sha256']==sha(base if arm=='baseline' else candidate) and r['raw_sha256']==proposal['raw_fixture']['sha256']
   assert r['affinity']==[22] and r['threads']==1 and r['callback_vs_direct_reference_exact'] and r['full_callbacks_including_masked_rows'] and r['torch_runtime_absent']
   wall=sum(x['wall_seconds'] for x in r['per_sequence']);assert wall==r['wall_seconds'] and len(r['per_sequence'])*20000==r['rows']
   ticks=sum(x['activity']['cpu62']['total_ticks'] for x in r['per_sequence']);busy=sum(x['activity']['cpu62']['busy_ticks'] for x in r['per_sequence'])
   assert ticks==r['timed_activity']['cpu62']['total_ticks'] and busy==r['timed_activity']['cpu62']['busy_ticks']
   clean=r['precheck_activity']['cpu62']['busy_fraction']<.05 and busy/ticks<.05;assert clean==r['quiet_SMT_below_5pct']
   projected=wall/r['rows']*39400000/60;assert projected==r['projected_minutes_39_4M'] and (not clean or projected<=50)
   assert r['status']==('PASS' if clean else 'NOISY') and not r['targets_read'] and r['WP_queries']==0
  assert pair['quiet']==all(r['quiet_SMT_below_5pct'] for r in rr.values()) and pair['overhead_ratio']==rr['prototype']['wall_seconds']/rr['baseline']['wall_seconds']
  evidence.append({'index':pair['index'],'attempt':pair['attempt'],'quiet':pair['quiet'],'ratio':pair['overhead_ratio'],'minutes':{k:r['projected_minutes_39_4M'] for k,r in rr.items()}})
 median=statistics.median(p['overhead_ratio'] for p in quiet if p['rows']==100000);assert median==result['quiet_short_pair_median_ratio'] and median<=1.05
 for index in range(4):
  seen=[p for p in pairs if p['index']==index];assert len(seen) in [1,2] and seen[0]['attempt']==0
  if len(seen)==2:assert not seen[0]['quiet'] and seen[1]['attempt']==1 and seen[1]['quiet']
 assert len(jobs)==5+2*len(pairs)
 review={'status':'PASS_TARGET_FREE_INPUT_NATIVE_INTEGRITY','checker_sha256':sha(__file__),'allocation_sha256':sha(allocation),'result_sha256':sha(S/'native-execution/result.json'),'fit_receipt_sha256':sha(S/'knots/fit.json'),'package_receipt_sha256':sha(S/'native-pilot/package.json'),'prototype_sha256':sha(candidate),'median_short_overhead':median,'pairs':evidence,'source_hashes_checked':len(alloc['sources']),'jobs_checked':len(jobs),'original_member_arrays_exact':True,'deterministic_nonzero_adapter_exact':True,'knots_padding_dtype_finite_checked':True,'targets_read':False,'raw_empirical_inputs_read':False,'model_or_inference_executed':False,'WP_queries':0,'limitation':'Receipts and derived knot/prototype artifacts audited; no independent replay of actual raw callbacks. Historical footer identity is not full29GB source hash. Runtime measuredv3 only.'}
 a.output.write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review))
if __name__=='__main__':main()
