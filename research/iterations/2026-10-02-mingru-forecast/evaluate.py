"""One fixed minGRU4 versus current-student4 readout, reusing frozen B mathematics."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
B=ROOT/'research/iterations/2026-10-01-ensemble-backbone';OLD=ROOT/'research/iterations/2026-10-01-overnight';EDA=ROOT/'research/iterations/2026-10-01-error-analysis'
BP=B/'evaluate.py';BSHA='c5f3ecc795ebd680872c39a652075a894192bfe9b4e654eca7e826c0310980e9'
LB=ROOT/'research/iterations/2026-10-01-late-atom-head/input-bindings.json';LSHA='573cd5cbaaddbda844abd4fd6736fe01de01726b2ceb118ed1b5b4d1682b8ff8'
ARCH_SHA='16585c0fdcd6cf16b63234c64ea7a02d620a8f705f933fac4324c0a880a234a6'
assert hashlib.sha256(BP.read_bytes()).hexdigest()==BSHA
spec=importlib.util.spec_from_file_location('mingru_frozen_metric',BP);E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
ANCHORS=('on_ctl128s3','on_ctl128s4','on_ctlc500');MODELS=('mingru4','current_student4')
DONE='COMPLETE_4_FITS_2_OOF_NO_METRICS'

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--evaluation-source-sha256',required=True);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
 out=args.output_dir.resolve();assert out.is_relative_to(HERE) and out!=HERE and not out.exists();started=time.monotonic();tracked={};states={}
 def pin(path,digest):
  p=Path(path)
  if str(p) in tracked:assert tracked[str(p)]==digest;return
  assert E.sha(p)==digest,('hash',str(p));tracked[str(p)]=digest;s=p.stat();states[str(p)]=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 def produced(path,outputs):
  assert str(path.relative_to(ROOT)) in outputs,('not a worker output',str(path));return path
 freeze=HERE/'evaluation-source.json';pin(freeze,args.evaluation_source_sha256);frozen=json.loads(freeze.read_text())
 assert frozen['status']=='FROZEN_BEFORE_METRIC_READOUT' and frozen['legacy_bindings_sha256']==LSHA
 assert frozen['source_sha256'][str(Path(__file__).relative_to(ROOT))]==E.sha(__file__) and frozen['source_sha256'][str(BP.relative_to(ROOT))]==BSHA
 for rel,h in frozen['source_sha256'].items():pin(E.path_for(rel),h)
 apath=HERE/'training-allocation.json';pin(apath,frozen['training_allocation_sha256']);ah=tracked[str(apath)];a=json.loads(apath.read_text())
 assert a['execution_authorized'] and a['success_status']==DONE
 for rel,h in a['sources'].items():pin(E.path_for(rel),h)
 # Root pipeline already enforces large training-input stat identities. Bind those
 # manifests here without reading unused teacher tensors or the original parquet.
 for r in a['input_bindings']:pin(E.path_for(r['path']),r['sha256'])
 cp=HERE/'commands.json';assert tracked[str(cp)]==a['sources'][str(cp.relative_to(ROOT))];commands=json.loads(cp.read_text())
 expected=[(j['name'],j['stage']) for j in commands['jobs']]
 assert expected==[(name,stage) for f in (0,1) for name,stage in ((f'oct02_mingru_s3_f{f}','parent'),(f'oct02_mingru_s3h2_f{f}','H2'),(f'oct02_mingru_s3h2_f{f}_OOF','OOF'))]
 assert [(j['name'],j['stage']) for j in a['jobs']]==expected
 for j,c in zip(a['jobs'],commands['jobs']):assert j['argv']==c['argv']
 dp=HERE/'execution-01/result.json';pin(dp,E.sha(dp));d=json.loads(dp.read_text())
 assert a['execution_directory']==str(dp.parent.relative_to(ROOT)) and d['status']==DONE and d['allocation_sha256']==ah
 assert d['failure'] is None and (d['fits'],d['OOF_jobs'],d['metric_calls'],d['retries'])==(4,2,0,0)
 assert d['runner_sha256']==a['sources'][str((HERE/'run_queue.py').relative_to(ROOT))]
 assert [(j['name'],j['stage']) for j in d['jobs']]==expected
 outputs={}
 for j,plan in zip(d['jobs'],a['jobs']):
  assert j['exit_code']==0 and j['failure'] is None and j['remaining_scope_pids']==[] and not j.get('cleanup_error')
  wr=j['worker_result'];p=E.path_for(wr['path']);pin(p,wr['sha256']);w=json.loads(p.read_text())
  assert w['status']=='COMPLETE' and w['allocation_sha256']==ah and w['name']==j['name'] and w['stage']==j['stage'] and w['returncode']==0
  assert w['argv']==plan['argv'] and set(w['outputs'])==set(plan['required_outputs'])
  if j['stage']!='OOF':assert w['post_returncode']==0
  for rel,r in w['outputs'].items():
   p=E.path_for(rel);assert rel not in outputs and p.stat().st_size==r['bytes'];pin(p,r['sha256']);outputs[rel]=r
 integrity={}
 for job in commands['jobs']:
  if job['stage']=='OOF':continue
  p=produced(HERE/'execution-01'/f'{job["name"]}.check.json',outputs);q=json.loads(p.read_text());checkpoint=produced(ROOT/job['endpoint'],outputs)
  assert q['status']=='PASS_FIXED_ENDPOINT_FINITE_LINEAGE' and q['run']==job['name'] and q['stage']==job['stage'] and q['fold']==job['fold'] and q['epoch']==(3 if job['stage']=='parent' else 1) and q['n_in']==211
  assert q['checkpoint']==dict(path=str(checkpoint),sha256=tracked[str(checkpoint)]) and q['architecture_source_sha256']==ARCH_SHA
  assert q['model_pt_sha256']==tracked[str(checkpoint.with_name('model.pt'))]
  assert q['model_pt_state_byte_equal'] and q['all_resolved_args_equal'] and q['strict_state_shapes'] and q['all_state_finite']
  assert q['commands_sha256']==tracked[str(cp)] and q['validator_sha256']==a['sources'][str((HERE/'check_checkpoint.py').relative_to(ROOT))]
  if job['stage']=='H2':
   parent=ROOT/f'artifacts/runs/oct02_mingru_s3_f{job["fold"]}/model_ep3.pt'
   assert q['parent']==dict(path=str(parent),sha256=tracked[str(parent)])
   names=q['byte_identical_frozen_tensors'];assert 'mean' in names and 'std' in names and any(n.startswith('inp.') for n in names) and any(n.startswith('cells.') for n in names)
  else:assert q['parent'] is None
  integrity[job['name']]=q
 pin(LB,LSHA);records=json.loads(LB.read_text())['records'];legacy={r['path']:r for r in records}
 def oldpin(path):
  # The preserved manifest's remote lexical names are mapped to this checkout;
  # byte identity, not historical inode equality, permits local preservation copies.
  key='/home/grinchenkoeg/WunderFunder-creative/'+str(path.relative_to(ROOT));r=legacy[key]
  assert r.get('sha256') and path.stat().st_size==r['size'];pin(path,r['sha256']);return path
 rgp=oldpin(ROOT/'cache/valid_targets.npz');cfp=oldpin(ROOT/'cache/chain_folds.npz')
 with np.load(rgp,allow_pickle=False) as z:groups=z['row_group']
 with np.load(cfp,allow_pickle=False) as z:
  v=z['split']=='valid';info={int(g):(int(f),int(c),bool(p)) for g,f,c,p in zip(z['row_group'][v],z['fold'][v],z['chain'][v],z['pure'][v])};trainchains=set(z['chain'][z['split']=='train'].tolist())
 assert len(groups)==1873 and len(set(groups.tolist()))==1873 and set(info)==set(groups.tolist())
 assert all(c not in trainchains for f,c,p in info.values() if p)
 ep=oldpin(EDA/'extraction-01/scored-rows.npz');er=oldpin(EDA/'extraction-01/receipt.json');sp=oldpin(B/'evaluation-01/sufficient.npz')
 assert json.loads(er.read_text())['outputs'][ep.name]['sha256']==tracked[str(ep)]
 with np.load(sp,allow_pickle=False) as z:
  old=z['chain_stats'];counts=z['chain_counts'];keys=z['chain_metadata'];assert z['recipe_names'].tolist()==list(E.RECIPES) and z['scope_names'].tolist()==list(E.SCOPES)
 assert old.shape==(250,2,3,2,6) and keys.shape==(250,3) and counts.shape==(250,2) and counts[:,0].sum()==1614484
 with np.load(ep,allow_pickle=False) as z:
  y=z['y'];meta=z['metadata'];assert z['metadata_names'].tolist()==['cache_position','row_group','step','chain','fold','market_class_retrospective']
  base=z['predictions'][:,z['prediction_names'].tolist().index('shallow_seed3'),:]
  age=np.rint(np.expm1(z['X'][:,z['feature_names'].tolist().index('log_since_i0px')].astype(np.float64))).astype(np.int32)
 assert meta.shape==(1614484,6) and y.shape==base.shape==(1614484,2) and np.isfinite(y).all() and np.isfinite(base).all()
 stats=np.zeros((250,2,2,2,6));checkcounts=np.zeros_like(counts);ki={tuple(k):i for i,k in enumerate(keys)};sequences=0
 out.mkdir();E.write(out/'started.json',dict(source_sha256=E.sha(__file__),evaluation_source_sha256=args.evaluation_source_sha256,allocation_sha256=ah,input_sha256=tracked))
 for f in (0,1):
  run=f'oct02_mingru_s3h2_f{f}';p=produced(HERE/'cache'/f'pred_{run}_valid.npy',outputs);rp=produced(p.with_suffix('.inputs.json'),outputs);r=json.loads(rp.read_text())
  checkpoint=ROOT/f'artifacts/runs/{run}/model_ep1.pt'
  assert r['run']==run and r['fold']==f and r['epoch']==1 and r['prediction_path']==str(p) and r['prediction_sha256']==tracked[str(p)]
  assert r['checkpoint']==str(checkpoint) and r['checkpoint_sha256']==tracked[str(checkpoint)] and r['architecture_source_sha256']==ARCH_SHA and r['targets_read'] is False and r['metrics_computed']==0
  assert r['script_sha256']==a['sources'][str((HERE/'predict_minimal.py').relative_to(ROOT))]
  allrows=[i for i,g in enumerate(groups) if info[int(g)][0]==f]
  assert r['rows']==allrows and r['row_groups']==groups[allrows].tolist() and r['source_row_group_order_sha256']==hashlib.sha256(groups.tobytes()).hexdigest() and r['chain_folds_sha256']==tracked[str(cfp)]
  prediction=np.load(p,mmap_mode='r',allow_pickle=False);assert prediction.shape==(1873,20000,2) and prediction.dtype==np.float32
  # No all-array temporary: check every full sequence's held-fold/sentinel contract.
  chosen=set(allrows)
  for pos in range(1873):assert (np.isfinite(prediction[pos]).all() if pos in chosen else np.isnan(prediction[pos]).all())
  anchors=[]
  for name in ANCHORS:
   p=oldpin(OLD/'cache'/f'pred_{name}_f{f}_valid.npy');rp=oldpin(p.with_suffix('.inputs.json'));ar=json.loads(rp.read_text())
   assert ar['prediction_sha256']==tracked[str(p)] and ar['fold']==f and ar['epoch']==1 and ar['run']==f'{name}_f{f}'
   assert ar['rows']==allrows and ar['row_groups']==groups[allrows].tolist() and ar['source_row_group_order_sha256']==r['source_row_group_order_sha256'] and ar['chain_folds_sha256']==tracked[str(cfp)]
   arr=np.load(p,mmap_mode='r',allow_pickle=False);assert arr.shape==(1873,20000,2) and arr.dtype==np.float32;anchors.append(arr)
  take=meta[:,4]==f;mm=meta[take];yy=y[take];bb=base[take];aa=age[take]
  cuts=np.r_[0,np.flatnonzero(mm[1:,0]!=mm[:-1,0])+1,len(mm)];assert len(cuts)-1==(425 if f==0 else 435)
  seen=[]
  for lo,hi in zip(cuts[:-1],cuts[1:]):
   rows=mm[lo:hi];pos=int(rows[0,0]);cl=int(rows[0,5]);chain=int(rows[0,3]);steps=rows[:,2];seen.append(pos)
   assert int(groups[pos])==int(rows[0,1]) and info[int(groups[pos])]==(f,chain,True)
   assert np.all(rows[:,0:2]==rows[0,0:2]) and np.all(rows[:,3:]==rows[0,3:]) and np.all(steps>=99) and np.all(steps<20000) and np.all(np.diff(steps)>0)
   parts=[arr[pos,steps] for arr in anchors];assert all(np.isfinite(v).all() for v in parts)
   blends=(E.blend([prediction[pos,steps],*parts]),E.blend([bb[lo:hi],*parts]));j=ki[f,chain,cl]
   for s,keep in enumerate((np.ones(hi-lo,bool),(cl==3)&(aa[lo:hi]>=64))):
    checkcounts[j,s]+=int(keep.sum())
    for m,pred in enumerate(blends):stats[j,s,m]+=E.sufficient(yy[lo:hi][keep],pred[keep])
  assert seen==[i for i,g in enumerate(groups) if info[int(g)][0]==f and info[int(g)][2]];sequences+=len(seen)
 assert sequences==860;np.testing.assert_array_equal(checkcounts,counts)
 for m in (0,1):np.testing.assert_allclose(np.take(stats[:,:,m],[0,1,3],axis=-1),np.take(old[:,:,2],[0,1,3],axis=-1),atol=2e-8,rtol=0)
 np.testing.assert_allclose(stats[:,:,1],old[:,:,2],atol=2e-8,rtol=1e-12)
 assert abs(E.metric(stats[:,0,1].sum(0))['WP']-E.BASELINE_WP)<2e-12
 draws,strata=E.resamples(keys);scopes={}
 for s,name in enumerate(E.SCOPES):
  scopes[name]=E.comparison(stats[:,s],counts[:,s],draws)
  scopes[name]['folds']={str(f):E.comparison(stats[keys[:,0]==f,s],counts[keys[:,0]==f,s],draws[:,keys[:,0]==f]) for f in (0,1)}
 report=dict(status='COMPLETE_FIXED_MINGRU_SCREEN',models=MODELS,primary='mingru4 minus current_student4',scopes=scopes,sequences=860,chains=250,scored_rows=1614484,fits=4,OOF_jobs=2,baseline_reproduced=E.BASELINE_WP,fit_integrity=integrity,uncertainty=dict(draws=E.BOOTSTRAPS,seed=E.SEED,strata=strata,interpretation='Shared whole-chain fold/class-stratified paired draws; fixed-model conditional marginal95% intervals omit seed, selection and hidden-test uncertainty.'),interpretation='One fixed recurrent architecture recipe. The comparison function control field means existing incumbent, not a fresh matched control. Repeatedly exposed folds; fixed endpoints and quarter blend, no selection sweep or automatic spending gate.',source_sha256=E.sha(__file__),frozen_metric_sha256=BSHA,evaluation_source_sha256=args.evaluation_source_sha256,allocation_sha256=ah,input_sha256=tracked,seconds=time.monotonic()-started)
 for p,state in states.items():
  st=Path(p).stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==state,('input changed',p)
 np.savez_compressed(out/'sufficient.npz',chain_stats=stats,chain_counts=counts,chain_metadata=keys,scope_names=np.array(E.SCOPES),model_names=np.array(MODELS))
 E.write(out/'report.json',report);E.write(out/'receipt.json',dict(status=report['status'],source_sha256=E.sha(__file__),evaluation_source_sha256=args.evaluation_source_sha256,allocation_sha256=ah,outputs={p.name:dict(sha256=E.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()}))
 print(json.dumps(E.clean(dict(status=report['status'],all=scopes['all'],seconds=report['seconds']))))
if __name__=='__main__':main()
