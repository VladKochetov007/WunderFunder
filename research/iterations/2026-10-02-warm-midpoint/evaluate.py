"""One fixed half-parameter midpoint per seed/fold; paired descriptive no-fit screen."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
B=ROOT/'research/iterations/2026-10-01-ensemble-backbone';OLD=ROOT/'research/iterations/2026-10-01-overnight';EDA=ROOT/'research/iterations/2026-10-01-error-analysis'
BP=B/'evaluate.py';BSHA='c5f3ecc795ebd680872c39a652075a894192bfe9b4e654eca7e826c0310980e9'
LB=ROOT/'research/iterations/2026-10-01-late-atom-head/input-bindings.json';LSHA='573cd5cbaaddbda844abd4fd6736fe01de01726b2ceb118ed1b5b4d1682b8ff8'
assert hashlib.sha256(BP.read_bytes()).hexdigest()==BSHA
spec=importlib.util.spec_from_file_location('warm_midpoint_frozen_metric',BP);E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
W=ROOT/'research/iterations/2026-10-02-ordinary-warm'
WSHA='f29b97efa4e767cb85f83e9c44dcee678ed4b84e9f9e28dce18e31f22d6cf865'
W4=W.with_name('2026-10-02-ordinary-warm-seed4');W4SHA='adcf8d422dd6bc63bf76b5c72e087d830c9984c1e8ca03d96f72789239fdf6c5'
ANCHORS=('on_ctl128s3','on_ctl128s4','on_ctlc500');MODELS=('midpoint_seed3_4','midpoint_seed4_4','current_student4','original_seed4_4','warm_seed3_4','warm_seed4_4')
SCOPES=('all',)
PAIRS={'primary':(0,2),'confirmation':(1,3),'seed3_vs_warm':(0,4),'seed4_vs_warm':(1,5)}
DONE='COMPLETE_0_FITS_4_OOF_NO_METRICS'

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
 assert frozen['source_sha256'][str((W/'evaluate.py').relative_to(ROOT))]==WSHA and frozen['source_sha256'][str((W4/'evaluate.py').relative_to(ROOT))]==W4SHA
 for rel,h in frozen['source_sha256'].items():pin(E.path_for(rel),h)
 apath=HERE/'training-allocation.json';pin(apath,frozen['training_allocation_sha256']);ah=tracked[str(apath)];a=json.loads(apath.read_text())
 assert a['execution_authorized'] and a['success_status']==DONE
 for rel,h in a['sources'].items():pin(E.path_for(rel),h)
 # Root pipeline already enforces large training-input stat identities. Bind those
 # manifests here without reading unused teacher tensors or the original parquet.
 for r in a['input_bindings']:pin(E.path_for(r['path']),r['sha256'])
 cp=HERE/'commands.json';assert tracked[str(cp)]==a['sources'][str(cp.relative_to(ROOT))];commands=json.loads(cp.read_text())
 expected=[(j['name'],j['stage']) for j in commands['jobs']]
 assert expected==[(f'oct02_wm_s{seed}h2_f{f}_OOF','OOF') for seed in (3,4) for f in (0,1)]
 assert [(j['name'],j['stage']) for j in a['jobs']]==expected
 for j,c in zip(a['jobs'],commands['jobs']):assert j['argv']==c['argv']
 dp=HERE/'execution-01/result.json';pin(dp,E.sha(dp));d=json.loads(dp.read_text())
 assert a['execution_directory']==str(dp.parent.relative_to(ROOT)) and d['status']==DONE and d['allocation_sha256']==ah
 assert d['failure'] is None and (d['fits'],d['OOF_jobs'],d['metric_calls'],d['retries'])==(0,4,0,0)
 assert d['runner_sha256']==a['sources'][str((HERE/'run_queue.py').relative_to(ROOT))]
 assert [(j['name'],j['stage']) for j in d['jobs']]==expected
 outputs={}
 for j,plan in zip(d['jobs'],a['jobs']):
  assert j['exit_code']==0 and j['failure'] is None and j['remaining_scope_pids']==[] and not j.get('cleanup_error')
  wr=j['worker_result'];p=E.path_for(wr['path']);pin(p,wr['sha256']);w=json.loads(p.read_text())
  assert w['status']=='COMPLETE' and w['allocation_sha256']==ah and w['name']==j['name'] and w['stage']==j['stage'] and w['returncode']==0
  assert w['argv']==plan['argv'] and set(w['outputs'])==set(plan['required_outputs'])
  for rel,r in w['outputs'].items():
   p=E.path_for(rel);assert rel not in outputs and p.stat().st_size==r['bytes'];pin(p,r['sha256']);outputs[rel]=r
 # Midpoint construction receipts are checked with each target-free producer receipt below.
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
 ep=oldpin(EDA/'extraction-01/scored-rows.npz');er=oldpin(EDA/'extraction-01/receipt.json')
 assert json.loads(er.read_text())['outputs'][ep.name]['sha256']==tracked[str(ep)]
 prior_stats=[];prior_reports={};keys=counts=None
 for study,source,status,names in ((W,WSHA,'COMPLETE_FIXED_ORDINARY_WARM_SCREEN',['ordinary_warm4','interval_instantaneous4','current_student4']), (W4,W4SHA,'COMPLETE_FIXED_ORDINARY_WARM_SEED4_SCREEN',['ordinary_warm_seed4_4','original_seed4_4','current_student4'])):
  sp=study/'evaluation-01/sufficient.npz';rp=sp.with_name('receipt.json');pp=sp.with_name('report.json')
  for p in (sp,rp,pp):pin(p,frozen['input_sha256'][str(p.relative_to(ROOT))])
  receipt=json.loads(rp.read_text());report=json.loads(pp.read_text())
  assert receipt['status']==report['status']==status and receipt['source_sha256']==report['source_sha256']==source
  for p in (sp,pp):assert receipt['outputs'][p.name]==dict(sha256=tracked[str(p)],bytes=p.stat().st_size)
  with np.load(sp,allow_pickle=False) as z:
   old=z['chain_stats'][:,0];cc=z['chain_counts'][:,:1];kk=z['chain_metadata']
   assert z['model_names'].tolist()==names and z['scope_names'].tolist()[0]=='all'
  assert old.shape==(250,3,2,6) and kk.shape==(250,3) and cc.shape==(250,1) and cc.sum()==1614484 and np.isfinite(old).all()
  if keys is None:keys=kk;counts=cc
  else:np.testing.assert_array_equal(keys,kk);np.testing.assert_array_equal(counts,cc)
  for take,label in ((np.ones(250,bool),'all'),(keys[:,0]==0,'0'),(keys[:,0]==1,'1')):
   prior=report['comparisons']['primary']['scopes']['all'];prior=prior if label=='all' else prior['folds'][label]
   for i,name in ((0,'candidate'),(2 if study==W else 1,'control')):
    calculated=E.metric(old[take,i].sum(0));assert abs(calculated['WP']-prior[name]['WP'])<2e-12
    np.testing.assert_allclose(calculated['rho'],prior[name]['rho'],atol=2e-12,rtol=0)
  prior_stats.append(old);prior_reports[3 if study==W else 4]=report
 np.testing.assert_allclose(prior_stats[0][:,2],prior_stats[1][:,2],atol=2e-8,rtol=1e-12)
 assert abs(E.metric(prior_stats[0][:,2].sum(0))['WP']-E.BASELINE_WP)<2e-12
 assert abs(E.metric(prior_stats[1][:,1].sum(0))['WP']-.7022343699230701)<2e-12
 checkp=HERE/'prepare-01/checkpoints.json';pin(checkp,frozen['input_sha256'][str(checkp.relative_to(ROOT))]);check=json.loads(checkp.read_text())
 assert check['status']=='PASS_FIXED_WARM_MIDPOINT_CHECKPOINTS' and check['commands_sha256']==tracked[str(cp)]
 assert check['checker_sha256']==a['sources'][str((HERE/'check_checkpoint.py').relative_to(ROOT))]
 assert check['forecast_fits']==check['model_forward_calls']==check['data_reads']==check['metrics']==0
 expected_runs=[f'oct02_wm_s{seed}h2_f{f}' for seed in (3,4) for f in (0,1)]
 assert commands['alpha']==.5 and commands['counts']==dict(fits=0,OOF=4)
 assert [r['name'] for r in check['checkpoints']]==[p['name'] for p in commands['pairs']]==expected_runs
 integrity={}
 for pair,q in zip(commands['pairs'],check['checkpoints']):
  seed=pair['seed'];fold=pair['fold'];run=pair['name'];assert run==f'oct02_wm_s{seed}h2_f{fold}'
  assert q['seed']==seed and q['fold']==fold and q['alpha']==.5
  assert q['mean_std_byte_identical'] and q['all10_parameters_exact_midpoint'] and q['ordinary_strict_load']
  checkpoint=ROOT/pair['endpoint'];assert checkpoint==ROOT/f'artifacts/runs/{run}/model_ep1.pt';pin(checkpoint,q['checkpoint']['sha256'])
  assert q['checkpoint']==dict(path=pair['endpoint'],sha256=tracked[str(checkpoint)],epoch=1)
  for role,source_run in (('old',f'cont_sh256_s{seed}h2_f{fold}'),('warm',f'oct02_ow_s{seed}h2_f{fold}')):
   b=pair[role];assert b['path']==f'artifacts/runs/{source_run}/model_ep1.pt';pin(ROOT/b['path'],b['sha256'])
   assert q[f'{role}_checkpoint']==dict(path=b['path'],sha256=b['sha256'],epoch=1)
  assert q['warm_checkpoint']==prior_reports[seed]['fit_integrity'][f'oct02_ow_s{seed}h2_f{fold}']['checkpoint']
  crp=checkpoint.with_name('construction.json');pin(crp,q['construction_receipt_sha256']);cr=json.loads(crp.read_text())
  assert cr['status']=='CONSTRUCTED_FIXED_PARAMETER_MIDPOINT' and cr['name']==run and cr['seed']==seed and cr['fold']==fold and cr['alpha']==.5
  assert cr['checkpoint']==q['checkpoint'] and cr['old_checkpoint']==q['old_checkpoint'] and cr['warm_checkpoint']==q['warm_checkpoint']
  assert cr['mean_std_byte_identical'] and cr['all10_parameters_exact_midpoint'] and cr['ordinary_strict_load']
  assert cr['commands_sha256']==tracked[str(cp)] and cr['constructor_sha256']==a['sources'][str((HERE/'construct.py').relative_to(ROOT))] and cr['midpoint_source_sha256']==a['sources'][str((HERE/'midpoint.py').relative_to(ROOT))]
  assert cr['forecast_fits']==cr['metrics']==0 and cr['targets_read'] is False
  integrity[run]=q
 with np.load(ep,allow_pickle=False) as z:
  y=z['y'];meta=z['metadata'];assert z['metadata_names'].tolist()==['cache_position','row_group','step','chain','fold','market_class_retrospective']
 assert meta.shape==(1614484,6) and y.shape==(1614484,2) and np.isfinite(y).all()
 stats=np.zeros((250,1,6,2,6));stats[:,0,2]=prior_stats[0][:,2];stats[:,0,3]=prior_stats[1][:,1];stats[:,0,4]=prior_stats[0][:,0];stats[:,0,5]=prior_stats[1][:,0]
 checkcounts=np.zeros((2,250,1),dtype=np.int64);ki={tuple(k):i for i,k in enumerate(keys)};assert len(ki)==250;sequences=0
 out.mkdir();E.write(out/'started.json',dict(source_sha256=E.sha(__file__),evaluation_source_sha256=args.evaluation_source_sha256,allocation_sha256=ah,input_sha256=tracked))
 for seed in (3,4):
  for f in (0,1):
   run=f'oct02_wm_s{seed}h2_f{f}';p=produced(HERE/'cache'/f'pred_{run}_valid.npy',outputs);rp=produced(p.with_suffix('.inputs.json'),outputs);r=json.loads(rp.read_text())
   checkpoint=ROOT/f'artifacts/runs/{run}/model_ep1.pt'
   assert r['run']==run and r['fold']==f and r['epoch']==1 and r['prediction_path']==str(p) and r['prediction_sha256']==tracked[str(p)]
   assert r['checkpoint']==str(checkpoint) and r['checkpoint_sha256']==tracked[str(checkpoint)] and r['targets_read'] is False and r['metrics_computed']==0
   assert r['script_sha256']==a['sources'][str((HERE/'predict_fixed.py').relative_to(ROOT))] and r['midpoint_source_sha256']==a['sources'][str((HERE/'midpoint.py').relative_to(ROOT))]
   assert r['seed']==seed and r['construction_receipt_sha256']==integrity[run]['construction_receipt_sha256']
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
   take=meta[:,4]==f;mm=meta[take];yy=y[take]
   cuts=np.r_[0,np.flatnonzero(mm[1:,0]!=mm[:-1,0])+1,len(mm)];assert len(cuts)-1==(425 if f==0 else 435)
   seen=[]
   for lo,hi in zip(cuts[:-1],cuts[1:]):
    rows=mm[lo:hi];pos=int(rows[0,0]);cl=int(rows[0,5]);chain=int(rows[0,3]);steps=rows[:,2];seen.append(pos)
    assert int(groups[pos])==int(rows[0,1]) and info[int(groups[pos])]==(f,chain,True)
    assert np.all(rows[:,0:2]==rows[0,0:2]) and np.all(rows[:,3:]==rows[0,3:]) and np.all(steps>=99) and np.all(steps<20000) and np.all(np.diff(steps)>0)
    parts=[arr[pos,steps] for arr in anchors];assert all(np.isfinite(v).all() for v in parts)
    blended=E.blend([prediction[pos,steps],*parts]);j=ki[f,chain,cl]
    checkcounts[seed-3,j,0]+=hi-lo;stats[j,0,seed-3]+=E.sufficient(yy[lo:hi],blended)
   assert seen==[i for i,g in enumerate(groups) if info[int(g)][0]==f and info[int(g)][2]];sequences+=len(seen)
 assert sequences==1720
 for m in (0,1):np.testing.assert_array_equal(checkcounts[m],counts)
 for m in range(6):np.testing.assert_allclose(np.take(stats[:,:,m],[0,1,3],axis=-1),np.take(stats[:,:,2],[0,1,3],axis=-1),atol=2e-8,rtol=0)
 assert np.isfinite(stats).all() and E.metric(stats[:,0].sum(0))['valid'].all()
 for f in (0,1):assert E.metric(stats[keys[:,0]==f,0].sum(0))['valid'].all()
 draws,strata=E.resamples(keys);comparisons={}
 for label,pair in PAIRS.items():
  scopes={}
  for s,name in enumerate(SCOPES):
   selected=stats[:,s][:,list(pair)];scopes[name]=E.comparison(selected,counts[:,s],draws)
   scopes[name]['folds']={str(f):E.comparison(selected[keys[:,0]==f],counts[keys[:,0]==f,s],draws[:,keys[:,0]==f]) for f in (0,1)}
  comparisons[label]=dict(candidate=MODELS[pair[0]],control=MODELS[pair[1]],scopes=scopes)
 primary=comparisons['primary']['scopes']['all'];confirmation=comparisons['confirmation']['scopes']['all']
 gate=dict(seed3_pooled=primary['delta_WP']>=.0001,seed4_pooled=confirmation['delta_WP']>0,
  seed3_folds=all(primary['folds'][str(f)]['delta_WP']>0 for f in (0,1)),seed4_folds=all(confirmation['folds'][str(f)]['delta_WP']>0 for f in (0,1)))
 gate['passed']=all(gate.values())
 report=dict(status='COMPLETE_FIXED_WARM_MIDPOINT_SCREEN',models=MODELS,parameter_fraction=.5,comparisons=comparisons,sequences_per_seed=860,chains=250,scored_rows_per_seed=1614484,fits=0,OOF_jobs=4,construction_integrity=integrity,baseline_reproduced=E.BASELINE_WP,original_seed4_reproduced=.7022343699230701,gate=gate,uncertainty=dict(draws=E.BOOTSTRAPS,seed=E.SEED,strata=strata,interpretation='Same paired whole-chain fold/class-stratified draws across four prespecified contrasts, conditional marginal95% intervals; folds and examples are reused across seeds, not independent seed-population evidence.'),interpretation='Adaptive child after failed warm seed4 confirmation, which remains a negative result. One half-parameter interpolation of originalH2 and warmH2 separately for each seed/fold, no fit, coefficient/epoch search or cross-seed blend. Exact equal-quarter fixed anchors. Gate is a declared spending condition, not a statistical significance test or automatic deployment authorization.',source_sha256=E.sha(__file__),frozen_metric_sha256=BSHA,evaluation_source_sha256=args.evaluation_source_sha256,allocation_sha256=ah,input_sha256=tracked,seconds=time.monotonic()-started)
 for p,state in states.items():
  st=Path(p).stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==state,('input changed',p)
 np.savez_compressed(out/'sufficient.npz',chain_stats=stats,chain_counts=counts,chain_metadata=keys,scope_names=np.array(SCOPES),model_names=np.array(MODELS))
 E.write(out/'report.json',report);E.write(out/'receipt.json',dict(status=report['status'],source_sha256=E.sha(__file__),evaluation_source_sha256=args.evaluation_source_sha256,allocation_sha256=ah,outputs={p.name:dict(sha256=E.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()}))
 print(json.dumps(E.clean(dict(status=report['status'],comparisons=comparisons,seconds=report['seconds']))))
if __name__=='__main__':main()
