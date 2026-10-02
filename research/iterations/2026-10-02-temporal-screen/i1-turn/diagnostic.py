"""One root-allocated CPU diagnostic; no model inference or forecast WP calculation."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, resource, signal, sys, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
ERROR=ROOT/'research/iterations/2026-10-01-error-analysis'
HELPER=ERROR/'analyze_errors.py'
spec=importlib.util.spec_from_file_location('existing_error_math',HELPER)
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
from features import full
DIM=22;QY=18;QP=20
# 0=intercept,1:16=control15,16=A,17=placebo,18:20=clip(y),20:22=clip(p).
ARMS=('control','turn','placebo')
CONTROL=np.arange(1,16);ARM_INDICES=(CONTROL,np.r_[CONTROL,16],np.r_[CONTROL,17])
RAW_COLUMNS=['i0_p0','i0_p11','i1_p0','i1_p11']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def sign(chain):
 return 1. if (int(hashlib.sha256(f'i1turn-negative-control:20261002:{int(chain)}'.encode()).hexdigest(),16)&1) else -1.

def design(history,predictions,y,chain):
 # history: raw4, d2, intervalnet2,E,count,age,A; all states current-row inclusive.
 p=predictions[:,0,:].astype(np.float64)
 sd=np.std(predictions[:,3:7,:].astype(np.float64),axis=1)
 control=np.column_stack((p,sd,history[:,:8],np.log1p(history[:,8:11])))
 assert control.shape[1]==15
 return np.column_stack((np.ones(len(y)),control,history[:,11],sign(chain)*history[:,11],
                         np.clip(y,-2,2),np.clip(p,-2,2)))

def crossfit(grams,chain_keys,class_counts,class_values,bootstrap=2000):
 # grams populations all/class3/class2,chains,targets,22,22. Fit only all.
 assert grams.shape==(3,len(chain_keys),2,DIM,DIM)
 halves,dominant=H.make_halves(chain_keys,class_counts,class_values,seed=20261001)
 n=len(chain_keys);sse=np.zeros((3,n,2,3));constant=np.zeros((3,n,2));records=[]
 for fold in (0,1):
  for held in (0,1):
   tr=(chain_keys[:,0]==fold)&(halves!=held);te=(chain_keys[:,0]==fold)&(halves==held)
   assert tr.any() and te.any() and not np.any(tr&te)
   for target in (0,1):
    G=grams[0,tr,target].sum(0)
    alpha,qinfo=H.q_definition(G,target,prediction_index=QP+target,target_index=QY+target)
    base=np.zeros(DIM);base[0]=qinfo['q_fit_mean']
    for pop in range(3):constant[pop,te,target]=H.quadratic_chain(grams[pop,te,target],alpha-base)
    for ai,indices in enumerate(ARM_INDICES):
     beta,info=H.fit_ridge(G,alpha,indices,lam=.01)
     for pop in range(3):sse[pop,te,target,ai]=H.quadratic_chain(grams[pop,te,target],alpha-beta)
     records.append(dict(fold=fold,heldout_half=held,target=target,arm=ARMS[ai],training_chains=chain_keys[tr,1],heldout_chains=chain_keys[te,1],q=qinfo,ridge=info,coefficients=beta))
    # Diagnostic rank of candidate standardized covariance, no fit/config change.
    ix=ARM_INDICES[1];mu=G[0,ix]/G[0,0];cov=G[np.ix_(ix,ix)]/G[0,0]-np.outer(mu,mu)
    scales=np.sqrt(np.maximum(np.diag(cov),0));active=scales>1e-12;corr=np.zeros_like(cov)
    if active.any():corr[np.ix_(active,active)]=cov[np.ix_(active,active)]/np.outer(scales[active],scales[active])
    eig=np.linalg.eigvalsh(corr)
    records[-1]['candidate_covariance_eigenvalues']=eig
 return sse,constant,halves,records

def summarize(sse,constant,indices,pop=0):
 ss=sse[pop,indices].sum(0);bs=constant[pop,indices].sum(0)
 if not (bs>1e-12).all():return dict(defined=False,chains=int(np.count_nonzero(indices)))
 inc=(ss[:,0,None]-ss[:,1:])/bs[:,None]
 return dict(defined=True,chains=int(np.count_nonzero(indices)),SSE=ss,constant_SSE=bs,
             incremental_skill_targets=inc,turn_incremental_skill=float(inc[:,0].mean()),
             placebo_incremental_skill=float(inc[:,1].mean()))

def interval(sse,constant,keys,class_counts,class_values,replicates=2000):
 dominant=class_values[np.argmax(class_counts,axis=1)];rng=np.random.default_rng(20261002)
 strata=[np.flatnonzero((keys[:,0]==f)&(dominant==c)) for f in (0,1) for c in class_values]
 strata=[z for z in strata if len(z)];values=[]
 for _ in range(replicates):
  ix=np.concatenate([rng.choice(z,len(z),replace=True) for z in strata])
  ss=sse[0,ix].sum(0);bs=constant[0,ix].sum(0)
  values.append(float(((ss[:,0]-ss[:,1])/bs).mean()))
 return dict(replicates=replicates,seed=20261002,conditional_95=np.quantile(values,[.025,.975]),definition='whole-chain, outerfold×dominantclass strata; fixed nuisance fits, no refitting')

def write(p,x):
 with p.open('x') as f:json.dump(H.serial(x),f,indent=2,allow_nan=False);f.write('\n')

def deadline(*_):
 raise TimeoutError("300-second wall cap")

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
 signal.signal(signal.SIGALRM,deadline);signal.alarm(300);started=time.monotonic();plan=json.loads(a.plan.read_text())
 assert plan['execution_authorized'] is True and plan['seconds_max']==300 and plan['memory_bytes_max']==2*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):assert 0<int(os.environ.get(key,'99'))<=2
 for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h,rel
 out=a.output_dir.resolve();assert out.is_relative_to(HERE) and out!=HERE and not out.exists();out.mkdir()
 result=dict(status='RUNNING',sources=plan['source_sha256'],plan_sha256=sha(a.plan),forecast_fits=0,forecast_inference=0,forecast_WP_queries=0)
 write(out/'started.json',result)
 try:
  receipt_path=ERROR/'extraction-01/receipt.json';binding_path=ERROR/'extract-binding.json';rows_path=ERROR/'extraction-01/scored-rows.npz'
  for p in (receipt_path,binding_path):assert sha(p)==plan['input_sha256'][str(p.relative_to(ROOT))]
  receipt=json.loads(receipt_path.read_text());binding=json.loads(binding_path.read_text());assert sha(rows_path)==receipt['outputs']['scored-rows.npz']['sha256']
  tracked_inputs={}
  for p in (receipt_path,binding_path,rows_path):
   s=p.stat();tracked_inputs[str(p)]=(str(p.resolve()),s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
  raw_path=ROOT/'datasets/valid.parquet';rawinfo=binding['raw_input'];st=raw_path.stat()
  assert str(raw_path.resolve())==rawinfo['resolved'] and st.st_size==rawinfo['bytes'] and st.st_mtime_ns==rawinfo['mtime_ns']
  raw_identity=(str(raw_path.resolve()),st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)
  with np.load(rows_path,allow_pickle=False) as z:
   assert z['metadata_names'].tolist()==['cache_position','row_group','step','chain','fold','market_class_retrospective']
   md=z['metadata']
  assert md.shape==(1614484,6) and md.dtype==np.int32
  bounds=np.r_[0,np.flatnonzero(md[1:,0]!=md[:-1,0])+1,len(md)];assert len(bounds)==861
  first=md[bounds[:-1]];assert len(np.unique(first[:,0]))==len(np.unique(first[:,1]))==860
  keys,codes=np.unique(md[:,[4,3]],axis=0,return_inverse=True);assert len(keys)==250
  classes=np.unique(md[:,5]);cc=np.zeros((250,len(classes)),np.int64)
  for ci,c in enumerate(classes):cc[:,ci]=np.bincount(codes[md[:,5]==c],minlength=250)
  import pyarrow as pa,pyarrow.parquet as pq
  pa.set_cpu_count(1);pa.set_io_thread_count(1);pf=pq.ParquetFile(raw_path)
  assert pf.num_row_groups==1873 and all(x in pf.schema_arrow.names for x in RAW_COLUMNS)
  history=np.empty((len(md),12),np.float64);nonzero_prefix=0;event_pairs=0
  for seq,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
   b=md[lo:hi];assert np.all(b[:,[0,1,3,4]]==b[0,[0,1,3,4]]) and np.all(np.diff(b[:,2])>0)
   steps=b[:,2];assert steps[0]>=99 and steps[-1]<20000
   table=pf.read_row_group(int(b[0,1]),columns=RAW_COLUMNS,use_threads=False);assert table.num_rows==20000
   raw=np.column_stack([table.column(c).to_numpy() for c in RAW_COLUMNS]).astype(np.float32,copy=False)
   f=full(raw);assert np.isfinite(f).all() and np.all(f[:,4]>=0) and np.max(abs(f[:,7]))<=1+1e-12
   history[lo:hi,:4]=raw[steps];history[lo:hi,4:]=f[steps]
   nonzero_prefix+=int(np.count_nonzero(f[:,7]));event_pairs+=int(np.count_nonzero((f[:,5]>=2)&np.any(f[:,:2]!=0,axis=1)))
   if seq%100==0:print(json.dumps(dict(sequences=seq+1,total=860,seconds=time.monotonic()-started)),flush=True)
  result.update(retrospective_class_metadata_read=True,prefix_rows=17200000,prefix_A_nonzero_rows=nonzero_prefix,adjacent_event_pairs=event_pairs,sequences=860,scored_rows=len(md),chains=250,raw_columns=RAW_COLUMNS)
  if nonzero_prefix==0:
   result.update(status='STOP_UNINFORMATIVE_A_IDENTICALLY_ZERO',labels_read=False,predictions_read=False,gate_pass=False)
  else:
   with np.load(rows_path,allow_pickle=False) as z:
    assert z['prediction_names'].tolist()==['current_student4','strong4','fixed3','shallow_seed3','plain_seed3','plain_seed4','chunk500','shallow_seed4_auxiliary']
    y=z['y'];pred=z['predictions']
   assert y.shape==(len(md),2) and pred.shape==(len(md),8,2) and np.isfinite(y).all() and np.isfinite(pred).all()
   grams=np.zeros((3,250,2,DIM,DIM),np.float64);coverage=np.zeros((250,2,4),np.float64);quiet_counts=np.zeros(250,np.int64)
   for lo,hi in zip(bounds[:-1],bounds[1:]):
    ii=np.arange(lo,hi);ii=ii[history[ii,10]>=64]
    if not len(ii):continue
    ci=int(codes[lo]);chain=int(md[lo,3]);quiet_counts[ci]+=len(ii)
    Z=design(history[ii],pred[ii],y[ii],chain);w=np.abs(Z[:,QY:QY+2]);cl=md[ii,5]
    for target in (0,1):
     for pop,mask in enumerate((np.ones(len(ii),bool),cl==3,cl==2)):
      zz=Z[mask];grams[pop,ci,target]+=zz.T@(zz*w[mask,target,None])
     coverage[ci,target]=coverage[ci,target]+np.array([w[:,target].sum(),w[history[ii,11]==0,target].sum(),w[history[ii,9]<2,target].sum(),np.sum(w[:,target]*history[ii,11]**2)])
   assert np.isfinite(grams).all()
   sse,constant,halves,fits=crossfit(grams,keys,cc,classes)
   pooled=summarize(sse,constant,np.ones(250,bool));folds={str(f):summarize(sse,constant,keys[:,0]==f) for f in (0,1)}
   assert pooled['defined'] and all(z['defined'] for z in folds.values())
   gate=pooled['turn_incremental_skill']>=.002 and all(z['turn_incremental_skill']>0 for z in folds.values()) and pooled['turn_incremental_skill']>pooled['placebo_incremental_skill']
   result.update(status='COMPLETE_FIXED_I1_TURN_DIAGNOSTIC',labels_read=True,predictions_read=True,gate_pass=gate,pooled=pooled,folds=folds,bootstrap=interval(sse,constant,keys,cc,classes),descriptive_slices={str(c):summarize(sse,constant,np.ones(250,bool),pop) for c,pop in ((3,1),(2,2))},coverage=dict(quiet_rows=int(quiet_counts.sum()),target_mass=coverage[:,:,0].sum(0),zero_A_weight_fraction=coverage[:,:,1].sum(0)/coverage[:,:,0].sum(0),fewer_than_two_events_weight_fraction=coverage[:,:,2].sum(0)/coverage[:,:,0].sum(0)),ridge_fits=fits,interpretation='Incremental held-chain normalized-response SSE skill, not WP or an attainable forecast gain; one fixed placebo is not a permutation test.')
   np.savez_compressed(out/'chain-statistics.npz',gram=grams,sse=sse,constant_sse=constant,chain_metadata=keys,chain_class_counts=cc,class_values=classes,halves=halves,coverage=coverage,quiet_rows=quiet_counts)
  st=raw_path.stat();assert raw_identity==(str(raw_path.resolve()),st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)
  for name,before in tracked_inputs.items():
   p=Path(name);s=p.stat();assert before==(str(p.resolve()),s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev),name
  for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h,rel
 except BaseException as exc:
  result.update(status='FAILED_STOP_NO_RETRY',failure=repr(exc));raise
 finally:
  result.update(seconds=time.monotonic()-started,peak_RSS_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  write(out/'result.json',result)
 print(json.dumps(dict(status=result['status'],gate_pass=result.get('gate_pass'),seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
