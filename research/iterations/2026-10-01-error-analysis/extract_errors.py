"""One CPU raw/OOF pass. No Torch, inference, fitting, feature threshold search or WP readout.

Execution requires a root plan with exact source/bindings hashes. Schema is inspectable
without reading datasets via --schema. Scored-row features use only rows through t;
event-window anchor subsampling is explicitly retrospective descriptive alignment.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,resource,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
CONT=ROOT/'research/iterations/2026-10-01-continuation'
BINDINGS=CONT/'student-seed4/bindings-v2-01.json'
BINDINGS_SHA='e6be669708f25081362e4352da5a5d1e8e6dc8587868b95acded72e427c02f19'
MEMBERS=('cont_sh256_s3h2','on_ctl128s3','on_ctl128s4','on_ctlc500','on_ctld256','cont_sh256_s4h2')
PRED_NAMES=('current_student4','strong4','fixed3','shallow_seed3','plain_seed3','plain_seed4','chunk500','shallow_seed4_auxiliary')
FEATURE_NAMES=(
 [f'event_{s}' for s in ('i0px','i0bidvol','i0askvol','i0trade_px','i0trade_vol','i1px','i1vol','i1trade')]
 +[f'rate_{s}_rows{w}' for s in ('i0px','i0trade_vol') for w in (32,256,2048)]
 +['log_since_i0px','log_since_i0trade_vol','log_since_i1px']
 +['step_fraction','phase_chunk2000','phase_chunk500']
 +['delta_i0p0','delta_i0p11','delta_i1p0','delta_i1p11']
 +['rms_delta_i0px','rms_delta_i1px','rms_held_i0px','changed_i0px_count','changed_i1px_count','cumulative_i0px_events'])
META_NAMES=('cache_position','row_group','step','chain','fold','market_class_retrospective')
EVENT_META_NAMES=('cache_position','row_group','step','anchor_step','offset','chain','fold','market_class_retrospective')
OFFSETS=np.array([-32,-8,-1,0,1,4,16,64],np.int32)
EXPECTED_ROWS=1614484;EXPECTED_SEQ=860;MAX_EVENTS=860*64*8
SOURCES={'src/wnn/data.py':'a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00','src/wnn/features.py':'284c119d150b74bff0ccc2628bc2a195ca61c351a628b085b7d9dd85512bfeee','scripts/fold_eval.py':'d05791f5f8c6ea64d9863d4f4b45430eb46b9d0b29ccbdac1a4ff5b36a2573c0'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def schema():
 return {'main':{'X':'float32[N,30]','predictions':'float32[N,8,2]','y':'float32[N,2]','metadata':'int32[N,6]'},'feature_names':FEATURE_NAMES,'prediction_names':PRED_NAMES,'metadata_names':META_NAMES,'events':{'metadata':'int32[M,8]','y':'float32[M,2]','student_prediction':'float32[M,2]','anchor_price_code_delta':'float32[M,2]','row_event_flags':'uint8[M,8]','is_scored':'bool[M]','anchor_is_scored':'bool[M]','anchor_model_delta':'float32[M,2]','anchor_price_code_sign':'int8[M,2]','anchor_model_sign':'int8[M,2]'},'event_metadata_names':EVENT_META_NAMES,'anchor_delta_coordinates':['i0_p0','i0_p11'],'event_offsets':OFFSETS.tolist(),'max_anchors_per_sequence':64,'member_deviations':'Derive predictions[:,3:7,:]-predictions[:,0:1,:] (8columns) without duplicated storage. Base prediction/confidence is derivable from predictions[:,0,:].','sequence_statistics':'float64[860,8,2,6], exact frozen fold_eval.sums; no pooled score computed here.','chain_Gram':{'shape':[250,2,261,261],'dtype':'float64','weight':'abs(clip(y_t,-2,2)), separately for each target','blocks':{'intercept':[0,1],'base7':[1,8],'current211':[8,219],'history30':[219,249],'deviations10':[249,259],'clipped_y2':[259,261]},'arithmetic':'Z.T @ (Z * target_weight[:,None]), float64, summed per sequence into its whole-chain matrix','normalization':'Current211 uses fixed same-fold checkpoint float32 mean/std; no validation-fitted normalization.'},'class_usage':'Retrospective diagnostic only, not a causal feature or available serving input.','event_sampling':'Deterministic linspace over all observed joint i0price-code events in each sequence; selected anchors use whole-sequence event locations and are descriptive, not causal inputs.'}

def causal_summary(raw):
 from wnn.features import event_features
 e=event_features(raw);t=np.arange(len(raw),dtype=np.int64);flags=e[:,66:74]
 cols=[flags]
 for flag in (0,4):
  cs=np.r_[0,np.cumsum(flags[:,flag],dtype=np.int64)]
  for w in (32,256,2048):
   lo=np.maximum(0,t+1-w);cols.append(((cs[t+1]-cs[lo])/(t+1-lo)).astype(np.float32)[:,None])
 cols += [e[:,74:77],(t/19999.).astype(np.float32)[:,None],((t%2000)/1999.).astype(np.float32)[:,None],((t%500)/499.).astype(np.float32)[:,None],e[:,[0,11,44,55]]]
 for a in (e[:,:22],e[:,44:66],e[:,77:99]):cols.append(np.sqrt(np.mean(a.astype(np.float64)**2,axis=1)).astype(np.float32)[:,None])
 cols += [(e[:,:22]!=0).sum(1).astype(np.float32)[:,None],(e[:,44:66]!=0).sum(1).astype(np.float32)[:,None],np.cumsum(flags[:,0],dtype=np.int64).astype(np.float32)[:,None]]
 x=np.concatenate(cols,axis=1);assert x.shape==(len(raw),30) and x.dtype==np.float32 and np.isfinite(x).all()
 return x,flags.astype(np.uint8),e[:,[0,11]],e

def prediction_stack(parts):
 # Preserve exact existing float32 member addition order, including strong4.
 p0,p1,p2,p3,p4,p5=parts
 return np.stack([sum([p0,p1,p2,p3])/4,sum([p4,p1,p2,p3])/4,sum([p1,p2,p3])/3,p0,p1,p2,p3,p5],axis=1)

def event_indices(flags):
 anchors=np.flatnonzero(flags[:,0])
 if len(anchors)>64:anchors=anchors[np.linspace(0,len(anchors)-1,64,dtype=np.int64)]
 aa=np.repeat(anchors,8);off=np.tile(OFFSETS,len(anchors));steps=aa+off;ok=(steps>=0)&(steps<len(flags))
 return aa[ok],off[ok],steps[ok]

def save_npz(path,**arrays):
 assert not path.exists();partial=path.with_suffix('.npz.partial');assert not partial.exists()
 with partial.open('xb') as f:np.savez_compressed(f,**arrays)
 partial.replace(path)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--schema',action='store_true');ap.add_argument('--plan',type=Path);ap.add_argument('--plan-sha256');ap.add_argument('--output-dir',type=Path);a=ap.parse_args()
 if a.schema:print(json.dumps(schema(),indent=2));return
 assert a.plan and a.plan_sha256 and a.output_dir
 assert sha(a.plan)==a.plan_sha256;plan=json.loads(a.plan.read_text());assert plan['extraction_authorized'] is True and plan['source_sha256']==sha(__file__)
 assert sha(ROOT/plan['common_plan_path'])==plan['common_plan_sha256']
 assert plan['prior_bindings_sha256']==BINDINGS_SHA and sha(BINDINGS)==BINDINGS_SHA
 assert plan['expected_scored_rows']==EXPECTED_ROWS and plan['expected_pure_sequences']==EXPECTED_SEQ
 out=a.output_dir.resolve();assert out.is_relative_to(HERE) and out!=HERE and not out.exists()
 assert not os.environ.get('CUDA_VISIBLE_DEVICES','')
 import pyarrow as pa
 pa.set_cpu_count(1);pa.set_io_thread_count(1)
 sys.path.insert(0,str(ROOT/'src'));from wnn.data import SequenceFile
 tracked={}
 def pin(path,digest):
  assert sha(path)==digest,str(path);tracked[str(path)]=digest
 for rel,h in SOURCES.items():pin(ROOT/rel,h)
 pin(BINDINGS,BINDINGS_SHA);pin(a.plan,a.plan_sha256);pin(Path(__file__),plan['source_sha256']);pin(ROOT/plan['common_plan_path'],plan['common_plan_sha256'])
 bound=json.loads(BINDINGS.read_text())
 for rel,rec in bound['data'].items():assert (ROOT/rel).stat().st_size==rec['bytes'];pin(ROOT/rel,rec['sha256'])
 raw_path=ROOT/'datasets/valid.parquet';rawrec=plan['raw_input'];st=raw_path.stat()
 assert str(raw_path.resolve())==rawrec['resolved'] and (st.st_size,st.st_mtime_ns)==(rawrec['bytes'],rawrec['mtime_ns'])
 # One streaming identity hash before/after; no new raw feature cache.
 pin(raw_path,rawrec['sha256'])
 normalizers={}
 for fold in (0,1):
  nr=plan['normalizers'][str(fold)];npz=ROOT/nr['path'];pin(npz,nr['sha256'])
  with np.load(npz,allow_pickle=False) as z:mean=z['mean'];std=z['std']
  assert mean.shape==std.shape==(211,) and mean.dtype==std.dtype==np.float32 and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
  normalizers[fold]=(mean,std)
 metric_spec=importlib.util.spec_from_file_location('frozen_error_sums',ROOT/'scripts/fold_eval.py');fe=importlib.util.module_from_spec(metric_spec);metric_spec.loader.exec_module(fe)
 with np.load(ROOT/'cache/valid_targets.npz',allow_pickle=False) as z:rg=z['row_group'];seq_ix=z['seq_ix'];cls=z['market_class'];y=z['y'];mask=z['mask']
 assert y.shape==(1873,20000,2) and mask.shape==(1873,20000) and len(np.unique(rg))==1873
 with np.load(ROOT/'research/artifacts/market-classes-v1.npz',allow_pickle=False) as z:assert np.array_equal(cls,z['valid'][rg])
 with np.load(ROOT/'cache/chain_folds.npz',allow_pickle=False) as z:
  valid=z['split']=='valid';info={int(g):(int(f),int(c),bool(p)) for g,f,c,p in zip(z['row_group'][valid],z['fold'][valid],z['chain'][valid],z['pure'][valid],strict=True)}
 selected=[(fold,i) for fold in (0,1) for i,g in enumerate(rg) if info[int(g)][0]==fold and info[int(g)][2]]
 assert len(selected)==EXPECTED_SEQ and len({info[int(rg[i])][1] for _,i in selected})==250
 n=sum(int(np.count_nonzero(mask[i,99:])) for _,i in selected);assert n==EXPECTED_ROWS
 chain_keys=list(dict.fromkeys((fold,info[int(rg[i])][1]) for fold,i in selected));assert len(chain_keys)==250
 chain_index={key:j for j,key in enumerate(chain_keys)};class_values=np.unique(cls[[i for _,i in selected]]);class_index={int(c):j for j,c in enumerate(class_values)}
 group_sha=hashlib.sha256(rg.tobytes()).hexdigest();arrays={}
 for fold in (0,1):
  fold_rows=[i for i,g in enumerate(rg) if info[int(g)][0]==fold]
  for name in MEMBERS:
   rec=bound['predictions'][f'{name}_f{fold}'];p=ROOT/rec['path'];receipt=ROOT/rec['receipt'];pin(p,rec['sha256']);pin(receipt,rec['receipt_sha256']);m=json.loads(receipt.read_text())
   assert m['run']==f'{name}_f{fold}' and m['fold']==fold and m['epoch']==1 and m['rows']==fold_rows and m['row_groups']==rg[fold_rows].tolist()
   assert m['source_row_group_order_sha256']==group_sha and m['prediction_sha256']==rec['sha256'] and m['checkpoint_sha256']==rec['checkpoint_sha256']
   arr=np.load(p,mmap_mode='r',allow_pickle=False);assert arr.shape==(1873,20000,2) and arr.dtype==np.float32;arrays[name,fold]=arr
 out.mkdir();start=time.monotonic()
 (out/'started.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'plan_sha256':a.plan_sha256,'tracked_input_sha256':tracked,'schema':schema()},indent=2)+'\n')
 X=np.empty((n,30),np.float32);Y=np.empty((n,2),np.float32);Pred=np.empty((n,8,2),np.float32);meta=np.empty((n,6),np.int32)
 emeta=np.empty((MAX_EVENTS,8),np.int32);ey=np.empty((MAX_EVENTS,2),np.float32);ep=np.empty_like(ey);ed=np.empty_like(ey);ef=np.empty((MAX_EVENTS,8),np.uint8);em=np.empty(MAX_EVENTS,bool);eam=np.empty(MAX_EVENTS,bool);epdelta=np.empty((MAX_EVENTS,2),np.float32);ecsign=np.empty((MAX_EVENTS,2),np.int8);epsign=np.empty((MAX_EVENTS,2),np.int8)
 stats=np.empty((860,8,2,6),np.float64);smeta=np.empty((860,6),np.int32);counts=np.empty(860,np.int32);data=SequenceFile(raw_path);raw_digest=hashlib.sha256();cursor=ecursor=0;event_slots=0
 gram=np.zeros((250,2,261,261),np.float64);chain_counts=np.zeros(250,np.int64);chain_class_counts=np.zeros((250,len(class_values)),np.int64)
 from wnn.data import FEATURE_COLUMNS
 from wnn.features import FEATURE_NAMES as EVENT_NAMES
 gram_names=['intercept','clip_student_t0','clip_student_t1','abs_clip_student_t0','abs_clip_student_t1','square_clip_student_t0','square_clip_student_t1','product_clip_student_t0_t1']+[f'normalized_{n}' for n in FEATURE_COLUMNS+EVENT_NAMES]+FEATURE_NAMES+[f'deviation_{name}_t{j}' for name in ('shallow_seed3','plain_seed3','plain_seed4','chunk500','shallow_seed4') for j in (0,1)]+['clip_y0','clip_y1'];assert len(gram_names)==261
 assert len(data)==1873
 for si,(fold,i) in enumerate(selected):
  group=int(rg[i]);chain=info[group][1];cl=int(cls[i]);seq=data.read(group,targets=False,mask=False)
  assert seq.seq_ix==int(seq_ix[i]) and seq.targets is None and seq.scored is None and np.isfinite(seq.features).all()
  raw_digest.update(seq.features.tobytes());xx,flags,delta,events99=causal_summary(seq.features)
  parts=[arrays[name,fold][i] for name in MEMBERS];assert all(np.isfinite(p).all() for p in parts)
  pp=prediction_stack(parts);mm=mask[i].copy();mm[:99]=False;steps=np.flatnonzero(mm);k=len(steps);sl=slice(cursor,cursor+k)
  X[sl]=xx[steps];Y[sl]=y[i,steps];Pred[sl]=pp[steps];meta[sl]=np.column_stack([np.full(k,i),np.full(k,group),steps,np.full(k,chain),np.full(k,fold),np.full(k,cl)])
  assert np.isfinite(Y[sl]).all();counts[si]=k;smeta[si]=[i,group,-1,chain,fold,cl]
  mean,std=normalizers[fold];current211=(np.concatenate([seq.features,events99],axis=1)-mean)/std
  bp=np.clip(pp[steps,0,:],-2,2).astype(np.float64);base7=np.column_stack([bp,np.abs(bp),bp**2,bp[:,0]*bp[:,1]])
  deviations=(pp[steps,3:8,:]-pp[steps,0:1,:]).reshape(k,10)
  clipped_y=np.clip(y[i,steps].astype(np.float64),-2,2)
  Z=np.concatenate([np.ones((k,1)),base7,current211[steps].astype(np.float64),xx[steps].astype(np.float64),deviations.astype(np.float64),clipped_y],axis=1);assert Z.shape==(k,261) and np.isfinite(Z).all()
  ci=chain_index[fold,chain];chain_counts[ci]+=k;chain_class_counts[ci,class_index[cl]]+=k
  for target in (0,1):gram[ci,target]+=Z.T @ (Z*np.abs(clipped_y[:,target,None]))
  for j in range(8):stats[si,j]=fe.sums(y[i:i+1],pp[None,:,j,:],mm[None,:])[0]
  anchors,offsets,es=event_indices(flags);ek=len(es);event_slots+=min(64,int(np.count_nonzero(flags[:,0])))*8;esl=slice(ecursor,ecursor+ek)
  emeta[esl]=np.column_stack([np.full(ek,i),np.full(ek,group),es,anchors,offsets,np.full(ek,chain),np.full(ek,fold),np.full(ek,cl)])
  ey[esl]=y[i,es];ep[esl]=pp[es,0];ed[esl]=delta[anchors];ef[esl]=flags[es];em[esl]=mask[i,es];eam[esl]=mask[i,anchors];epdelta[esl]=pp[anchors,0]-pp[anchors-1,0];ecsign[esl]=np.sign(ed[esl]).astype(np.int8);epsign[esl]=np.sign(epdelta[esl]).astype(np.int8)
  cursor+=k;ecursor+=ek
  if si%50==0:print(json.dumps({'sequences_complete':si+1,'scored_rows':cursor,'event_rows':ecursor,'seconds':time.monotonic()-start}),flush=True)
 assert cursor==n and ecursor<=MAX_EVENTS
 payload=sum(v.nbytes for v in (X,Y,Pred,meta,emeta[:ecursor],ey[:ecursor],ep[:ecursor],ed[:ecursor],ef[:ecursor],em[:ecursor],eam[:ecursor],epdelta[:ecursor],ecsign[:ecursor],epsign[:ecursor],stats,smeta,counts,gram,chain_counts,chain_class_counts));assert payload<1000*2**20
 for path,digest in tracked.items():assert sha(path)==digest,('changed input',path)
 assert sha(a.plan)==a.plan_sha256
 save_npz(out/'scored-rows.npz',X=X,y=Y,predictions=Pred,metadata=meta,feature_names=np.asarray(FEATURE_NAMES),prediction_names=np.asarray(PRED_NAMES),metadata_names=np.asarray(META_NAMES))
 save_npz(out/'event-windows.npz',metadata=emeta[:ecursor],y=ey[:ecursor],student_prediction=ep[:ecursor],anchor_price_code_delta=ed[:ecursor],row_event_flags=ef[:ecursor],is_scored=em[:ecursor],anchor_is_scored=eam[:ecursor],anchor_model_delta=epdelta[:ecursor],anchor_price_code_sign=ecsign[:ecursor],anchor_model_sign=epsign[:ecursor],metadata_names=np.asarray(EVENT_META_NAMES),offsets=OFFSETS)
 save_npz(out/'sequence-sums.npz',sums=stats,metadata=smeta,scored_counts=counts,prediction_names=np.asarray(PRED_NAMES),metadata_names=np.asarray(META_NAMES))
 assert np.isfinite(gram).all() and chain_counts.sum()==n and chain_class_counts.sum()==n
 save_npz(out/'chain-gram.npz',gram=gram,chain_metadata=np.asarray(chain_keys,dtype=np.int32),chain_metadata_names=np.array(['fold','chain']),chain_scored_counts=chain_counts,chain_class_scored_counts=chain_class_counts,class_values=class_values,feature_names=np.asarray(gram_names),block_bounds_json=np.array(json.dumps(schema()['chain_Gram']['blocks'])),normalization_mean=np.stack([normalizers[f][0] for f in (0,1)]),normalization_std=np.stack([normalizers[f][1] for f in (0,1)]))
 saved={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.glob('*.npz'))};assert sum(v['bytes'] for v in saved.values())<1024*2**20
 report={'status':'COMPLETE_ONE_PASS_ERROR_ANALYSIS_EXTRACTION_NO_FITS_NO_WP_READOUT','utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'plan_sha256':a.plan_sha256,'rows':n,'sequences':860,'chains':250,'event_rows':ecursor,'event_slots_before_edge_filter':event_slots,'event_dropped_out_of_bounds':event_slots-ecursor,'uncompressed_payload_bytes':payload,'outputs':saved,'seconds_after_input_verification':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'selected_raw_features_sha256':raw_digest.hexdigest(),'inputs_unchanged_before_after':True,'schema':schema(),'fits':0,'inference_runs':0,'pooled_metric_calls':0,'causality':'X uses rows through t with per-sequence resets; class/chain/labels and event-anchor sample are retrospective diagnostic metadata; masks applied after feature histories.'}
 (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','rows','event_rows','seconds_after_input_verification','peak_RSS_KiB')}),flush=True)

if __name__=='__main__':main()
