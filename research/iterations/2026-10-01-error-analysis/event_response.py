"""Fixed retrospective observed-event response analysis; no inference or forecast search."""
from __future__ import annotations
from pathlib import Path
import argparse,csv,hashlib,json,os,resource,signal,time
import numpy as np
HERE=Path(__file__).resolve().parent
OFFSETS=np.array([-32,-8,-1,0,1,4,16,64],np.int32)
MAIN_META=['cache_position','row_group','step','chain','fold','market_class_retrospective']
EVENT_META=['cache_position','row_group','step','anchor_step','offset','chain','fold','market_class_retrospective']

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def normalizers(y,p):
 y=np.clip(np.asarray(y,np.float64),-2,2);p=np.clip(np.asarray(p,np.float64),-2,2);w=abs(y)
 W=w.sum(0);assert np.all(W>1e-8)
 my=(w*y).sum(0)/W;mp=(w*p).sum(0)/W
 sy=np.sqrt((w*(y-my)**2).sum(0)/W);sp=np.sqrt((w*(p-mp)**2).sum(0)/W)
 assert np.all(sy>1e-8) and np.all(sp>1e-8),'Degenerate main population; no standardized event curve'
 rho=(w*(y-my)*(p-mp)).sum(0)/(W*sy*sp)
 assert np.all(abs(rho)<=1+1e-12)
 return {k:v.tolist() for k,v in dict(W=W,mean_y=my,mean_p=mp,sd_y=sy,sd_p=sp,rho=rho).items()}

def row_coordinates(y,p,par):
 y=np.clip(np.asarray(y,np.float64),-2,2);raw=np.asarray(p,np.float64);p=np.clip(raw,-2,2)
 u=(y-par['mean_y'])/par['sd_y'];z=(p-par['mean_p'])/par['sd_p'];q=u-np.asarray(par['rho'])*z
 active=(raw>-2)&(raw<2)
 return {'y':y,'p':p,'w':abs(y),'residual':y-p,'e':u-z,'q':q,'q_active':q*active,'active':active}

def validate_events(events):
 m=events['metadata'];assert m.ndim==2 and m.shape[1]==8
 assert events['metadata_names'].tolist()==EVENT_META and np.array_equal(events['offsets'],OFFSETS)
 n=len(m);assert n>0 and np.all(m[:,2]==m[:,3]+m[:,4]) and np.all((m[:,2]>=0)&(m[:,2]<20000))
 assert np.isin(m[:,4],OFFSETS).all() and np.isin(m[:,6],[0,1]).all() and np.isin(m[:,7],[0,1,2,3]).all()
 keys=m[:,[0,3,4]];assert len(np.unique(keys,axis=0))==n,'Duplicated event offset'
 anchor_keys=m[:,[0,3]];unique,group=np.unique(anchor_keys,axis=0,return_inverse=True)
 bases=np.flatnonzero(m[:,4]==-1);assert len(bases)==len(unique)
 lookup=np.full(len(unique),-1,np.int64);lookup[group[bases]]=bases;assert np.all(lookup>=0)
 base=lookup[group]
 zero=np.flatnonzero(m[:,4]==0);assert len(zero)==len(unique)
 zlookup=np.full(len(unique),-1,np.int64);zlookup[group[zero]]=zero;assert np.all(zlookup>=0)
 atzero=zlookup[group]
 for name in ['y','student_prediction','anchor_price_code_delta','anchor_model_delta']:
  assert events[name].shape==(n,2) and np.isfinite(events[name]).all(),name
 for name in ['anchor_price_code_delta','anchor_model_delta','anchor_is_scored']:
  assert np.array_equal(events[name],events[name][atzero]),name
 assert np.array_equal(events['anchor_is_scored'],events['is_scored'][atzero])
 assert np.all(events['row_event_flags'][zero,0]==1)
 # Delta is the stored raw current forecast minus the same anchor's previous-row forecast.
 expected=events['student_prediction'][atzero]-events['student_prediction'][base]
 np.testing.assert_array_equal(events['anchor_model_delta'],expected)
 for name,source in [('anchor_model_sign','anchor_model_delta'),('anchor_price_code_sign','anchor_price_code_delta')]:
  if name in events:np.testing.assert_array_equal(events[name],np.sign(events[source]))
 return base,atzero,len(unique)

def quiet_secondary(main_meta,cumulative,events):
 """Use already-scored endpoint count keys; no labels or predictions enter selection."""
 count=np.asarray(cumulative)
 assert count.ndim==1 and len(count)==len(main_meta) and np.isfinite(count).all()
 assert np.all((count>=0)&(count<=20000)) and np.array_equal(count,np.rint(count))
 keys=main_meta[:,0].astype(np.int64)*20000+main_meta[:,2]
 order=np.argsort(keys);keys=keys[order];count=count[order].astype(np.int64)
 assert np.all(np.diff(keys)>0)
 m=events['metadata'];eligible=(m[:,4]>=0)&(m[:,2]>=99)&(m[:,3]>=99)&events['is_scored']&events['anchor_is_scored']
 result=np.zeros(len(m),bool);rows=np.flatnonzero(eligible)
 anchor=m[rows,0].astype(np.int64)*20000+m[rows,3];current=m[rows,0].astype(np.int64)*20000+m[rows,2]
 ia=np.searchsorted(keys,anchor);ic=np.searchsorted(keys,current)
 assert np.all(ia<len(keys)) and np.all(ic<len(keys))
 assert np.array_equal(keys[ia],anchor) and np.array_equal(keys[ic],current),'Scored anchor/offset endpoint missing from main table'
 difference=count[ic]-count[ia];assert np.all(difference>=0)
 assert np.all(difference[m[rows,4]==0]==0)
 result[rows]=difference==0
 return result,{'eligible_positive_endpoint_pairs':len(rows),'matched_pairs':len(rows),'quiet_pairs':int(result.sum()),
  'rule':'Exact cumulative_i0px_events(offset)-cumulative_i0px_events(anchor)==0; endpoints already scored',
  'causal_limit':'Conditioning on no subsequent event selects future quietness relative to anchor. This is descriptive, not deployable selection.'}

def aggregate_curves(events,parameters,quiet=None):
 m=events['metadata'];base,zero,nanchors=validate_events(events)
 eligible=m[:,2]>=99
 scored=events['is_scored'].astype(bool)&eligible
 anchor_scored=events['anchor_is_scored'].astype(bool)&(m[:,3]>=99)
 oi=np.searchsorted(OFFSETS,m[:,4]);fold=m[:,6];cls=m[:,7]
 result=[];coverage=[]
 for scope in ['pooled','fold0','fold1']:
  pop=np.ones(len(m),bool) if scope=='pooled' else fold==int(scope[-1])
  par=parameters[scope];c=row_coordinates(events['y'],events['student_prediction'],par)
  dy=c['y']-c['y'][base];dp=c['p']-c['p'][base]
  for cl in [-1,0,1,2,3]:
   domain=pop & (np.ones(len(m),bool) if cl==-1 else cls==cl)
   for offset in OFFSETS:
    mask=domain & (m[:,4]==offset)&eligible
    a=anchor_scored[mask];s=scored[mask]
    coverage.append({'scope':scope,'class':cl,'offset':int(offset),'available_postwarmup_rows':int(mask.sum()),
     'offset_scored_rows':int(s.sum()),'anchor_scored_rows':int(a.sum()),'both_scored_rows':int((a&s).sum()),
     'anchor_scored_offset_unscored_rows':int((a&~s).sum()),'anchor_unscored_offset_scored_rows':int((~a&s).sum()),
     'neither_scored_rows':int((~a&~s).sum()),'offset_i0_event_rows':int(events['row_event_flags'][mask,0].sum())})
   for t in range(2):
    directions={'model_delta':np.sign(events['anchor_model_delta'][:,t]),
                'raw_code_p0':np.sign(events['anchor_price_code_delta'][:,0]),
                'raw_code_p11':np.sign(events['anchor_price_code_delta'][:,1])}
    for name,sign in directions.items():
     for gate in ['offset_scored','anchor_and_offset_scored']+(['quiet_anchor_and_offset_scored'] if quiet is not None else []):
      mask=domain&scored
      if gate=='anchor_and_offset_scored':mask &= anchor_scored
      if gate=='quiet_anchor_and_offset_scored':mask &= anchor_scored&quiet
      zero_counts=np.bincount(oi[mask&(sign==0)],minlength=8)
      use=mask&(sign!=0);ix=oi[use];s=sign[use].astype(np.float64);w=c['w'][use,t]
      W=np.bincount(ix,weights=w,minlength=8);count=np.bincount(ix,minlength=8)
      pos=np.bincount(ix,weights=(s>0),minlength=8);neg=np.bincount(ix,weights=(s<0),minlength=8)
      values={'oriented_target_increment':dy[use,t]*s,'oriented_prediction_increment':dp[use,t]*s,
              'oriented_clipped_residual':c['residual'][use,t]*s,'oriented_standardized_error':c['e'][use,t]*s,
              'oriented_q_postclip':c['q'][use,t]*s,'oriented_q_clip_active':c['q_active'][use,t]*s,
              'clipping_active_fraction':c['active'][use,t].astype(float)}
      means={k:np.divide(np.bincount(ix,weights=w*v,minlength=8),W,out=np.full(8,np.nan),where=W>0) for k,v in values.items()}
      for j,offset in enumerate(OFFSETS):
       row={'scope':scope,'class':cl,'target':t,'orientation':name,'mask_gate':gate,'offset':int(offset),
            'sample_rows':int(count[j]),'positive_orientation_rows':int(pos[j]),'negative_orientation_rows':int(neg[j]),
            'zero_orientation_excluded_rows':int(zero_counts[j]),'target_weight':float(W[j])}
       row.update({k:float(v[j]) if np.isfinite(v[j]) else None for k,v in means.items()})
       result.append(row)
 return result,coverage,nanchors

def plot_results(rows,coverage,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 xx=np.arange(8)
 def curve(scope,t,orientation,gate,key):
  r=[v for v in rows if v['scope']==scope and v['class']==-1 and v['target']==t and v['orientation']==orientation and v['mask_gate']==gate]
  assert len(r)==8
  return np.array([np.nan if v[key] is None else v[key] for v in r])
 fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
 for t in range(2):
  for f in range(2):
   ax=axes[t,f];scope=f'fold{f}'
   for key,label,color in [('oriented_target_increment','Clipped target','#C2533F'),('oriented_prediction_increment','Clipped prediction','#226C9A')]:
    ax.plot(xx,curve(scope,t,'model_delta','anchor_and_offset_scored',key),'-o',label=label,color=color,ms=3)
    ax.plot(xx,curve(scope,t,'model_delta','quiet_anchor_and_offset_scored',key),':s',label=label+'; no intervening event',color=color,ms=3)
   ax.axhline(0,color='.6',lw=.7);ax.axvline(3,color='.5',ls=':',lw=1);ax.set_title(f'Fold {f}, target {t}')
   ax.set_ylabel('Weighted increment from anchor −1');ax.set_xticks(xx,OFFSETS);ax.set_xlabel('Row offset (discrete selected positions)')
 axes[0,0].legend(fontsize=8);fig.suptitle('Observed-event response, oriented by the anchor prediction change\nAnchor ∩ offset scored; dotted positive-offset curves select future quietness',fontsize=12)
 fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'01-event-response.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
 styles=[('model_delta','Model Δ direction','#226C9A'),('raw_code_p0','Raw code p0 direction','#C2533F'),('raw_code_p11','Raw code p11 direction','#6A8F37')]
 for t in range(2):
  for f in range(2):
   ax=axes[t,f]
   for orient,label,color in styles:
    ax.plot(xx,curve(f'fold{f}',t,orient,'anchor_and_offset_scored','oriented_q_postclip'),'-o',ms=3,label=label,color=color)
    ax.plot(xx,curve(f'fold{f}',t,orient,'offset_scored','oriented_q_postclip'),'--',alpha=.45,color=color)
    if orient=='model_delta':ax.plot(xx,curve(f'fold{f}',t,orient,'quiet_anchor_and_offset_scored','oriented_q_postclip'),':s',ms=3,color=color,label='Model Δ; no intervening event')
   ax.axhline(0,color='.6',lw=.7);ax.axvline(3,color='.5',ls=':',lw=1);ax.set_title(f'Fold {f}, target {t}')
   ax.set_ylabel('Weighted oriented q = u − ρz');ax.set_xticks(xx,OFFSETS);ax.set_xlabel('Row offset')
 axes[0,0].legend(fontsize=8);fig.suptitle('Correlation sensitivity in the clipped-prediction coordinate\nSolid: anchor ∩ offset scored; dashed: offset scored. Raw-code sign is not price direction.',fontsize=11)
 fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'02-event-gradient.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for f,ax in enumerate(axes):
  rr=[r for r in coverage if r['scope']==f'fold{f}' and r['class']==-1]
  ratio=lambda a,b:np.array([r[a]/r[b] if r[b] else np.nan for r in rr])
  ax.plot(xx,ratio('offset_scored_rows','available_postwarmup_rows'),'-o',label='Offset scored / all available')
  ax.plot(xx,ratio('both_scored_rows','anchor_scored_rows'),'-o',label='Offset scored / anchor-scored subset')
  ax.set_title(f'Fold {f}');ax.set_xticks(xx,OFFSETS);ax.set_xlabel('Row offset');ax.set_ylim(0,1);ax.axvline(3,color='.5',ls=':')
  ax.set_ylabel('Fraction of event-window rows');ax.legend(fontsize=8)
 fig.suptitle('Mask overlap changes the population along each event curve\nAt most 64 input-selected anchors per sequence; not all-event prevalence',fontsize=11)
 fig.tight_layout(rect=(0,0,1,.86));fig.savefig(out/'03-event-mask-coverage.png',dpi=150);plt.close(fig)

def selftest():
 # Independent scalar weighted-Pearson central differences, including a saturated raw prediction.
 rng=np.random.default_rng(28);y=rng.normal(size=(11,2));p=rng.normal(scale=.4,size=(11,2));p[0,0]=3
 par=normalizers(y,p);c=row_coordinates(y,p,par)
 analytic=.5*c['w']*c['q_active']/np.asarray(par['W'])/np.asarray(par['sd_p'])
 def literal_score(pred):
  values=[]
  for t in range(2):
   yy=[max(-2,min(2,float(v))) for v in y[:,t]];pp=[max(-2,min(2,float(v))) for v in pred[:,t]]
   w=[abs(v) for v in yy];W=sum(w);a=sum(q*x for q,x in zip(w,yy))/W;b=sum(q*x for q,x in zip(w,pp))/W
   cov=sum(q*(x-a)*(z-b) for q,x,z in zip(w,yy,pp));vy=sum(q*(x-a)**2 for q,x in zip(w,yy));vp=sum(q*(x-b)**2 for q,x in zip(w,pp))
   values.append(cov/(vy*vp)**.5)
  return sum(values)/2
 err=0
 for i in range(len(y)):
  for t in range(2):
   plus=p.copy();minus=p.copy();plus[i,t]+=1e-6;minus[i,t]-=1e-6
   err=max(err,abs((literal_score(plus)-literal_score(minus))/2e-6-analytic[i,t]))
 assert err<1e-8 and analytic[0,0]==0
 # Event gathers use exact t=anchor+offset; later rows cannot change an earlier gathered row.
 anchors=np.array([100,180]);aa=np.repeat(anchors,8);off=np.tile(OFFSETS,2);step=aa+off;n=len(step)
 series=np.arange(300,dtype=np.float32);pred=np.stack([series*.002,-series*.003],axis=1);yy=pred*.8
 meta=np.column_stack([np.zeros(n,int),np.zeros(n,int),step,aa,off,np.ones(n,int),np.zeros(n,int),np.zeros(n,int)]).astype(np.int32)
 ev={'metadata':meta,'metadata_names':np.array(EVENT_META),'offsets':OFFSETS,'y':yy[step],
     'student_prediction':pred[step],'anchor_price_code_delta':np.tile([1.,-1.],(n,1)).astype(np.float32),
     'anchor_model_delta':pred[aa]-pred[aa-1],'anchor_is_scored':aa%2==0,'is_scored':step%2==0,
     'row_event_flags':np.column_stack([np.isin(step,anchors),np.zeros((n,7),bool)]).astype(np.uint8)}
 base,zero,count=validate_events(ev);assert count==2 and np.all(meta[base,4]==-1) and np.all(meta[zero,4]==0)
 assert np.array_equal(yy[step]-yy[meta[base,2]],ev['y']-ev['y'][base])
 changed=pred.copy();changed[201:]=999;assert np.array_equal(changed[step[step<=200]],pred[step[step<=200]])
 wrong=dict(ev);wrong['metadata']=meta.copy();wrong['metadata'][0,2]+=1
 try:validate_events(wrong)
 except AssertionError:pass
 else:raise AssertionError('Alignment corruption not caught')
 wrong=dict(ev);wrong['anchor_model_delta']=ev['anchor_model_delta'].copy();wrong['anchor_model_delta'][0,0]+=1
 try:validate_events(wrong)
 except AssertionError:pass
 else:raise AssertionError('Future/wrong anchor delta not caught')
 pars={k:normalizers(y,p) for k in ['pooled','fold0','fold1']}
 curves,cov,_=aggregate_curves(ev,pars)
 zero_curve=[r for r in curves if r['offset']==-1 and r['target_weight']>0]
 assert all(r['oriented_target_increment']==r['oriented_prediction_increment']==0 for r in zero_curve)
 atzero=[r for r in cov if r['offset']==0]
 assert all(r['both_scored_rows']==r['offset_scored_rows']==r['anchor_scored_rows'] for r in atzero)
 # Exact integer prefix-count join and future-quiet subset, independent of outcomes.
 allsteps=np.arange(99,260);mm=np.column_stack([np.zeros(len(allsteps),int),np.zeros(len(allsteps),int),allsteps,np.ones(len(allsteps),int),np.zeros(len(allsteps),int),np.zeros(len(allsteps),int)])
 cc=(allsteps>=100).astype(int)+(allsteps>=180)+(allsteps>=190)
 quiet,join=quiet_secondary(mm,cc,ev)
 assert quiet[(aa==100)&(off==64)][0] and not quiet[(aa==180)&(off==64)][0]
 assert not quiet[off<0].any() and join['eligible_positive_endpoint_pairs']==join['matched_pairs']
 changed_events=dict(ev);changed_events['y']=ev['y']+123;changed_events['student_prediction']=ev['student_prediction']-99
 assert np.array_equal(quiet_secondary(mm,cc,changed_events)[0],quiet)
 return {'status':'PASS','generated_only':True,'quiet_integer_join_label_independence':True,'central_difference_max_error':err,'saturated_raw_gradient_zero':True,
         'offset_alignment_and_corruption_detection':True,'exact_pre_event_origin':True,'anchor_delta_local_only':True,
         'later_rows_do_not_change_earlier_gathered_values':True,
         'scope_limit':'Global normalizers and anchor subsampling are retrospective; this is not a causal serving routine.',
         'actual_data_reads':0,'model_calls':0,'source_sha256':sha(__file__)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--input-dir',type=Path);ap.add_argument('--output-dir',type=Path);ap.add_argument('--plan',type=Path);ap.add_argument('--plan-sha256');ap.add_argument('--source-sha256');a=ap.parse_args()
 if a.self_test:
  result=selftest();(HERE/'event-response-selftest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Event analysis300s cap')));signal.alarm(298)
 assert a.input_dir and a.output_dir and a.plan and sha(a.plan)==a.plan_sha256
 assert a.source_sha256==sha(__file__)
 plan=json.loads(a.plan.read_text());assert plan['population']['scored_rows_after_warmup']==1614484
 assert not os.environ.get('CUDA_VISIBLE_DEVICES','')
 out=a.output_dir.resolve();assert out.is_relative_to(HERE) and out!=HERE and not out.exists()
 inp=a.input_dir.resolve();receipt_path=inp/'receipt.json';receipt=json.loads(receipt_path.read_text())
 assert receipt['status']=='COMPLETE_ONE_PASS_ERROR_ANALYSIS_EXTRACTION_NO_FITS_NO_WP_READOUT'
 assert receipt['rows']==1614484 and receipt['sequences']==860 and receipt['chains']==250
 paths=[inp/'scored-rows.npz',inp/'event-windows.npz',receipt_path,a.plan,Path(__file__)]
 hashes={str(p):sha(p) for p in paths}
 for p in paths[:2]:assert receipt['outputs'][p.name]['sha256']==hashes[str(p)]
 with np.load(paths[0],allow_pickle=False) as data:
  assert data['metadata_names'].tolist()==MAIN_META and data['prediction_names'][0]=='current_student4'
  y=data['y'];p=data['predictions'][:,0,:];meta=data['metadata']
  assert data['feature_names'][29]=='cumulative_i0px_events';cumulative=data['X'][:,29].copy()
 assert y.shape==p.shape==(1614484,2) and meta.shape==(1614484,6) and np.all(meta[:,2]>=99)
 assert np.isfinite(y).all() and np.isfinite(p).all()
 params={scope:normalizers(y[use],p[use]) for scope,use in [('pooled',np.ones(len(y),bool)),('fold0',meta[:,4]==0),('fold1',meta[:,4]==1)]}
 assert abs(np.mean(params['pooled']['rho'])-plan['metric']['expected_WP']['current_student4'])<1e-10

 main_info=np.unique(meta[:,[0,1,3,4,5]],axis=0)
 del y,p
 with np.load(paths[1],allow_pickle=False) as f:events={k:f[k] for k in f.files}
 event_info=np.unique(events['metadata'][:,[0,1,5,6,7]],axis=0)
 main_keys=set(map(tuple,main_info));assert all(tuple(v) in main_keys for v in event_info)
 quiet,quiet_receipt=quiet_secondary(meta,cumulative,events)
 del meta,cumulative
 rows,coverage,nanchors=aggregate_curves(events,params,quiet)
 out.mkdir();plot_results(rows,coverage,out)
 for name,table in [('event-curves.csv',rows),('mask-coverage.csv',coverage)]:
  with (out/name).open('x',newline='') as stream:
   writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
 for path,digest in hashes.items():assert sha(path)==digest,('changed input',path)
 report={'status':'COMPLETE','source_sha256':sha(__file__),'plan_sha256':a.plan_sha256,'input_sha256':hashes,
  'normalizers':params,'quiet_secondary':quiet_receipt,'event_rows':len(events['metadata']),'anchors':nanchors,'curves':rows,'mask_coverage':coverage,
  'definitions':{'q':'u-rho*z, post-clipped prediction coordinate; global main-row normalization within named scope',
   'raw_prediction_derivative':'0.5*|clip(y)|/(W*sd_p)*q*1(-2<raw_p<2); the clipping-kink derivative is not defined and is reported inactive',
   'orientation':'sign of raw anchor model delta per target OR raw code p0/p11 delta; never sign(target), never physical-price direction',
   'weights':'Each event-window mean uses |clip(y)| for that displayed offset and target. No claim of all-event population representativeness.',
   'mask_gates':'offset_scored=(is_scored & step>=99); anchor_and_offset additionally requires anchor_is_scored & anchor_step>=99',
   'scope':'Retrospective input-selected anchor sample. Pre-event orientations use the later anchor for alignment only. Overlapping windows can repeat a row and can cross further i0 events. The secondary no-intervening-event curve joins exact cumulative counts at scored endpoints; it conditions on future quietness and still cannot identify causal latency. No curve adds to an official metric budget.',
   'causal_limit':'Event response cannot identify whether missing target increments were predictable from already available information. No shifted forecast or error correction is scored.'},
  'seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'forecast_fits':0,'model_inference_runs':0,'new_candidate_scores':0}
 assert report['seconds']<300
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 signal.alarm(0);print(json.dumps({k:report[k] for k in ['status','anchors','event_rows','seconds','peak_RSS_KiB']}))

if __name__=='__main__':main()
