from __future__ import annotations
import hashlib,importlib.util,json,sys,time,csv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
Q=Path('/data/epanov_s/tabdata/runtime/quality-horizon-20261031-r1');H=Path('/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1');out=Q/'reports/fixed-tl3-mixtures-r1';out.mkdir(exist_ok=False)
sys.path.insert(0,str(H/'source6-eval/src'));sys.path.append(str(H/'cpu-deps-r1'))
from wnn.gmetric import global_wp
helper=H/'requests/verify-primary-independent-r1.py';assert hashlib.sha256(helper.read_bytes()).hexdigest()=='5df00bb5e209ba723b32d04ba8c8748380b3207d07c4108f35d0543f572a9963';spec=importlib.util.spec_from_file_location('independent_primary',helper);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def sha(p:Path)->str:
 return v.sha256(p)
def write(p:Path,obj:object)->None:
 with p.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
def evaluate(name:str,prediction:np.ndarray)->dict[str,object]:
 path=out/(name+'.npy');np.save(path,prediction,allow_pickle=False);metrics=global_wp(target,prediction,np.ones(len(target),dtype=bool));metrics['wmse']=float(np.sum(np.abs(np.clip(target.astype(np.float64),-2,2))*np.square(prediction.astype(np.float64)-np.clip(target.astype(np.float64),-2,2)))/np.sum(np.abs(np.clip(target.astype(np.float64),-2,2))));independent=v.independent_metrics(target,prediction);error=max(abs(metrics[k]-independent[k]) for k in ['wp','t0','t1','wmse']);assert error<1e-10;row={'name':name,'metrics':metrics,'prediction':{'path':str(path),'sha256':sha(path),'shape':list(prediction.shape),'dtype':'float32'},'independent_max_absolute_error':error};records.append(row);return row
split=json.loads((H/'data/chronological_primary.json').read_text());target,mask,raw_identity=v.read_primary_labels(H/'data',split);groups=split['evaluation']['valid'];bank=Q/'legacy-replay/data/cache';runs=['tl3c_w18final_distilled_s5','tl3c_fttall_tlead_s5','tl3c_fttall_tlead_s6'];arrays=[];positions=[];inputs=[]
for name in runs:
 p=bank/('pred_'+name+'_valid.npy');m=json.loads(p.with_suffix('.json').read_text());assert sha(p)==m['prediction_sha256'];assert m['split_manifest_sha256']==v.PRIMARY and m['inference_precision']=='fp32' and set(m['row_groups'])==set(groups);a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.shape==(1873,20000,2) and a.dtype==np.float32;arrays.append(a);positions.append({n:i for i,n in enumerate(m['row_groups'])});inputs.append({'run':name,'prediction_sha256':m['prediction_sha256'],'checkpoint_sha256':m['checkpoint_sha256'],'metadata_sha256':sha(p.with_suffix('.json'))})
selected=[np.empty((v.ROWS,2),dtype=np.float32) for _ in runs];source=pq.ParquetFile(H/'data/datasets/valid.parquet');offset=0
for group in groups:
 rowmask=source.read_row_group(group,columns=['is_scored'],use_threads=False)['is_scored'].to_numpy().astype(bool,copy=True);rowmask[:99]=False;n=int(rowmask.sum())
 for j,a in enumerate(arrays):selected[j][offset:offset+n]=a[positions[j][group]][rowmask]
 offset+=n
assert offset==v.ROWS;records=[];baseline_prediction=v.reconstruct_block(selected,np.asarray([.34,.33,.33],dtype=np.float64));baseline=evaluate('replayed_TL3',baseline_prediction);assert abs(baseline['metrics']['wp']-.6923774245652725)<1e-6,baseline['metrics'];r=json.loads((H/'reports/full-primary-refits-r1/full_primary_refit_report.json').read_text());candidates=[('gru',0),('lstm',2),('transformer',3)]
for family,index in candidates:
 member=r['singleton_scores'][index]['score'];receipt=member['predictions'];assert all(receipt[k]==mask[k] for k in v.MASK_KEYS);candidate=v._array(Path(receipt['path']),receipt['sha256'])
 for alpha in [.05,.10]:
  mixed=v.reconstruct_block([baseline_prediction,candidate],np.asarray([1-alpha,alpha],dtype=np.float64));row=evaluate(f'TL3_plus_{family}_{alpha:.2f}',mixed);row.update(candidate_fit_id=member['fit_id'],candidate_model_sha256=member['checkpoint']['sha256'],candidate_prediction_sha256=receipt['sha256'],candidate_weight=alpha);print(row['name'],row['metrics']['wp'],flush=True)
for row in records:row['delta_vs_replayed_TL3']=row['metrics']['wp']-baseline['metrics']['wp']
summary={'schema':'wnn.quality-fixed-tl3-mixtures.v1','status':'complete','new_fits':0,'created_epoch':time.time(),'primary_split_sha256':v.PRIMARY,'scored_rows':v.ROWS,'evaluation_sequences':1873,'mask_and_order':mask,'raw_validation_source':raw_identity,'legacy_member_inputs':inputs,'records':records,'historical_TL3_reference':.6923774245652725,'historical_TL3_plus_GALAM_reference':.6924964468441978,'historical_GALAM_reference_rescored':False,'weights_predeclared':True,'primary_valid_used_for_weight_learning':False,'exposure':'already development-exposed official valid; no significance or public-score claim','arithmetic':'Source6 official global_wp; separately checked NumPy covariance; FP64 raw mixture then one FP32 cast before official clipping'};write(out/'REPORT.json',summary)
with (out/'COMPARISON.csv').open('x',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['name','wp','t0','t1','wmse','delta_vs_replayed_TL3']);writer.writeheader();writer.writerows({'name':row['name'],**{k:row['metrics'][k] for k in ['wp','t0','t1','wmse']},'delta_vs_replayed_TL3':row['delta_vs_replayed_TL3']} for row in records)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,5));rows=records[1:];ax.barh([row['name'] for row in rows],[row['delta_vs_replayed_TL3'] for row in rows]);ax.axvline(0,color='black',linewidth=1);ax.set_xlabel('Full official-valid WP delta vs freshly replayed TL3');ax.set_title('Six predeclared fixed mixtures; development-exposed reporting');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(out/('mixture_deltas.'+ext),dpi=160,bbox_inches='tight')
plt.close(fig);print('FIXED_SIX_MIXTURES_COMPLETE',baseline['metrics']['wp'],flush=True)
