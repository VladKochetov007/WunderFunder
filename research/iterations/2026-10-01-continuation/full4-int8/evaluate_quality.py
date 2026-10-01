"""One fixed quant4 versus same-member FP16 comparison; frozen fold_eval math."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 output=P/'quality.json';assert not output.exists()
 alloc=json.loads((P/'allocation.json').read_text());assert alloc['Q4_quality_evaluation_authorized'] is True
 metric=ROOT/'scripts/fold_eval.py';assert sha(metric)=='d05791f5f8c6ea64d9863d4f4b45430eb46b9d0b29ccbdac1a4ff5b36a2573c0'
 spec=importlib.util.spec_from_file_location('frozen_q4_metric',metric);fe=importlib.util.module_from_spec(spec);spec.loader.exec_module(fe)
 frozen=json.loads((ROOT/'research/iterations/2026-10-01-overnight/grader/frozen-data.json').read_text())
 for name,v in frozen.items():assert sha(ROOT/name)==v['sha256']
 tg=np.load(ROOT/'cache/valid_targets.npz',allow_pickle=False);groups=tg['row_group'];y=tg['y'];mask=tg['mask'];cf=np.load(ROOT/'cache/chain_folds.npz',allow_pickle=False);valid=cf['split']=='valid'
 info={int(g):(int(f),int(c),bool(p))for g,f,c,p in zip(cf['row_group'][valid],cf['fold'][valid],cf['chain'][valid],cf['pure'][valid],strict=True)}
 train=set(cf['chain'][cf['split']=='train'].tolist())
 for chain in np.unique(cf['chain'][valid]):
  sel=valid&(cf['chain']==chain);assert len(np.unique(cf['fold'][sel]))==1
  if cf['pure'][sel].any():assert int(chain)not in train
 orderhash=hashlib.sha256(groups.tobytes()).hexdigest();stats={mode:[]for mode in ['quant','fp16']};foldids=[];positions=[];provenance={}
 for fold in [0,1]:
  arrays={};selected=[i for i,g in enumerate(groups)if info[int(g)][0]==fold]
  for mode in stats:
   f=P/'cache'/f'pred_on_q4{mode}_f{fold}_valid.npy';m=json.loads(f.with_suffix('.json').read_text());digest=sha(f)
   assert m['prediction_sha256']==digest and m['fold']==fold and m['epoch']==1 and m['rows']==selected and m['row_groups']==groups[selected].tolist()
   assert m['source_row_group_order_sha256']==orderhash and m['chain_folds_sha256']==sha(ROOT/'cache/chain_folds.npz')
   assert m['coefficients']==[.25]*4 and len(m['child_checkpoints'])==4
   manifest=json.loads(Path(m['checkpoint']).read_text());assert m['checkpoint_sha256']==sha(m['checkpoint']) and manifest['allocation_sha256']==sha(P/'allocation.json')
   assert [c['run']for c in m['child_checkpoints']]==[f'{n}_f{fold}'for n in ['on_ctld256','on_ctl128s3','on_ctl128s4','on_ctlc500']]
   if mode=='fp16':assert m['child_checkpoints']==quantmeta['child_checkpoints']
   else:quantmeta=m
   arr=np.load(f,mmap_mode='r');assert arr.shape==(1873,20000,2)and arr.dtype==np.float32
   chosen=set(selected)
   for i in range(len(groups)):
    assert np.isfinite(arr[i]).all()if i in chosen else np.isnan(arr[i]).all()
   arrays[mode]=arr;provenance[f'{mode}_fold{fold}']={'prediction_sha256':digest,'receipt_sha256':sha(f.with_suffix('.json'))}
  for i in selected:
   if not info[int(groups[i])][2]:continue
   mm=mask[i:i+1].copy();mm[:,:99]=False
   for mode in stats:stats[mode].append(fe.sums(y[i:i+1],arrays[mode][i:i+1],mm)[0])
   foldids.append(fold);positions.append(i)
 stats={k:np.stack(v)for k,v in stats.items()};foldids=np.asarray(foldids)
 def result(sel):
  q=fe.pooled(stats['quant'][sel].sum(0));r=fe.pooled(stats['fp16'][sel].sum(0));return {'quant_wp':q,'same_member_FP16_wp':r,'delta':q-r}
 report={'status':'COMPLETE_FIXED_COMPARISON','primary':result(np.ones(len(foldids),bool)),'by_fold':{str(f):result(foldids==f)for f in [0,1]},'pure_sequences':len(foldids),'member_order':['d256','h128s3','h128s4','c500'],'coefficients':[.25]*4,'native_quant_vs_native_FP16':True,'no_50minute_or_point_gain_gate':True,'exposed_folds_no_fresh_confirmation':True,'provenance':provenance,'metric_source_sha256':sha(metric),'allocation_sha256':sha(P/'allocation.json')}
 for mode in stats:
  for fold in [0,1]:assert sha(P/'cache'/f'pred_on_q4{mode}_f{fold}_valid.npy')==provenance[f'{mode}_fold{fold}']['prediction_sha256']
 np.savez(P/'quality-sufficient.npz',**stats,fold_ids=foldids,positions=np.asarray(positions));output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
