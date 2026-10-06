"""Native composite OOF prediction, no targets/masks, allocation gated; grader unchanged."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,zipfile
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--avx2-audit-allocation',type=Path);a=ap.parse_args();start=time.monotonic();manifest=a.manifest.resolve();m=json.loads(manifest.read_text());manifest_sha=sha(manifest)
 assert m['mode'] in ('quant','fp16'), 'only two fixed Q4 modes allocated'
 assert m['schema']=='q4-composite-checkpoint-v1' and m['kind']=='ensemble_recipe_manifest_not_torch_checkpoint' and m['epoch']==1 and m['fold'] in [0,1]
 assert m['predictor_source_sha256']==sha(__file__) and m['allocation_sha256']==sha(m['allocation']);alloc=json.loads(Path(m['allocation']).read_text());assert alloc.get('Q4_heldfold_predictions_authorized') is True and alloc.get('runtime_gate') in ('PASS','ALLOCATED_SEPARATE_FINAL_RUNTIME')
 for pathkey,hashkey in [('native','native_sha256'),('zip','zip_sha256'),('data_reader_source','data_reader_source_sha256')]:assert sha(m[pathkey])==m[hashkey]
 assert m['zip_sha256']==alloc['zip_sha256'];assert len(m['children'])==4
 audit_allocation=None
 if m['mode']=='quant_avx2':
  assert a.avx2_audit_allocation is not None;audit_allocation=json.loads(a.avx2_audit_allocation.read_text())
  assert audit_allocation.get('Q3_avx2_identity_audit_authorized') is True and audit_allocation.get('avx2_runtime_gate')=='PASS' and audit_allocation['zip_sha256']==m['zip_sha256']
 if m['mode'].startswith('quant'):
  assert m['zip_native_entry']==f'{m["native_cpu_level"]}/{Path(m["native"]).name}'
  with zipfile.ZipFile(m['zip']) as z:assert hashlib.sha256(z.read(m['zip_native_entry'])).hexdigest()==m['native_sha256']
 weights=[];expect=['on_ctld256','on_ctl128s3','on_ctl128s4','on_ctlc500']
 def verify_children():
  for i,c in enumerate(m['children']):
   assert c['run']==f'{expect[i]}_f{m["fold"]}' and c['epoch']==1 and c['fold']==m['fold'] and c['coef_float64']==.25 and c['coef_float32_hex']==np.float32(.25).tobytes().hex()
   assert sha(c['checkpoint'])==c['checkpoint_sha256'] and sha(c['export'])==c['export_sha256']
 verify_children()
 for c in m['children']:
  with np.load(c['export'],allow_pickle=False) as z:
   assert set(z.files)==set(c['arrays']);w={k:np.ascontiguousarray(z[k]) for k in z.files}
  for k,v in w.items():assert str(v.dtype)==c['arrays'][k]['dtype'] and list(v.shape)==c['arrays'][k]['shape'] and hashlib.sha256(v.tobytes()).hexdigest()==c['arrays'][k]['sha256']
  weights.append(w)
 sys.path.insert(0,str(Path(m['native']).parent));import wnn_cpp
 assert sha(wnn_cpp.__file__)==m['native_sha256']
 if m['mode'].startswith('quant'):
  if m['mode']=='quant':assert m['quant_backend']=='vnni' and wnn_cpp.VNNI_SUPPORTED and m['native_cpu_level']=='v3'
  else:assert m['mode']=='quant_avx2' and m['quant_backend']=='avx2' and wnn_cpp.AVX2_QUANT_SUPPORTED and m['native_cpu_level']=='avx2'
  engine=wnn_cpp.Ensemble(weights,[.25]*4,fast_activations=True,dynamic_int8=True,quant_backend=m['quant_backend'])
 else:
  assert m['mode']=='fp16' and wnn_cpp.HALF_SUPPORTED;engine=wnn_cpp.Ensemble(weights,[.25]*4,fast_activations=True,half_weights=True)
 sys.path.insert(0,str(ROOT/'src'));from wnn.data import SequenceFile
 groups=np.load(ROOT/'cache/valid_targets.npz')['row_group'];assert hashlib.sha256(groups.tobytes()).hexdigest()==m['source_row_group_order_sha256'] and len(groups)==1873
 assert sha(ROOT/'cache/chain_folds.npz')==m['chain_folds_sha256'];cf=np.load(ROOT/'cache/chain_folds.npz');v=cf['split']=='valid';folds={int(g):int(f) for g,f in zip(cf['row_group'][v],cf['fold'][v])};rows=np.asarray([i for i,g in enumerate(groups) if folds[int(g)]==m['fold']],dtype=np.int64)
 (P/'cache').mkdir(exist_ok=True)
 name={'quant':'on_q4quant','fp16':'on_q4fp16','quant_avx2':'on_q4avx2'}[m['mode']];path=P/'cache'/f'pred_{name}_f{m["fold"]}_valid.npy';partial=path.with_suffix('.npy.partial');receipt=path.with_suffix('.json');assert not any(p.exists() for p in [path,partial,receipt])
 reference_path=None
 if m['mode']=='quant_avx2':
  reference_path=P/'cache'/f'pred_on_q4quant_f{m["fold"]}_valid.npy';reference_meta=json.loads(reference_path.with_suffix('.json').read_text());assert reference_meta['prediction_sha256']==sha(reference_path)
  assert reference_meta['child_checkpoints']==m['children'] and reference_meta['epoch']==1 and reference_meta['fold']==m['fold'] and reference_meta['source_row_group_order_sha256']==m['source_row_group_order_sha256'] and reference_meta['zip_sha256']==m['zip_sha256']
 out=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(1873,20000,2));out[:]=np.nan;data=SequenceFile(ROOT/'datasets/valid.parquet')
 for count,pos in enumerate(rows):
  sequence=data.read(int(groups[pos]),targets=False,mask=False);assert sequence.targets is None and sequence.scored is None
  pred=engine.run(sequence.features);assert pred.shape==(20000,2) and pred.dtype==np.float32 and np.isfinite(pred).all();out[pos]=pred
  if count%50==0:print(json.dumps({'mode':m['mode'],'fold':m['fold'],'completed':count+1,'total':len(rows)}),flush=True)
 out.flush();del out;verify_children();assert sha(manifest)==manifest_sha;partial.replace(path)
 r={'run':name,'checkpoint':str(manifest),'checkpoint_kind':m['kind'],'checkpoint_sha256':manifest_sha,'child_checkpoints':m['children'],'epoch':1,'fold':m['fold'],'rows':rows.tolist(),'row_groups':groups[rows].tolist(),'source_row_group_order_sha256':m['source_row_group_order_sha256'],'chain_folds_sha256':m['chain_folds_sha256'],'prediction_sha256':sha(path),'prediction_path':str(path),'native_sha256':m['native_sha256'],'zip_sha256':m['zip_sha256'],'sum':m['sum'],'coefficients':[.25]*4,'nonselected_rows':'NaN','seconds':time.monotonic()-start,'targets_read':False,'masks_read':False,'metrics_computed':0,'source_sha256':sha(__file__)}
 if reference_path is not None:
  assert sha(reference_path)==reference_meta['prediction_sha256'];match=r['prediction_sha256']==reference_meta['prediction_sha256']
  r['avx2_output_identity_audit']={'status':'MATCH' if match else 'MISMATCH','reference':str(reference_path),'reference_prediction_sha256':reference_meta['prediction_sha256'],'entire_array_file_bytes_identical':match,'heldfold_quality':'same output as qualified VNNI only if MATCH; otherwise unqualified, no rescue readout','allocation':str(a.avx2_audit_allocation.resolve()),'allocation_sha256':sha(a.avx2_audit_allocation)}
 receipt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['run','fold','prediction_sha256','seconds']}))
if __name__=='__main__':main()
