"""Metadata-only input inventory; later assemble complete partner bindings. No Torch.

Preparation reads only NPZ row maps/chain metadata, NPY headers, small receipts,
file stats and byte hashes. It never loads target/mask/prediction tensor values.
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OLD=ROOT/'research/iterations/2026-10-01-overnight/ensemble'
PRIOR=ROOT/'research/iterations/2026-10-01-online-normalization/input-bindings.json'
PRIOR_SHA='6c05909ebbfa415658709e42dd52637d3f8430d8d78d089278a4090741fda548'
AUDIT=OLD/'cache-completion-audit.json';AUDIT_SHA='6b7fddf06661a1b5b6b76ded41f41f8ec15a15c8327a479f76a90934326d75a4'
SLOTS=('128s3','128s4','c500')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rel(p):return str(Path(p).relative_to(ROOT))
def path(r):
 p=Path(r);assert not p.is_absolute() and '..' not in p.parts
 return ROOT/p

def stat(p):
 p=Path(p);s=p.stat();return {'resolved':str(p.resolve()),'size':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev}
def checkstat(p,r):
 s=stat(p);assert all(s[k]==r[k] for k in s),('input changed',str(p));return s

def header(p):
 with Path(p).open('rb') as f:
  v=np.lib.format.read_magic(f)
  shape,order,dtype=(np.lib.format.read_array_header_1_0(f) if v==(1,0) else np.lib.format.read_array_header_2_0(f))
 assert not order and dtype==np.dtype('float32')
 return list(shape)

def write(p,r):
 assert p.resolve().is_relative_to(HERE) and not p.exists() and not p.is_symlink()
 with p.open('x') as f:json.dump(r,f,indent=2);f.write('\n')

def verify_inputs(p,digest):
 assert sha(p)==digest;r=json.loads(p.read_text())
 assert r['schema']=='ensemble-backbone-inputs-v1' and r['status']=='PREPARED_INPUT_IDENTITIES'
 for name,h in r['source_sha256'].items():assert sha(path(name))==h,('source changed',name)
 for item in r['records']:
  checkstat(item['path'],item)
  if item.get('sha256'):assert sha(item['path'])==item['sha256']
 return r

def prepare(output):
 assert sha(PRIOR)==PRIOR_SHA and sha(AUDIT)==AUDIT_SHA
 prior=json.loads(PRIOR.read_text());audit=json.loads(AUDIT.read_text());assert audit['status']=='PASS'
 records=[]
 for old in prior['records']:
  checkstat(old['path'],old)
  if old.get('sha256'):assert sha(old['path'])==old['sha256']
  records.append(dict(old))
 def pin(p,digest=None,large=False):
  before=stat(p);actual=sha(p);assert digest is None or actual==digest,(str(p),actual,digest);checkstat(p,before)
  r={'path':str(p),**before,('prior_sha256' if large else 'sha256'):actual};records.append(r);return actual,before
 pin(PRIOR,PRIOR_SHA);pin(AUDIT,AUDIT_SHA)
 rowmaps={};groups={}
 for split,n in (('train',10606),('valid',1873)):
  p=ROOT/f'cache/{split}_targets.npz';old=next(r for r in records if r['path']==str(p))
  with np.load(p,allow_pickle=False) as z:g=z['row_group']
  assert len(g)==n and len(np.unique(g))==n and np.issubdtype(g.dtype,np.integer)
  if split=='train':assert set(map(int,g))==set(range(10607))-{528}
  groups[split]=g
  rowmaps[split]={'path':rel(p),'sha256':old.get('sha256') or old.get('prior_sha256'),'stat':stat(p),'row_group_sha256':hashlib.sha256(g.tobytes()).hexdigest(),'row_group_dtype':str(g.dtype),'rows':n,'row_groups':g.tolist(),'excluded':[528] if split=='train' else []}
 cfpath=ROOT/'cache/chain_folds.npz';cfsha=sha(cfpath)
 with np.load(cfpath,allow_pickle=False) as z:
  valid=z['split']=='valid';folds={int(g):int(f) for g,f in zip(z['row_group'][valid],z['fold'][valid])};chains={int(g):int(c) for g,c in zip(z['row_group'][valid],z['chain'][valid])}
 positions={str(f):[i for i,g in enumerate(groups['valid']) if folds[int(g)]!=f] for f in (0,1)}
 assert [len(positions[str(f)]) for f in (0,1)]==[1406,1404]
 partners={}
 for fold in (0,1):
  for slot in SLOTS:
   run=f'on_ctl{slot}_f{fold}';old=next(c for c in audit['checks'] if c['run']==run)
   assert old['fold']==fold and old['all_training_predictions_finite'] and old['all_held_predictions_NaN'] and old['chain_sets_disjoint']
   ck=ROOT/'artifacts/runs'/run/'model_ep1.pt';cksha,ckstat=pin(ck,old['checkpoint_sha256'])
   cache=OLD/'cache'/f'pred_{run}_training.npy';rp=cache.with_suffix('.json');rsha,rstat=pin(rp);r=json.loads(rp.read_text())
   assert r['run']==run and r['fold']==fold and r['epoch']==1 and r['role']=='eligible_training_rows'
   assert r['rows']==positions[str(fold)] and r['row_groups']==groups['valid'][positions[str(fold)]].tolist()
   assert r['source_row_group_order_sha256']==rowmaps['valid']['row_group_sha256'] and r['chain_folds_sha256']==cfsha
   assert r['checkpoint_sha256']==cksha and r['prediction_sha256']==old['prediction_sha256']
   assert r['WP_queries']==0 and r['target_columns_read'] is False and r['mask_read'] is False
   assert header(cache)==[1873,20000,2];csha,cstat=pin(cache,r['prediction_sha256'],large=True)
   hold=[i for i in range(1873) if i not in set(positions[str(fold)])]
   assert not {chains[int(groups['valid'][i])] for i in hold}&{chains[int(groups['valid'][i])] for i in positions[str(fold)]}
   partners[run]={'run':run,'fold':fold,'checkpoint':{'path':rel(ck),'sha256':cksha,'stat':ckstat,'epoch':1},'valid':{'path':rel(cache),'sha256':csha,'stat':cstat,'receipt_path':rel(rp),'receipt_sha256':rsha,'receipt_stat':rstat,'shape':[1873,20000,2],'dtype':'float32','held_rows':'NaN; same bytes as preserved completed audit'}}
 sources=['scripts/train_rnn.py','src/wnn/data.py','src/wnn/features.py','src/wnn/nn.py','src/wnn/resources.py','src/wnn/outcomes.py','src/wnn/propensity.py','src/wnn/splits.py','src/wnn/gmetric.py','src/wnn/metric.py','src/wnn/event_attn.py',rel(HERE/'cache_partners.py'),rel(Path(__file__))]
 result={'schema':'ensemble-backbone-inputs-v1','status':'PREPARED_INPUT_IDENTITIES','created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':{s:sha(path(s)) for s in sources},'prior_bindings':{'path':rel(PRIOR),'sha256':PRIOR_SHA},'prior_valid_audit':{'path':rel(AUDIT),'sha256':AUDIT_SHA},'records':records,'row_maps':rowmaps,'chain_folds':{'path':rel(cfpath),'sha256':cfsha,'stat':stat(cfpath)},'training_valid_positions':positions,'partners':partners,'large_input_identity_limit':'Existing N data/teachers/row-map files retain registered stat/inode and historical hashes where known. Target NPZ full hashes can be null; row_group metadata is freshly hashed. Six valid partner caches freshly byte-hashed against historical finite/heldNaN audit. No target, mask or prediction arrays decoded; no Torch.'}
 # Verify every file identity again after the inventory; no repeated large hashes.
 for r in records:checkstat(r['path'],r)
 write(output,result);print(json.dumps({'status':result['status'],'path':rel(output),'sha256':sha(output),'partners':6,'records':len(records)}))

def assemble(inputs,digest):
 r=verify_inputs(inputs,digest)
 for fold in (0,1):
  dest=HERE/f'partner-bindings_f{fold}.json';assert not dest.exists() and not dest.is_symlink();members=[]
  for slot in SLOTS:
   run=f'on_ctl{slot}_f{fold}';member=dict(r['partners'][run]);p=HERE/'cache'/f'pred_{run}_train.npy';rp=p.with_suffix('.json')
   receipt=json.loads(rp.read_text());assert receipt['schema']=='ensemble-backbone-train-cache-v1' and receipt['status']=='COMPLETE'
   assert receipt['run']==run and receipt['fold']==fold and receipt['epoch']==1 and receipt['role']=='all_train_rows'
   assert receipt['checkpoint_sha256']==member['checkpoint']['sha256'] and receipt['inputs_sha256']==digest
   assert receipt['source_code_sha256']==r['source_sha256'][rel(HERE/'cache_partners.py')]
   assert receipt['rows']==list(range(10606)) and receipt['row_groups']==r['row_maps']['train']['row_groups']
   assert receipt['source_row_group_order_sha256']==r['row_maps']['train']['row_group_sha256'] and receipt['all_predictions_finite'] is True
   assert receipt['target_columns_read'] is False and receipt['mask_read'] is False and receipt['WP_queries']==0
   assert header(p)==[10606,20000,2];before=stat(p);assert sha(p)==receipt['prediction_sha256'];checkstat(p,before)
   member['train']={'path':rel(p),'sha256':receipt['prediction_sha256'],'stat':before,'receipt_path':rel(rp),'receipt_sha256':sha(rp),'receipt_stat':stat(rp),'shape':[10606,20000,2],'dtype':'float32'}
   members.append(member)
  obj={'schema':'ensemble-backbone-partners-v1','fold':fold,'partner_order':[x['run'] for x in members],'row_maps':r['row_maps'],'chain_folds':r['chain_folds'],'training_valid_positions':r['training_valid_positions'][str(fold)],'members':members,'source_input_receipt':{'path':rel(inputs),'sha256':digest},'assembly_source_sha256':sha(__file__),'all_selected_cache_values_finite_provenance':'new train producer online check; reused valid-cache complete audit + fresh bound hash','no_prediction_value_load_in_assembly':True}
  verify_inputs(inputs,digest);write(dest,obj);print(json.dumps({'path':rel(dest),'sha256':sha(dest),'fold':fold,'members':3}))

def main():
 ap=argparse.ArgumentParser();mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--assemble',action='store_true');ap.add_argument('--output',type=Path,default=HERE/'input-bindings.json');ap.add_argument('--inputs',type=Path,default=HERE/'input-bindings.json');ap.add_argument('--inputs-sha256');a=ap.parse_args()
 if a.prepare:prepare(a.output)
 else:assert a.inputs_sha256;assemble(a.inputs,a.inputs_sha256)
if __name__=='__main__':main()
