"""Fixed same-row GRU-context diagnostic. No training, model/row selection or metric calls.

Both AMP and FP32 companions are mandatory. Inputs211 are built once from each whole
raw sequence, before any GRU reset. Registered historical inputs may resolve through
read-only repository symlinks; new outputs must remain physically under HERE.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, os, resource, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BINDINGS='research/iterations/2026-10-01-continuation/student-seed4/bindings-v2-01.json'
BINDINGS_SHA='e6be669708f25081362e4352da5a5d1e8e6dc8587868b95acded72e427c02f19'
STARTS=(4096,8192,12288,16384); LENGTH=512; CONTEXTS=(256,2048)
STEPS=np.concatenate([np.arange(s,s+LENGTH,dtype=np.int64) for s in STARTS])
ENGINES=('amp_fp16','fp32_tf32_off'); ARMS=('full','reset256','reset2048')
MEMBERS=('cont_sh256_s3h2','on_ctl128s3','on_ctl128s4','on_ctlc500')
META=('cache_position','row_group','step','chain','fold','market_class_retrospective','age_i0')
EXPECTED_SEQ=860;EXPECTED_ROWS=EXPECTED_SEQ*len(STEPS)
CACHE_MAXABS=3e-3;CACHE_RMSE=3e-4

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def repository_input(rel):
 p=Path(rel)
 if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('unsafe repository-relative input')
 return ROOT/p

def boundaries(total=20000,starts=STARTS,length=LENGTH):
 # Common 512-row GEMM shapes for all scored blocks; full path carries true state.
 cuts={0,total,*starts,*(s+length for s in starts)}
 ordered=sorted(cuts);out=[0]
 for end in ordered[1:]:
  while end-out[-1]>4000:out.append(out[-1]+4000)
  out.append(end)
 return list(zip(out[:-1],out[1:]))

def core(forward,clone,inputs,total=20000,starts=STARTS,length=LENGTH,contexts=CONTEXTS,verify=True):
 """forward(inputs, lo, hi, h)->(numpy float32 main predictions, state).
 No labels/masks/origins are inferred here. Useful with a generated CPU oracle too.
 """
 full=[];state=None;checks=[]
 for lo,hi in boundaries(total,starts,length):
  incoming=clone(state) if verify and lo==starts[0] else None
  out,state=forward(inputs,lo,hi,state);full.append(out)
  if verify and lo==starts[0]:
   repeat,rstate=forward(inputs,lo,hi,incoming)
   checks.append((out.dtype==repeat.dtype and out.shape==repeat.shape and out.tobytes()==repeat.tobytes(),clone(('equal',state,rstate))))
 full=np.concatenate(full,axis=1)
 outputs=[np.concatenate([full[:,s:s+length] for s in starts],axis=1)]
 for context in contexts:
  blocks=[]
  for s in starts:
   _,h=forward(inputs,s-context,s,None)
   out,_=forward(inputs,s,s+length,h);blocks.append(out)
  outputs.append(np.concatenate(blocks,axis=1))
 return full,np.stack(outputs,axis=2),checks

def schema():
 return {'file':'replay-rows.npz','metadata':list(META),'engines':list(ENGINES),'arms':list(ARMS),
 'arrays':{'metadata':'int32[N,7]','y':'float32[N,2]','is_scored':'bool[N]',
 'member_predictions':'float32[N,2engines,3arms,2targets]','partner_predictions':'float32[N,3partners,2]',
 'cached_shallow_reference':'float32[N,2]'},'partners':list(MEMBERS[1:]),'rows':EXPECTED_ROWS,
 'ensemble_arithmetic':'Sequential float32 (((member+partner0)+partner1)+partner2)/4, then scorer clipping.',
 'interpretation':'GRU-only state intervention with unchanged fullprefix features; OOD state, not natural suffix restart.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path);ap.add_argument('--plan-sha256');ap.add_argument('--output-dir',type=Path);ap.add_argument('--schema',action='store_true');a=ap.parse_args()
 if a.schema:print(json.dumps(schema(),indent=2));return
 assert a.plan and a.plan_sha256 and a.output_dir
 assert sha(a.plan)==a.plan_sha256
 plan=json.loads(a.plan.read_text());assert plan['execution_authorized'] is True
 source_hashes=plan['source_sha256'];self_hash=source_hashes[str(Path(__file__).resolve().relative_to(ROOT))];assert self_hash==sha(__file__)
 assert plan.get('batch',16)==16 and plan.get('engines',list(ENGINES))==list(ENGINES)
 out=a.output_dir.resolve();assert out.is_relative_to(HERE) and out!=HERE and not out.exists()
 out.mkdir();started=time.monotonic();tracked={}
 def pin(path,digest):
  assert sha(path)==digest,('hash mismatch',str(path));tracked[str(path)]=digest
 pin(a.plan,a.plan_sha256);pin(Path(__file__),self_hash)
 required=('scripts/train_rnn.py','src/wnn/nn.py','src/wnn/features.py','src/wnn/data.py','src/wnn/resources.py','research/iterations/2026-10-01-continuation/student/predict_fixed.py')
 assert all(k in source_hashes for k in required)
 for rel,h in source_hashes.items():pin(repository_input(rel),h)
 pin(repository_input(BINDINGS),BINDINGS_SHA);bound=json.loads(repository_input(BINDINGS).read_text())
 for rel,rec in bound['data'].items():
  path=repository_input(rel);assert path.stat().st_size==rec['bytes'];pin(path,rec['sha256'])
 raw=ROOT/'datasets/valid.parquet';rawrec=plan['raw_input'];st=raw.stat()
 assert str(raw.resolve())==rawrec['resolved'] and st.st_size==rawrec['bytes'] and st.st_mtime_ns==rawrec['mtime_ns']
 assert len(rawrec['sha256'])==64
 rawstat=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
 with np.load(ROOT/'cache/valid_targets.npz',allow_pickle=False) as z:groups=z['row_group'];seqids=z['seq_ix'];classes=z['market_class'];Y=z['y'];M=z['mask']
 assert Y.shape==(1873,20000,2) and M.shape==(1873,20000)
 with np.load(ROOT/'research/artifacts/market-classes-v1.npz',allow_pickle=False) as z:assert np.array_equal(classes,z['valid'][groups])
 with np.load(ROOT/'cache/chain_folds.npz',allow_pickle=False) as z:
  v=z['split']=='valid';info={int(g):(int(f),int(c),bool(p)) for g,f,c,p in zip(z['row_group'][v],z['fold'][v],z['chain'][v],z['pure'][v],strict=True)}
 selected=[(f,i) for f in (0,1) for i,g in enumerate(groups) if info[int(g)][0]==f and info[int(g)][2]]
 assert len(selected)==EXPECTED_SEQ and len({info[int(groups[i])][1] for _,i in selected})==250
 group_sha=hashlib.sha256(groups.tobytes()).hexdigest();arrays={};ckpaths={}
 for f in (0,1):
  foldrows=[i for i,g in enumerate(groups) if info[int(g)][0]==f]
  for name in MEMBERS:
   rec=bound['predictions'][f'{name}_f{f}'];path=repository_input(rec['path']);receipt=repository_input(rec['receipt'])
   pin(path,rec['sha256']);pin(receipt,rec['receipt_sha256']);r=json.loads(receipt.read_text())
   assert r['run']==f'{name}_f{f}' and r['fold']==f and r['epoch']==1 and r['rows']==foldrows and r['row_groups']==groups[foldrows].tolist()
   assert r['source_row_group_order_sha256']==group_sha and r['prediction_sha256']==rec['sha256'] and r['checkpoint_sha256']==rec['checkpoint_sha256']
   arr=np.load(path,mmap_mode='r',allow_pickle=False);assert arr.shape==(1873,20000,2) and arr.dtype==np.float32;arrays[name,f]=arr
  p=ROOT/'artifacts/runs'/f'{MEMBERS[0]}_f{f}'/'model_ep1.pt';pin(p,bound['predictions'][f'{MEMBERS[0]}_f{f}']['checkpoint_sha256']);ckpaths[f]=p
 # Imports and GPU initialization only after identities and root authorization checked.
 sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
 import torch
 import train_rnn
 import pyarrow as pa
 from wnn.data import SequenceFile
 from wnn.resources import configure_torch
 assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')
 configure_torch();pa.set_cpu_count(1);pa.set_io_thread_count(1)
 cap=float(os.environ['WNN_GPU_FRACTION']);assert cap*torch.cuda.get_device_properties(0).total_memory<=4*2**30+1
 device=torch.device('cuda');data=SequenceFile(raw);assert len(data)==1873
 for name in ('RAW_ONLY','ONLINE_NORM','QUANT','QUANT_I1','HEURISTICS'):setattr(train_rnn,name,False)
 train_rnn.LOOKAHEAD=();train_rnn.STATE_RESET=0;train_rnn.READOUT=None;train_rnn.INPUT_PROPENSITY=None
 meta=np.empty((EXPECTED_ROWS,7),np.int32);yy=np.empty((EXPECTED_ROWS,2),np.float32);mask=np.empty(EXPECTED_ROWS,bool)
 preds=np.empty((EXPECTED_ROWS,2,3,2),np.float32);partners=np.empty((EXPECTED_ROWS,3,2),np.float32);cached=np.empty((EXPECTED_ROWS,2),np.float32)
 diff={e:{'count':0,'sum_squared':0.,'max_abs':0.} for e in ENGINES};checks=[];rawhash=hashlib.sha256();featurehash=hashlib.sha256();cursor=0;done=0
 initial_tf32={'matmul':torch.backends.cuda.matmul.allow_tf32,'cudnn':torch.backends.cudnn.allow_tf32}
 (out/'started.json').write_text(json.dumps({'PID':os.getpid(),'plan_sha256':a.plan_sha256,'schema':schema(),'initial_tf32':initial_tf32,'inputs':tracked,'raw_identity':rawrec,'raw_verification':'registered hash provenance + device/inode/size/mtime before/after; no full parquet rehash'},indent=2)+'\n')
 for fold in (0,1):
  ck=torch.load(ckpaths[fold],map_location='cpu',weights_only=False);args=ck['args']
  assert ck['epoch']==1 and args['valid_fold']==fold and ck['n_in']==211 and args['hidden']==256 and args['layers']==1
  assert args.get('arch','gru')=='gru' and args.get('cell','gru')=='gru' and args.get('enc','linear')=='linear' and args.get('dropout',0)==0
  assert not any(args.get(k) for k in ('lookahead','raw_only','online_norm','quant_feats','quant_i1','state_reset','class_heads','atom_head','heuristics'))
  model=train_rnn.make_model(args,211,ck['model']['head.2.weight'].shape[0]).to(device);model.load_state_dict(ck['model'],strict=True);model.eval()
  assert all(torch.isfinite(t).all() for t in model.state_dict().values())
  ids=[i for f,i in selected if f==fold]
  for offset in range(0,len(ids),16):
   batch=ids[offset:offset+16];raws=[]
   for i in batch:
    seq=data.read(int(groups[i]),targets=False,mask=False)
    assert seq.seq_ix==int(seqids[i]) and seq.targets is None and seq.scored is None and np.isfinite(seq.features).all()
    raws.append(seq.features);rawhash.update(seq.features.tobytes())
   x=torch.from_numpy(np.stack(raws)).to(device)
   with torch.no_grad():inputs=train_rnn.build_inputs(x)
   assert inputs.shape==(len(batch),20000,211) and inputs.dtype==torch.float32 and torch.isfinite(inputs).all()
   # Compute age from the identical event flags, not by inverse-transforming a float log.
   flag=inputs[:,:,178].detach().cpu().numpy()>0.5;idx=np.broadcast_to(np.arange(20000),(len(batch),20000));age=idx-np.maximum.accumulate(np.where(flag,idx,0),axis=1)
   featurehash.update(inputs.detach().cpu().numpy().tobytes())
   batchout=[];cached_full=np.stack([arrays[MEMBERS[0],fold][i] for i in batch]);assert np.isfinite(cached_full).all()
   for ename in ENGINES:
    amp=ename=='amp_fp16';torch.backends.cuda.matmul.allow_tf32=initial_tf32['matmul'] if amp else False;torch.backends.cudnn.allow_tf32=initial_tf32['cudnn'] if amp else False
    def clone(h):
     if isinstance(h,tuple) and h[0]=='equal':return bool(torch.equal(h[1],h[2]))
     return None if h is None else h.clone()
    def forward(inp,lo,hi,h):
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):v,state=train_rnn.forward_chunk(model,inp,slice(lo,hi),h)
     p=v[...,:2].float().cpu().numpy();assert np.isfinite(p).all() and torch.isfinite(state).all()
     return p,state
    full,rows,check=core(forward,clone,inputs,verify=offset==0)
    assert all(a and b for a,b in check),('incoming state reproduction',fold,ename,check)
    checks.extend({'fold':fold,'engine':ename,'batch_size':len(batch),'output_equal':a,'state_equal':b} for a,b in check)
    d=full.astype(np.float64)-cached_full.astype(np.float64);dd=diff[ename];dd['count']+=d.size;dd['sum_squared']+=float(np.sum(d*d));dd['max_abs']=max(dd['max_abs'],float(np.max(np.abs(d))))
    if amp:assert dd['max_abs']<=CACHE_MAXABS,('AMP cache maxabs',dd)
    batchout.append(rows)
   torch.backends.cuda.matmul.allow_tf32=initial_tf32['matmul'];torch.backends.cudnn.allow_tf32=initial_tf32['cudnn']
   for j,i in enumerate(batch):
    sl=slice(cursor,cursor+len(STEPS));g=int(groups[i]);chain=info[g][1]
    meta[sl]=np.column_stack([np.full(len(STEPS),i),np.full(len(STEPS),g),STEPS,np.full(len(STEPS),chain),np.full(len(STEPS),fold),np.full(len(STEPS),classes[i]),age[j,STEPS]])
    yy[sl]=Y[i,STEPS];mask[sl]=M[i,STEPS];preds[sl]=np.stack([p[j] for p in batchout],axis=1)
    partners[sl]=np.stack([arrays[name,fold][i,STEPS] for name in MEMBERS[1:]],axis=1);cached[sl]=cached_full[j,STEPS]
    assert np.isfinite(yy[sl]).all() and np.isfinite(partners[sl]).all()
    cursor+=len(STEPS);done+=1
   del inputs,x,raws,batchout
   print(json.dumps({'sequences_complete':done,'rows':cursor,'elapsed_s':time.monotonic()-started,'cuda_allocated_peak':torch.cuda.max_memory_allocated(),'cuda_reserved_peak':torch.cuda.max_memory_reserved()}),flush=True)
  del model,ck;torch.cuda.empty_cache()
 assert cursor==EXPECTED_ROWS and done==EXPECTED_SEQ and len(checks)==4
 for d in diff.values():d['rmse']=(d['sum_squared']/d['count'])**.5
 assert diff[ENGINES[0]]['rmse']<=CACHE_RMSE,('AMP cache RMSE',diff[ENGINES[0]])
 for p,h in tracked.items():assert sha(p)==h,('changed input',p)
 st=raw.stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==rawstat
 target=out/'replay-rows.npz';partial=target.with_suffix('.npz.partial')
 with partial.open('xb') as f:np.savez_compressed(f,metadata=meta,y=yy,is_scored=mask,member_predictions=preds,partner_predictions=partners,cached_shallow_reference=cached,metadata_names=np.array(META),engine_names=np.array(ENGINES),arm_names=np.array(ARMS),partner_names=np.array(MEMBERS[1:]),block_starts=np.array(STARTS),block_length=np.array(LENGTH))
 partial.replace(target)
 receipt={'status':'COMPLETE','parity_checks_passed':True,'source_sha256':sha(__file__),'plan_sha256':a.plan_sha256,'rows':cursor,'sequences':done,'chains':250,'engines':list(ENGINES),'arms':list(ARMS),'cached_comparison_all20000rows':diff,'AMP_cache_tolerances':{'max_abs':CACHE_MAXABS,'rmse':CACHE_RMSE},'FP32_vs_cache':'descriptive only; different arithmetic; no AMP tolerance applied','incoming_state_checks':checks,'inputs_unchanged_before_after':True,'raw_identity':rawrec,'raw_verification':'stat/inode before-after plus registered historical SHA; no new full hash','selected_raw_sha256':rawhash.hexdigest(),'fullprefix211_sha256':featurehash.hexdigest(),'outputs':{target.name:{'sha256':sha(target),'bytes':target.stat().st_size}},'schema':schema(),'seconds':time.monotonic()-started,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cuda_max_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_max_reserved_bytes':torch.cuda.max_memory_reserved(),'initial_tf32':initial_tf32,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'fits':0,'metric_calls':0}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'seconds':receipt['seconds'],'parity':diff}),flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  traceback.print_exc();raise
