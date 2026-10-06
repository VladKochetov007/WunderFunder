"""Train-target-only censor coverage; no model import, fit, prediction or WP."""
from pathlib import Path
import argparse,hashlib,json,struct,time,zipfile
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()

def stored_array(path,name):
 """Map an np.savez ZIP_STORED .npy payload without inflating the1.7GiB y array."""
 with zipfile.ZipFile(path) as z:
  info=z.getinfo(name+'.npy');assert info.compress_type==zipfile.ZIP_STORED,'Require existing uncompressed cache; no conversion fallback'
 with Path(path).open('rb') as f:
  f.seek(info.header_offset);h=f.read(30);assert h[:4]==b'PK\x03\x04'
  n,e=struct.unpack('<HH',h[26:30]);f.seek(n+e,1)
  version=np.lib.format.read_magic(f)
  if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
  elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
  else:raise ValueError(('Unsupported cache npy version',version))
  offset=f.tell();assert not order and not dtype.hasobject
  assert np.prod(shape,dtype=np.int64)*dtype.itemsize<=info.file_size
 return np.memmap(path,mode='r',dtype=dtype,shape=shape,offset=offset,order='C')

def masks(y):
 """Same float32 target contrast and strictly-future event semantics as trainer."""
 assert y.dtype==np.float32 and y.ndim==2 and y.shape[1]==2
 n=len(y);rows=np.arange(n);d=y[:,0]-y[:,1]
 changed=np.zeros(n,bool);changed[1:]=d[1:]!=d[:-1]
 marker=np.where(changed,rows,n);at=np.minimum.accumulate(marker[::-1])[::-1]
 nxt=np.r_[at[1:],n];censored=nxt==n
 age=rows-np.maximum.accumulate(np.where(changed,rows,0))
 remaining=n-1-rows
 return dict(changed=changed,censored=censored,eventful_tail=censored & bool(changed.any()),
             no_event_sequence=censored & (not bool(changed.any())),informative=remaining>0,
             target_quiet=age>=64,lower_bound=(np.log1p(remaining.astype(np.float64))-3)/2,
             trainer_sentinel=(np.log1p((n-rows).astype(np.float64))-3)/2)

NAMES=('eligible','right_censored','informative_censored','eventful_tail','informative_eventful_tail','no_event_sequence')
def empty():return {scope:{name:[0,0.] for name in NAMES} for scope in ('all','target_quiet')}
def update(dst,y,m):
 valid=np.arange(len(y))>=99;w=np.abs(np.clip(y,-2,2)).astype(np.float64).sum(1)
 categories={'eligible':np.ones(len(y),bool),'right_censored':m['censored'],
  'informative_censored':m['censored']&m['informative'],'eventful_tail':m['eventful_tail'],
  'informative_eventful_tail':m['eventful_tail']&m['informative'],'no_event_sequence':m['no_event_sequence']}
 for scope,base in (('all',valid),('target_quiet',valid&m['target_quiet'])):
  for name,keep in categories.items():
   mask=base&keep;dst[scope][name][0]+=int(mask.sum());dst[scope][name][1]+=float(w[mask].sum())

def summarize(total):
 out={}
 for scope,items in total.items():
  nr,nw=items['eligible'];out[scope]={}
  for name,(r,w) in items.items():
   other=max(0.,nw-w)
   out[scope][name]=dict(rows=r,target_mass=w,row_fraction=r/nr if nr else None,
    target_mass_fraction=w/nw if nw else None,
    hypothetical_since_factor_fraction_lower=w/(w+4*other) if nw else None,
    hypothetical_since_factor_fraction_upper=4*w/(4*w+other) if nw else None)
 return out

def identity(path):
 s=path.stat();return dict(resolved=str(path.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol-sha256',required=True);ap.add_argument('--source-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 t=time.monotonic();protocol=HERE/'protocol.json';assert sha(protocol)==a.protocol_sha256 and sha(Path(__file__))==a.source_sha256
 out=a.output.resolve();assert out.is_relative_to(HERE) and not out.exists() and not out.is_symlink()
 spec=json.loads(protocol.read_text());assert sha(ROOT/spec['trainer_source'])==spec['trainer_sha256']
 bp=ROOT/spec['input_binding'];assert sha(bp)==spec['input_binding_sha256'];records=json.loads(bp.read_text())['records'];bound={};states={}
 for relative in spec['inputs']:
  path=ROOT/relative;r=next(r for r in records if r['path']==str(path));s=identity(path)
  assert all(r[k]==v for k,v in s.items()),('Input binding changed',relative)
  if s['size']<=20*2**20:assert sha(path)==r['sha256']
  prior_hash=r.get('sha256',r.get('prior_sha256'))
  states[relative]=s;bound[relative]=dict(**s,previously_verified_sha256=prior_hash,full_hash_previously_verified=prior_hash is not None)
 path=ROOT/'cache/train_targets.npz';y=stored_array(path,'y');mask=stored_array(path,'mask')
 with np.load(path,allow_pickle=False) as z:groups=z['row_group']
 assert y.shape==(10606,20000,2) and y.dtype==np.float32 and mask.shape==y.shape[:2] and mask.dtype==np.bool_
 assert len(set(groups.tolist()))==10606 and set(groups.tolist())==set(range(10607))-{528}
 with np.load(ROOT/'cache/chain_folds.npz',allow_pickle=False) as z:
  keep=z['split']=='train';chains={int(g):int(c) for g,c in zip(z['row_group'][keep],z['chain'][keep])}
 assert set(groups.tolist())<=set(chains)
 total=empty();halves=[empty(),empty()];perchain={};examples=[];noevents=0;max_bound=-float('inf');min_sentinel_gap=float('inf')
 for i,g in enumerate(groups):
  assert time.monotonic()-t<54,'Stop before60second resource budget; no adaptive sample fallback'
  yy=np.asarray(y[i]);assert np.isfinite(yy).all() and not mask[i].any()
  c=chains[int(g)];half=int(hashlib.sha256(f'TARGET_CENSOR_COVERAGE01|half|{c}'.encode()).hexdigest(),16)%2
  m=masks(yy);update(total,yy,m);update(halves[half],yy,m)
  if c not in perchain:perchain[c]=dict(half=half,sequences=0,no_event_sequences=0,informative_eventful_tail_rows=0,target_mass=0.,informative_eventful_tail_mass=0.)
  row=perchain[c];row['sequences']+=1;row['no_event_sequences']+=int(not m['changed'].any());noevents+=int(not m['changed'].any())
  valid=np.arange(20000)>=99;keep=valid&m['eventful_tail']&m['informative'];w=np.abs(np.clip(yy,-2,2)).astype(np.float64).sum(1)
  row['informative_eventful_tail_rows']+=int(keep.sum());row['target_mass']+=float(w[valid].sum());row['informative_eventful_tail_mass']+=float(w[keep].sum())
  cens=valid&m['censored']&m['informative']
  if cens.any():
   max_bound=max(max_bound,float(m['lower_bound'][cens].max()));min_sentinel_gap=min(min_sentinel_gap,float((m['trainer_sentinel']-m['lower_bound'])[cens].min()))
 hs=[summarize(h) for h in halves];gate=[h['all']['informative_eventful_tail']['hypothetical_since_factor_fraction_upper'] for h in hs]
 passed=all(v is not None and v>=spec['one_fixed_gate']['threshold_each_half'] for v in gate)
 for relative,s in states.items():assert identity(ROOT/relative)==s
 assert sha(protocol)==a.protocol_sha256 and sha(Path(__file__))==a.source_sha256 and sha(bp)==spec['input_binding_sha256']
 result=dict(status='COMPLETE_TRAIN_TARGET_CENSOR_COVERAGE',gate_decision='POTENTIALLY_MATERIAL_TARGET_CENSORING' if passed else 'STOP_LOW_TARGET_COVERAGE',gate_values=gate,gate=spec['one_fixed_gate'],
  population='Original train component only; no valid contribution or H2 claim',sequences=10606,chains=len(perchain),no_event_sequences=noevents,
  summaries=summarize(total),chain_halves=hs,per_chain={str(k):v for k,v in sorted(perchain.items())},
  duration_semantics=dict(actual_lower_bound_max_standardized=max_bound if np.isfinite(max_bound) else None,trainer_sentinel_exceeds_actual_bound_min=min_sentinel_gap if np.isfinite(min_sentinel_gap) else None,time_clipping=False,last_row_noninformative=True),
  cached_auxiliary_prediction_status='Not in frozen inputs; global cache existence not audited',lower_bound_prediction_violations=None,predictions_generated=0,WP_reads=0,forecast_fits=0,validation_targets_read=False,raw_features_read=False,
  input_bindings=bound,protocol_sha256=a.protocol_sha256,source_sha256=a.source_sha256,seconds=time.monotonic()-t)
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ('status','gate_decision','gate_values','sequences','chains','no_event_sequences','seconds')}))

if __name__=='__main__':main()
