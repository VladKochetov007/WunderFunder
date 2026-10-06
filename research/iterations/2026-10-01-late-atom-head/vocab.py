"""Fixed64 dictionary from original-training-only chains. No validation targets."""
from pathlib import Path
import argparse, collections, hashlib, json, time
import numpy as np
import pyarrow.parquet as pq
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
Q=0.65687
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def identity(p):
 p=Path(p);s=p.stat();return dict(resolved=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True);a=ap.parse_args()
 out=a.output_dir.resolve();assert out.is_relative_to(HERE) and out!=HERE;out.mkdir(exist_ok=False)
 inputs=json.loads(a.inputs.read_text());paths=[ROOT/'datasets/train.parquet',ROOT/'cache/train_targets.npz',ROOT/'cache/chain_folds.npz']
 identities={str(p):identity(p) for p in paths}
 # Check the existing remote stat-bound large-input provenance without another full scan.
 for p in paths:
  matches=[r for r in inputs['records'] if Path(r['path']).resolve()==p.resolve()];assert len(matches)==1,str(p)
  r=matches[0]
  for k,v in identities[str(p)].items():assert r[k]==v,(str(p),k)
  if r.get('sha256') and p.stat().st_size <=20*(1<<20):assert sha(p)==r['sha256']
 with np.load(paths[1],allow_pickle=False) as z:groups=z['row_group'];classes=z['market_class']
 with np.load(paths[2],allow_pickle=False) as z:
  valid_chains=set(z['chain'][z['split']=='valid'].tolist())
  train_map={int(g):int(c) for g,c in zip(z['row_group'][z['split']=='train'],z['chain'][z['split']=='train'])}
 assert len(groups)==10606 and 528 not in groups and len(set(groups.tolist()))==len(groups)
 allowed=[int(g) for g,c in zip(groups,classes) if int(c)==3 and train_map[int(g)] not in valid_chains]
 assert allowed and 528 not in allowed and not {train_map[g] for g in allowed}&valid_chains
 masses=collections.defaultdict(float);counts=collections.Counter();pf=pq.ParquetFile(paths[0]);started=time.monotonic();rows=0
 for i,g in enumerate(allowed):
  t=pf.read_row_group(g,columns=['t0','t1'],use_threads=False);assert t.num_rows==20000
  y=np.column_stack([t.column(c).to_numpy() for c in ('t0','t1')])[99:].astype(np.float64)
  assert np.isfinite(y).all();ticks=np.rint(y/Q).astype(np.int64)
  assert np.max(np.abs(ticks*Q-y))<1e-3,('noncanonical class3',g)
  u,inv=np.unique(ticks,axis=0,return_inverse=True);w=np.abs(np.clip(y,-2,2)).mean(-1)
  mass=np.bincount(inv,weights=w,minlength=len(u));count=np.bincount(inv,minlength=len(u));rows+=len(y)
  for pair,m,n in zip(u,mass,count):key=tuple(map(int,pair));masses[key]+=float(m);counts[key]+=int(n)
  if i%100==0:print(json.dumps(dict(groups_done=i+1,groups_total=len(allowed))),flush=True)
 keys=sorted((k for k in masses if masses[k]>0),key=lambda k:(-masses[k],k[0],k[1]));assert len(keys)>=64,('insufficient support',len(keys))
 selected=keys[:64];atoms=(np.asarray(selected,np.float64)*Q).astype(np.float32);assert (np.abs(np.clip(atoms,-2,2)).mean(-1)>0).all()
 np.save(out/'atoms.npy',atoms,allow_pickle=False)
 for p in paths:assert identity(p)==identities[str(p)]
 rec=dict(status='COMPLETE_TRAIN_ONLY_VOCAB',source_sha256=sha(__file__),inputs=str(a.inputs),inputs_sha256=sha(a.inputs),input_identity=identities,chain_folds_sha256=sha(paths[2]),groups=allowed,chains=sorted({train_map[g] for g in allowed}),excluded_all_valid_sharing_chains=True,read_columns=['t0','t1'],warmup=99,rows=rows,support=len(keys),quantum=Q,atoms=64,selected_ticks=selected,selected_counts=[counts[k] for k in selected],selected_mass=[masses[k] for k in selected],total_positive_mass=sum(masses.values()),selected_mass_fraction=sum(masses[k] for k in selected)/sum(masses.values()),atoms_sha256=sha(out/'atoms.npy'),seconds=time.monotonic()-started,metrics=0,fits=0)
 (out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)
if __name__=='__main__':main()
