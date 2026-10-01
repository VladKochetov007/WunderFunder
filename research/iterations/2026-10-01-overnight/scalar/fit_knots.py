"""Separately allocated train-only44-column quantile fit, no labels/masks."""
from pathlib import Path
import argparse,hashlib,json,os,resource,signal,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def fingerprint(path):
 st=path.stat()
 with path.open('rb') as f:
  f.seek(-8,2);tail=f.read(8);assert tail[4:]==b'PAR1';size=int.from_bytes(tail[:4],'little');f.seek(-size-8,2);footer=f.read(size)+tail
 return {'path':str(path),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev,'footer_bytes':size+8,'footer_sha256':hashlib.sha256(footer).hexdigest(),'whole_file_hashed':False}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);a=ap.parse_args();alloc=json.loads(a.allocation.read_text());assert alloc.get('SCALARPLE_train_only_knots_authorized') is True
 assert not alloc.get('forecast_fit_authorized',False)
 for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
 os.environ['CUDA_VISIBLE_DEVICES']='';resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30));resource.setrlimit(resource.RLIMIT_CPU,(480,480))
 signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('480second knotfit cap')));signal.alarm(480)
 import numpy as np
 import pyarrow.parquet as pq
 sys.path.insert(0,str(ROOT/'src'));from wnn.data import FEATURE_COLUMNS
 manifest_path=P.parent/'renewal/selected-chains.json';assert sha(manifest_path)=='13e39a60cdd88f5f0c35ea2c8abfe1558561c75cce4c8ddc8f1671446704a13d'
 manifest=json.loads(manifest_path.read_text());selected=manifest['selected'];assert len(selected)==256 and len({r['chain'] for r in selected})==256
 metadata=Path(manifest['historical_chain_metadata']['path']);assert sha(metadata)=='315b84f385f5368615c76087288b623f99c33bd680e9ebf7c3465bf5ab8afd35'
 with np.load(metadata,allow_pickle=False) as z:chain=z['chain'];split=z['split'].astype(str);rg=z['row_group']
 for row in selected:
  members=chain==row['chain'];match=(split=='train')&(rg==row['row_group']);assert match.sum()==1 and int(chain[match][0])==row['chain'] and members.any() and np.all(split[members]=='train')
 source=Path(manifest['source']['path']);assert source==Path('/home/grinchenkoeg/WunderFunder/datasets/train.parquet');before=fingerprint(source)
 for key in ['bytes','mtime_ns','footer_bytes','footer_sha256']:assert before[key]==manifest['source'][key],key
 cols=[FEATURE_COLUMNS[j] for j in list(range(22))+list(range(52,74))];assert cols==[f'i{instrument}_p{j}' for instrument in range(2) for j in range(22)]
 out=P/'knots';out.mkdir(exist_ok=False);started=time.monotonic();pf=pq.ParquetFile(source);samples=[];raw_hash=hashlib.sha256()
 for row in selected:
  assert pf.metadata.row_group(row['row_group']).num_rows==20000
  t=pf.read_row_group(row['row_group'],columns=cols,use_threads=False);assert t.column_names==cols
  x=np.column_stack([t[c].to_numpy(zero_copy_only=False) for c in cols]).astype(np.float32);assert x.shape==(20000,44) and np.isfinite(x).all()
  raw_hash.update(x.tobytes());samples.append(x[::20].copy())
 values=np.concatenate(samples);assert values.shape==(256000,44)
 q=np.quantile(values.astype(np.float64),np.linspace(0,1,16),axis=0,method='linear').T.astype(np.float32);assert np.isfinite(q).all();q[q==0]=np.float32(0)
 knots=np.empty((44,16),np.float32);counts=np.empty(44,np.int64)
 for j in range(44):
  k=np.unique(q[j]);n=len(k);assert 1<=n<=16 and np.all(np.diff(k.astype(np.float64))>0);counts[j]=n;knots[j,:n]=k;knots[j,n:]=k[-1]
 after=fingerprint(source);assert before==after and sha(metadata)=='315b84f385f5368615c76087288b623f99c33bd680e9ebf7c3465bf5ab8afd35'
 dest=out/'knots.npz';np.savez(dest,knots=knots,counts=counts)
 r={'status':'PASS','schema':'scalarple01-44x16-r16-v1','allocation_sha256':sha(a.allocation),'source_sha256':sha(__file__),'manifest_sha256':sha(manifest_path),'chain_metadata_sha256':sha(metadata),'selected':selected,'selected_count':256,'source_before':before,'source_after':after,'source_footprint_limitation':'Historical size/mtime/footer identity plus digest of all selected44-column raw rows; entire29GB parquet not hashed.','columns':cols,'sample_rows_per_chain':list(range(0,20000,20)),'sample_shape':list(values.shape),'selected_raw_float32_sha256':raw_hash.hexdigest(),'sample_array_sha256':hashlib.sha256(values.tobytes()).hexdigest(),'quantile_method':'float64 linear16quantiles inclusive endpoints, thenfloat32 canonical+0 dedup padding','knots_path':str(dest.relative_to(ROOT)),'knots_file_sha256':sha(dest),'knots_array_sha256':hashlib.sha256(knots.tobytes()).hexdigest(),'counts_array_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),'unique_counts':counts.tolist(),'targets_read':False,'masks_read':False,'validation_chains_read':False,'forecast_fits':0,'WP_queries':0,'wall_seconds':time.monotonic()-started,'max_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'affinity':sorted(os.sched_getaffinity(0))}
 (out/'fit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['status','knots_file_sha256','unique_counts','wall_seconds','max_rss_KiB']}))
if __name__=='__main__':main()
